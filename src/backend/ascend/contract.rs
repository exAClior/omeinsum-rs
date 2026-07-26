use super::{
    ffi::{checked_product, finish_pending, tensor, EnqueueAttempt, PendingOperation},
    normalize::normalize_repeated_labels,
    permute::{dense_permutation, enqueue as enqueue_permutation, materialize, DensePermutation},
    reduce::reduce_trace,
    runtime::Runtime,
    storage::AscendStorage,
    sys, AscendError,
};
use crate::backend::{
    contract_plan::{is_identity_materialization, plan_contraction},
    Storage,
};
use std::{collections::HashMap, ptr, sync::Arc};

const MAX_PIPELINE_OPERATIONS: usize = 4;

pub(crate) fn validate_operand(
    storage_len: usize,
    shape: &[usize],
    strides: &[usize],
    modes: &[i32],
) -> Result<(), AscendError> {
    if shape.len() != strides.len() || shape.len() != modes.len() {
        return Err(AscendError::status("Ascend metadata rank mismatch", -1));
    }
    let numel = checked_product(shape, "Ascend shape product overflow")?;
    if numel == 0 {
        return Ok(());
    }
    let max_offset = shape
        .iter()
        .zip(strides)
        .try_fold(0usize, |offset, (&dim, &stride)| {
            (dim - 1)
                .checked_mul(stride)
                .and_then(|term| offset.checked_add(term))
                .ok_or_else(|| AscendError::status("Ascend strided offset overflow", -1))
        })?;
    if max_offset >= storage_len {
        return Err(AscendError::status(
            "Ascend strided view exceeds storage",
            -1,
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn validate_metadata(
    a_len: usize,
    shape_a: &[usize],
    strides_a: &[usize],
    modes_a: &[i32],
    b_len: usize,
    shape_b: &[usize],
    strides_b: &[usize],
    modes_b: &[i32],
    shape_c: &[usize],
    modes_c: &[i32],
) -> Result<usize, AscendError> {
    validate_operand(a_len, shape_a, strides_a, modes_a)?;
    validate_operand(b_len, shape_b, strides_b, modes_b)?;
    if shape_c.len() != modes_c.len() {
        return Err(AscendError::status("Ascend output rank mismatch", -1));
    }
    if modes_c
        .iter()
        .enumerate()
        .any(|(i, mode)| modes_c[..i].contains(mode))
    {
        return Err(AscendError::status("Ascend repeated output mode", -1));
    }
    let mut extents = HashMap::new();
    for (&mode, &extent) in modes_a.iter().zip(shape_a) {
        extents.insert(mode, extent);
    }
    for (&mode, &extent) in modes_b.iter().zip(shape_b) {
        if extents.get(&mode).is_some_and(|&known| known != extent) {
            return Err(AscendError::status("Ascend mode extent mismatch", -1));
        }
        extents.insert(mode, extent);
    }
    for (&mode, &extent) in modes_c.iter().zip(shape_c) {
        if extents.get(&mode) != Some(&extent) {
            return Err(AscendError::status("Ascend output shape mismatch", -1));
        }
    }
    checked_product(shape_c, "Ascend output shape product overflow")
}

#[allow(clippy::too_many_arguments)]
fn enqueue_matmul(
    runtime: &Runtime,
    a: &AscendStorage<f32>,
    b: &AscendStorage<f32>,
    output: &AscendStorage<f32>,
    a_shape: &[usize],
    b_shape: &[usize],
    c_shape: &[usize],
    batched: bool,
) -> Result<EnqueueAttempt, AscendError> {
    let ta = tensor(a.as_mut_ptr(), a_shape)?;
    let tb = tensor(b.as_mut_ptr(), b_shape)?;
    let tc = tensor(output.as_mut_ptr(), c_shape)?;
    let mut workspace_size = 0u64;
    let mut executor = ptr::null_mut();
    let status = unsafe {
        if batched {
            sys::aclnnBatchMatMulGetWorkspaceSize(
                tb.0,
                ta.0,
                tc.0,
                0,
                &mut workspace_size,
                &mut executor,
            )
        } else {
            sys::aclnnMatmulGetWorkspaceSize(
                tb.0,
                ta.0,
                tc.0,
                0,
                &mut workspace_size,
                &mut executor,
            )
        }
    };
    if status != sys::ACL_SUCCESS {
        return Err(AscendError::status("aclnnMatmulGetWorkspaceSize", status));
    }
    if executor.is_null() {
        return Err(AscendError::null("aclnnMatmulGetWorkspaceSize executor"));
    }
    let bytes = usize::try_from(workspace_size)
        .map_err(|_| AscendError::status("aclnnMatmul workspace overflow", -1))?;
    let workspace = runtime.alloc_locked(bytes, false)?;
    let operation = PendingOperation::new(vec![ta, tb, tc], Vec::new(), workspace);
    let status = unsafe {
        if batched {
            sys::aclnnBatchMatMul(workspace, workspace_size, executor, runtime.stream)
        } else {
            sys::aclnnMatmul(workspace, workspace_size, executor, runtime.stream)
        }
    };
    let result = (status == sys::ACL_SUCCESS)
        .then_some(())
        .ok_or_else(|| AscendError::status("aclnnMatmul", status));
    Ok(EnqueueAttempt::new(operation, result))
}

fn record_attempt(
    pending: &mut Vec<PendingOperation>,
    attempt: EnqueueAttempt,
) -> Result<(), AscendError> {
    let (operation, result) = attempt.into_parts();
    pending.push(operation);
    result
}

fn finish_pipeline(
    runtime: &Runtime,
    pending: Vec<PendingOperation>,
    enqueue_result: Result<(), AscendError>,
) -> Result<(), AscendError> {
    finish_pending(
        pending,
        enqueue_result,
        || runtime.synchronize_locked(),
        |operation| operation.release(runtime),
    )
}

fn prepare_materialization(
    runtime: &Arc<Runtime>,
    source: &AscendStorage<f32>,
    shape: &[usize],
    strides: &[usize],
    permutation: &[usize],
) -> Result<(Option<AscendStorage<f32>>, Option<DensePermutation>), AscendError> {
    if is_identity_materialization(source.len(), shape, strides, permutation) {
        return Ok((None, None));
    }
    if let Some(plan) = dense_permutation(source.len(), shape, strides, permutation)? {
        let output = AscendStorage::allocate(runtime.clone(), source.len(), false)?;
        return Ok((Some(output), Some(plan)));
    }
    Ok((
        Some(materialize(runtime, source, shape, strides, permutation)?),
        None,
    ))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn contract(
    runtime: &Arc<Runtime>,
    a: &AscendStorage<f32>,
    shape_a: &[usize],
    strides_a: &[usize],
    modes_a: &[i32],
    b: &AscendStorage<f32>,
    shape_b: &[usize],
    strides_b: &[usize],
    modes_b: &[i32],
    shape_c: &[usize],
    modes_c: &[i32],
) -> Result<AscendStorage<f32>, AscendError> {
    if !Arc::ptr_eq(runtime, a.runtime()) || !Arc::ptr_eq(runtime, b.runtime()) {
        return Err(AscendError::status("Ascend runtime/device mismatch", -1));
    }
    if a.runtime().device != b.runtime().device {
        return Err(AscendError::status("Ascend device mismatch", -1));
    }
    let normalized_a = normalize_repeated_labels(runtime, a, shape_a, strides_a, modes_a)?;
    let normalized_b = normalize_repeated_labels(runtime, b, shape_b, strides_b, modes_b)?;
    let (a, shape_a, strides_a, modes_a) =
        normalized_a
            .as_ref()
            .map_or((a, shape_a, strides_a, modes_a), |operand| {
                (
                    &operand.storage,
                    operand.shape.as_slice(),
                    operand.strides.as_slice(),
                    operand.modes.as_slice(),
                )
            });
    let (b, shape_b, strides_b, modes_b) =
        normalized_b
            .as_ref()
            .map_or((b, shape_b, strides_b, modes_b), |operand| {
                (
                    &operand.storage,
                    operand.shape.as_slice(),
                    operand.strides.as_slice(),
                    operand.modes.as_slice(),
                )
            });
    let output_len = validate_metadata(
        a.len(),
        shape_a,
        strides_a,
        modes_a,
        b.len(),
        shape_b,
        strides_b,
        modes_b,
        shape_c,
        modes_c,
    )?;
    if output_len == 0 {
        return AscendStorage::allocate(runtime.clone(), 0, false);
    }
    let initial_plan = plan_contraction(modes_a, shape_a, modes_b, shape_b, modes_c);
    let reduced_a = (!initial_plan.left_trace.is_empty()).then(|| {
        reduce_trace(
            runtime,
            a,
            shape_a,
            strides_a,
            modes_a,
            &initial_plan.left_trace,
        )
    });
    let reduced_a = reduced_a.transpose()?;
    let reduced_b = (!initial_plan.right_trace.is_empty()).then(|| {
        reduce_trace(
            runtime,
            b,
            shape_b,
            strides_b,
            modes_b,
            &initial_plan.right_trace,
        )
    });
    let reduced_b = reduced_b.transpose()?;
    let (a, shape_a, strides_a, modes_a) =
        reduced_a
            .as_ref()
            .map_or((a, shape_a, strides_a, modes_a), |operand| {
                (
                    &operand.storage,
                    operand.shape.as_slice(),
                    operand.strides.as_slice(),
                    operand.modes.as_slice(),
                )
            });
    let (b, shape_b, strides_b, modes_b) =
        reduced_b
            .as_ref()
            .map_or((b, shape_b, strides_b, modes_b), |operand| {
                (
                    &operand.storage,
                    operand.shape.as_slice(),
                    operand.strides.as_slice(),
                    operand.modes.as_slice(),
                )
            });
    let plan = plan_contraction(modes_a, shape_a, modes_b, shape_b, modes_c);
    let zero_contract_extent = plan.contracted_modes.iter().any(|mode| {
        let axis = modes_a
            .iter()
            .position(|candidate| candidate == mode)
            .expect("planner contracted mode must belong to A");
        shape_a[axis] == 0
    });
    if zero_contract_extent {
        return AscendStorage::allocate(runtime.clone(), output_len.max(1), true);
    }

    let a_permutation = plan.a_permutation(modes_a);
    let b_permutation = plan.b_permutation(modes_b);
    let (canonical_a, a_device_plan) =
        prepare_materialization(runtime, a, shape_a, strides_a, &a_permutation)?;
    let (canonical_b, b_device_plan) =
        prepare_materialization(runtime, b, shape_b, strides_b, &b_permutation)?;
    let matmul_a = canonical_a.as_ref().unwrap_or(a);
    let matmul_b = canonical_b.as_ref().unwrap_or(b);
    let matmul_output = AscendStorage::allocate(runtime.clone(), output_len.max(1), false)?;
    let (a_shape, b_shape, c_shape, batched) = if plan.batch_size > 1 {
        (
            vec![plan.batch_size, plan.contract_size, plan.left_size],
            vec![plan.batch_size, plan.right_size, plan.contract_size],
            vec![plan.batch_size, plan.right_size, plan.left_size],
            true,
        )
    } else {
        (
            vec![plan.contract_size, plan.left_size],
            vec![plan.right_size, plan.contract_size],
            vec![plan.right_size, plan.left_size],
            false,
        )
    };
    let mut host_output_materialization = None;
    let output_device_plan = if let Some(perm) = plan.output_perm.as_ref() {
        let canonical_shape: Vec<usize> = plan
            .left_modes
            .iter()
            .chain(&plan.right_modes)
            .chain(&plan.batch_modes)
            .map(|mode| {
                let axis = modes_c
                    .iter()
                    .position(|candidate| candidate == mode)
                    .unwrap();
                shape_c[axis]
            })
            .collect();
        let mut stride = 1usize;
        let canonical_strides: Vec<usize> = canonical_shape
            .iter()
            .map(|&extent| {
                let current = stride;
                stride *= extent;
                current
            })
            .collect();
        let device_plan = dense_permutation(
            matmul_output.len(),
            &canonical_shape,
            &canonical_strides,
            perm,
        )?;
        if device_plan.is_none() {
            // aclnnPermute supports at most rank 8. Keep high-rank contractions
            // correct via the same host fallback used for operand views.
            host_output_materialization = Some((canonical_shape, canonical_strides));
        }
        device_plan
    } else {
        None
    };
    let final_output = output_device_plan
        .as_ref()
        .map(|_| AscendStorage::allocate(runtime.clone(), output_len.max(1), false))
        .transpose()?;

    runtime.with_transaction(|runtime| {
        // Reserve before the first launch so recording an attempted operation
        // cannot allocate after asynchronous stream work may have been submitted.
        let mut pending = Vec::with_capacity(MAX_PIPELINE_OPERATIONS);
        let enqueue_result = (|| {
            if let Some(device_plan) = a_device_plan.as_ref() {
                record_attempt(
                    &mut pending,
                    enqueue_permutation(
                        runtime,
                        a,
                        canonical_a
                            .as_ref()
                            .expect("device permutation must have an output"),
                        device_plan,
                    )?,
                )?;
            }
            if let Some(device_plan) = b_device_plan.as_ref() {
                record_attempt(
                    &mut pending,
                    enqueue_permutation(
                        runtime,
                        b,
                        canonical_b
                            .as_ref()
                            .expect("device permutation must have an output"),
                        device_plan,
                    )?,
                )?;
            }
            record_attempt(
                &mut pending,
                enqueue_matmul(
                    runtime,
                    matmul_a,
                    matmul_b,
                    &matmul_output,
                    &a_shape,
                    &b_shape,
                    &c_shape,
                    batched,
                )?,
            )?;
            if let Some(device_plan) = output_device_plan.as_ref() {
                record_attempt(
                    &mut pending,
                    enqueue_permutation(
                        runtime,
                        &matmul_output,
                        final_output
                            .as_ref()
                            .expect("output permutation must have an output"),
                        device_plan,
                    )?,
                )?;
            }
            Ok(())
        })();
        finish_pipeline(runtime, pending, enqueue_result)
    })?;

    if let Some(output) = final_output {
        Ok(output)
    } else if let Some((canonical_shape, canonical_strides)) = host_output_materialization {
        materialize(
            runtime,
            &matmul_output,
            &canonical_shape,
            &canonical_strides,
            plan.output_perm
                .as_ref()
                .expect("host output materialization requires a permutation"),
        )
    } else {
        Ok(matmul_output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failed_launch_is_recorded_before_error_propagates() {
        let operation = PendingOperation::new(Vec::new(), Vec::new(), ptr::null_mut());
        let attempt = EnqueueAttempt::new(
            operation,
            Err(AscendError::status("injected ACLNN launch failure", -1)),
        );
        let mut pending = Vec::with_capacity(1);

        assert!(record_attempt(&mut pending, attempt).is_err());
        assert_eq!(pending.len(), 1);
    }

    #[test]
    fn launch_error_synchronizes_and_releases_before_returning() {
        let launch_error = AscendError::status("injected ACLNN launch failure", -1);
        let mut synchronized = false;
        let mut released = Vec::new();

        let result = finish_pending(
            vec![1, 2],
            Err(launch_error.clone()),
            || {
                synchronized = true;
                Ok(())
            },
            |operation| released.push(operation),
        );

        assert_eq!(result, Err(launch_error));
        assert!(synchronized);
        assert_eq!(released, vec![1, 2]);
    }

    #[test]
    fn synchronization_error_retains_pending_resources() {
        let sync_error = AscendError::status("injected synchronization failure", -1);
        let mut released = Vec::new();

        let result = finish_pending(
            vec![1, 2],
            Err(AscendError::status("injected ACLNN launch failure", -1)),
            || Err(sync_error.clone()),
            |operation| released.push(operation),
        );

        assert_eq!(result, Err(sync_error));
        assert!(released.is_empty());
    }

    #[test]
    fn standalone_launch_error_synchronizes_before_release() {
        let launch_error = AscendError::status("injected ACLNN launch failure", -1);
        let events = std::cell::RefCell::new(Vec::new());

        let result = super::super::ffi::finish_operation(
            1,
            Err(launch_error.clone()),
            || {
                events.borrow_mut().push("synchronize");
                Ok(())
            },
            |_| events.borrow_mut().push("release"),
        );

        assert_eq!(result, Err(launch_error));
        assert_eq!(*events.borrow(), vec!["synchronize", "release"]);
    }
}

#![allow(dead_code)]

use std::collections::HashMap;

use approx::assert_relative_eq;
use num_complex::Complex64;
use omeco::NestedEinsum;
use omeinsum::realify::{merge_factor_tensors, mul_vertex_tensor};
use omeinsum::{
    realify_code, realify_data, realify_tree_code, split_re_im, Cpu, Einsum, RealifiedOutput,
    RealifyInput, RealifyTreePlan, Standard, Tensor,
};

#[derive(Clone)]
pub(super) enum OwnedInput {
    Real {
        data: Vec<f64>,
        shape: Vec<usize>,
    },
    Complex {
        data: Vec<Complex64>,
        shape: Vec<usize>,
        conjugate: bool,
    },
}

#[derive(Clone, Copy)]
pub(super) enum Execution {
    Unoptimized,
    Greedy,
    TreeSa,
}

impl OwnedInput {
    fn shape(&self) -> &[usize] {
        match self {
            OwnedInput::Real { shape, .. } | OwnedInput::Complex { shape, .. } => shape,
        }
    }

    fn is_complex(&self) -> bool {
        matches!(self, OwnedInput::Complex { .. })
    }

    fn as_realify_input(&self) -> RealifyInput<'_, f64> {
        match self {
            OwnedInput::Real { data, shape } => RealifyInput::Real { data, shape },
            OwnedInput::Complex {
                data,
                shape,
                conjugate,
            } => RealifyInput::Complex {
                data,
                shape,
                conjugate: *conjugate,
            },
        }
    }

    pub(super) fn native_tensor(&self) -> Tensor<Complex64, Cpu> {
        match self {
            OwnedInput::Real { data, shape } => {
                let complex: Vec<Complex64> = data
                    .iter()
                    .copied()
                    .map(|value| Complex64::new(value, 0.0))
                    .collect();
                Tensor::from_data(&complex, shape)
            }
            OwnedInput::Complex {
                data,
                shape,
                conjugate,
            } => {
                let effective: Vec<Complex64> = if *conjugate {
                    data.iter()
                        .map(|value| Complex64::new(value.re, -value.im))
                        .collect()
                } else {
                    data.clone()
                };
                Tensor::from_data(&effective, shape)
            }
        }
    }

    pub(super) fn realified_tensor(&self) -> Tensor<f64, Cpu> {
        match self {
            OwnedInput::Real { data, shape } => Tensor::from_data(data, shape),
            OwnedInput::Complex {
                data,
                shape,
                conjugate,
            } => {
                let realified = realify_data(data, *conjugate);
                let mut real_shape = shape.clone();
                real_shape.push(2);
                Tensor::from_data(&realified, &real_shape)
            }
        }
    }
}

pub(super) fn infer_sizes(inputs: &[OwnedInput], ixs: &[Vec<usize>]) -> HashMap<usize, usize> {
    assert_eq!(inputs.len(), ixs.len());
    let mut sizes = HashMap::new();
    for (input, ix) in inputs.iter().zip(ixs.iter()) {
        assert_eq!(input.shape().len(), ix.len());
        for (&label, &size) in ix.iter().zip(input.shape().iter()) {
            if let Some(existing) = sizes.insert(label, size) {
                assert_eq!(existing, size, "label {label} has inconsistent sizes");
            }
        }
    }
    sizes
}

pub(super) fn execute_native(
    inputs: &[OwnedInput],
    ixs: &[Vec<usize>],
    iy: &[usize],
    execution: Execution,
) -> Tensor<Complex64, Cpu> {
    let tensors: Vec<Tensor<Complex64, Cpu>> =
        inputs.iter().map(OwnedInput::native_tensor).collect();
    let tensor_refs: Vec<&Tensor<Complex64, Cpu>> = tensors.iter().collect();
    let mut einsum = Einsum::new(ixs.to_vec(), iy.to_vec(), infer_sizes(inputs, ixs));
    optimize(&mut einsum, execution);
    einsum.execute::<Standard<Complex64>, Complex64, Cpu>(&tensor_refs)
}

pub(super) fn execute_native_tree(
    inputs: &[OwnedInput],
    ixs: &[Vec<usize>],
    iy: &[usize],
    tree: NestedEinsum<usize>,
) -> Tensor<Complex64, Cpu> {
    let tensors: Vec<Tensor<Complex64, Cpu>> =
        inputs.iter().map(OwnedInput::native_tensor).collect();
    let tensor_refs: Vec<&Tensor<Complex64, Cpu>> = tensors.iter().collect();
    let mut einsum = Einsum::new(ixs.to_vec(), iy.to_vec(), infer_sizes(inputs, ixs));
    einsum.set_contraction_tree(tree);
    einsum.execute::<Standard<Complex64>, Complex64, Cpu>(&tensor_refs)
}

pub(super) fn prepare_tree_realified(
    inputs: &[OwnedInput],
    ixs: &[Vec<usize>],
    iy: &[usize],
    tree: &NestedEinsum<usize>,
) -> (RealifyTreePlan, Vec<Tensor<f64, Cpu>>) {
    let is_complex: Vec<bool> = inputs.iter().map(OwnedInput::is_complex).collect();
    let plan = realify_tree_code(tree, ixs, iy, &infer_sizes(inputs, ixs), &is_complex);
    let mut tensors: Vec<Tensor<f64, Cpu>> =
        inputs.iter().map(OwnedInput::realified_tensor).collect();

    for &(u_position, v_position, w_position) in &plan.factor_vertex_positions {
        assert_eq!(u_position, tensors.len());
        assert_eq!(v_position, u_position + 1);
        assert_eq!(w_position, v_position + 1);
        let (u, w) = merge_factor_tensors::<f64, Cpu>(Cpu);
        tensors.push(u.clone());
        tensors.push(u);
        tensors.push(w);
    }
    assert_eq!(tensors.len(), plan.einsum.ixs.len());
    (plan, tensors)
}

pub(super) fn execute_tree_realified(
    inputs: &[OwnedInput],
    ixs: &[Vec<usize>],
    iy: &[usize],
    tree: &NestedEinsum<usize>,
) -> (Tensor<f64, Cpu>, RealifiedOutput) {
    let (plan, tensors) = prepare_tree_realified(inputs, ixs, iy, tree);
    let output = plan.output;
    let tensor_refs: Vec<&Tensor<f64, Cpu>> = tensors.iter().collect();
    (
        plan.einsum.execute::<Standard<f64>, f64, Cpu>(&tensor_refs),
        output,
    )
}

pub(super) fn execute_realified(
    inputs: &[OwnedInput],
    ixs: &[Vec<usize>],
    iy: &[usize],
    execution: Execution,
) -> (Tensor<f64, Cpu>, RealifiedOutput) {
    let is_complex: Vec<bool> = inputs.iter().map(OwnedInput::is_complex).collect();
    let mut plan = realify_code(ixs, iy, &infer_sizes(inputs, ixs), &is_complex);
    let output = plan.output;

    let mut tensors: Vec<Tensor<f64, Cpu>> =
        inputs.iter().map(OwnedInput::realified_tensor).collect();
    for _ in 0..plan.num_mul_vertices {
        tensors.push(mul_vertex_tensor::<f64, Cpu>(Cpu));
    }

    optimize(&mut plan.einsum, execution);
    let tensor_refs: Vec<&Tensor<f64, Cpu>> = tensors.iter().collect();
    (
        plan.einsum.execute::<Standard<f64>, f64, Cpu>(&tensor_refs),
        output,
    )
}

fn optimize(einsum: &mut Einsum<usize>, execution: Execution) {
    match execution {
        Execution::Unoptimized => {}
        Execution::Greedy => {
            einsum.optimize_greedy();
        }
        Execution::TreeSa => {
            einsum.optimize_treesa();
        }
    }
}

pub(super) fn assert_realified_matches_native(
    inputs: &[OwnedInput],
    ixs: &[Vec<usize>],
    iy: &[usize],
    execution: Execution,
    tolerance: f64,
) {
    let native = execute_native(inputs, ixs, iy, execution);
    let (realified, output) = execute_realified(inputs, ixs, iy, execution);

    assert_shape_matches_output(&realified, output, native.shape());
    let actual = recover_complex(&realified, output);
    assert_complex_vec_close(&actual, &native.to_vec(), tolerance);
}

pub(super) fn assert_shape_matches_output(
    realified: &Tensor<f64, Cpu>,
    output: RealifiedOutput,
    native_shape: &[usize],
) {
    match output {
        RealifiedOutput::Real => assert_eq!(realified.shape(), native_shape),
        RealifiedOutput::ReImAxis => {
            let mut expected = native_shape.to_vec();
            expected.push(2);
            assert_eq!(realified.shape(), expected);
        }
    }
}

pub(super) fn recover_complex(
    result: &Tensor<f64, Cpu>,
    output: RealifiedOutput,
) -> Vec<Complex64> {
    match output {
        RealifiedOutput::Real => result
            .to_vec()
            .into_iter()
            .map(|value| Complex64::new(value, 0.0))
            .collect(),
        RealifiedOutput::ReImAxis => {
            let (re, im) = split_re_im(result);
            re.into_iter()
                .zip(im)
                .map(|(re, im)| Complex64::new(re, im))
                .collect()
        }
    }
}

pub(super) fn assert_complex_vec_close(
    actual: &[Complex64],
    expected: &[Complex64],
    tolerance: f64,
) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_relative_eq!(
            actual.re,
            expected.re,
            epsilon = tolerance,
            max_relative = tolerance,
        );
        assert_relative_eq!(
            actual.im,
            expected.im,
            epsilon = tolerance,
            max_relative = tolerance,
        );
        assert!(
            actual.re.is_finite() && actual.im.is_finite(),
            "non-finite result at output index {index}: {actual:?}"
        );
    }
}

pub(super) fn patterned_complex(len: usize, seed: usize) -> Vec<Complex64> {
    (0..len)
        .map(|index| {
            let real = ((index.wrapping_mul(17) + seed.wrapping_mul(31)) % 257) as f64;
            let imag = ((index.wrapping_mul(29) + seed.wrapping_mul(13)) % 251) as f64;
            Complex64::new((real - 128.0) / 37.0, (imag - 125.0) / 41.0)
        })
        .collect()
}

pub(super) fn patterned_real(len: usize, seed: usize) -> Vec<f64> {
    (0..len)
        .map(|index| {
            let value = ((index.wrapping_mul(19) + seed.wrapping_mul(23)) % 199) as f64;
            (value - 97.0) / 29.0
        })
        .collect()
}

pub(super) fn borrowed_inputs(inputs: &[OwnedInput]) -> Vec<RealifyInput<'_, f64>> {
    inputs.iter().map(OwnedInput::as_realify_input).collect()
}

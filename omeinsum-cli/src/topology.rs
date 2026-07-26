use std::collections::HashMap;

use omeco::NestedEinsum;
use omeinsum::algebra::Scalar;
use omeinsum::{Cpu, Tensor};

pub(crate) fn validate_labels(
    ixs: &[Vec<usize>],
    iy: &[usize],
    size_dict: &HashMap<usize, usize>,
) -> Result<(), String> {
    for &label in ixs.iter().flatten().chain(iy.iter()) {
        if !size_dict.contains_key(&label) {
            return Err(format!(
                "Missing size for label index {label} referenced by topology expression"
            ));
        }
    }
    Ok(())
}

pub(crate) fn validate_tensor_dimensions<T>(
    tensors: &[&Tensor<T, Cpu>],
    ixs: &[Vec<usize>],
    size_dict: &HashMap<usize, usize>,
) -> Result<(), String>
where
    T: Scalar,
{
    for (tensor_index, (tensor, labels)) in tensors.iter().zip(ixs).enumerate() {
        if tensor.ndim() != labels.len() {
            return Err(format!(
                "Tensor {tensor_index} has {} dims but topology expression specifies {} labels",
                tensor.ndim(),
                labels.len()
            ));
        }
        for (axis, &label) in labels.iter().enumerate() {
            let expected = size_dict[&label];
            let actual = tensor.shape()[axis];
            if actual != expected {
                return Err(format!(
                    "Tensor {tensor_index} axis {axis} has size {actual}, but topology label index {label} has size {expected}"
                ));
            }
        }
    }
    Ok(())
}

pub(crate) fn validate_tree(
    tree: &NestedEinsum<usize>,
    source_ixs: &[Vec<usize>],
    size_dict: &HashMap<usize, usize>,
    leaf_counts: &mut [usize],
) -> Result<Vec<usize>, String> {
    match tree {
        NestedEinsum::Leaf { tensor_index } => {
            if *tensor_index >= source_ixs.len() {
                return Err(format!(
                    "Topology leaf tensor_index {tensor_index} out of range for {} tensors",
                    source_ixs.len()
                ));
            }
            leaf_counts[*tensor_index] += 1;
            Ok(source_ixs[*tensor_index].clone())
        }
        NestedEinsum::Node { args, eins } => {
            if args.len() != 2 {
                return Err(format!(
                    "Topology tree must be binary, found node with {} children",
                    args.len()
                ));
            }
            if eins.ixs.len() != 2 {
                return Err(format!(
                    "Topology binary node must contain two input index lists, found {}",
                    eins.ixs.len()
                ));
            }
            for label in eins.ixs.iter().flatten().chain(eins.iy.iter()) {
                if !size_dict.contains_key(label) {
                    return Err(format!("Topology references unknown label index {label}"));
                }
            }
            let child_outputs = args
                .iter()
                .map(|arg| validate_tree(arg, source_ixs, size_dict, leaf_counts))
                .collect::<Result<Vec<_>, _>>()?;
            if eins.ixs != child_outputs {
                return Err(format!(
                    "Topology node input labels {:?} do not match child outputs {:?}",
                    eins.ixs, child_outputs
                ));
            }
            Ok(eins.iy.clone())
        }
    }
}

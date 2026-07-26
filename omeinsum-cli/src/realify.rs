use num_complex::Complex;
use num_traits::Float;
use omeinsum::algebra::Scalar;
use omeinsum::{
    merge_factor_tensors, realify_code, realify_data, realify_tree_code, BackendScalar, Cpu,
    Einsum, Tensor,
};

use crate::common::{build_explicit_einsum, load_complex_tensors};
use crate::format::TensorsFile;

#[derive(Clone, Copy)]
pub(crate) enum RealifyStrategy {
    Cascade,
    Tree,
}

pub(crate) struct PreparedRealify<T: Scalar> {
    pub(crate) einsum: Einsum<usize>,
    pub(crate) tensors: Vec<Tensor<T, Cpu>>,
    pub(crate) source_input_count: usize,
    pub(crate) source_output_shape: Vec<usize>,
}

pub(crate) fn prepare<T>(
    tensors_file: &TensorsFile,
    topology_path: Option<&str>,
    expr: Option<&str>,
    strategy: RealifyStrategy,
    make_complex: fn(f64, f64) -> Complex<T>,
) -> Result<PreparedRealify<T>, String>
where
    T: Scalar + Float + BackendScalar<Cpu>,
    Complex<T>: Scalar + BackendScalar<Cpu>,
{
    let complex_tensors = load_complex_tensors(tensors_file, make_complex)?;
    let complex_refs: Vec<&Tensor<Complex<T>, Cpu>> = complex_tensors.iter().collect();
    let source = build_explicit_einsum(&complex_refs, topology_path, expr)?;
    let source_output_shape = source
        .iy
        .iter()
        .map(|label| source.size_dict[label])
        .collect();
    let source_input_count = complex_tensors.len();
    let is_complex = vec![true; source_input_count];
    let mut tensors = complex_tensors
        .iter()
        .map(realify_tensor)
        .collect::<Vec<_>>();

    let einsum = match strategy {
        RealifyStrategy::Cascade => {
            let mut plan = realify_code(&source.ixs, &source.iy, &source.size_dict, &is_complex);
            for _ in 0..plan.num_mul_vertices {
                tensors.push(omeinsum::realify::mul_vertex_tensor::<T, Cpu>(Cpu));
            }

            // Source trees have no leaves for inserted dense multiplication
            // vertices, so the legacy cascade is planned independently.
            plan.einsum.optimize_greedy();
            plan.einsum
        }
        RealifyStrategy::Tree => {
            let source_tree = source
                .contraction_tree()
                .expect("explicit CLI einsum must carry a contraction tree");
            let plan = realify_tree_code(
                source_tree,
                &source.ixs,
                &source.iy,
                &source.size_dict,
                &is_complex,
            );
            for &(u_position, v_position, w_position) in &plan.factor_vertex_positions {
                if (u_position, v_position, w_position)
                    != (tensors.len(), tensors.len() + 1, tensors.len() + 2)
                {
                    return Err("Tree-realified factor positions are not contiguous".to_string());
                }
                let (u, w) = merge_factor_tensors::<T, Cpu>(Cpu);
                tensors.push(u.clone());
                tensors.push(u);
                tensors.push(w);
            }
            plan.einsum
        }
    };

    debug_assert_eq!(tensors.len(), einsum.ixs.len());
    Ok(PreparedRealify {
        einsum,
        tensors,
        source_input_count,
        source_output_shape,
    })
}

pub(crate) fn realify_tensor<T>(tensor: &Tensor<Complex<T>, Cpu>) -> Tensor<T, Cpu>
where
    T: Scalar + Float + BackendScalar<Cpu>,
    Complex<T>: Scalar + BackendScalar<Cpu>,
{
    let data = realify_data(&tensor.to_vec(), false);
    let mut shape = tensor.shape().to_vec();
    shape.push(2);
    Tensor::from_data(&data, &shape)
}

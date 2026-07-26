//! Complex-to-real tensor network conversion.
//!
//! Realification rewrites a complex-valued einsum into an ordinary real-valued
//! einsum by appending one trailing dimension-2 Re/Im leg to every genuinely
//! complex input and merging those legs with a fixed multiplication tensor. The
//! transformed network can run through the existing [`Standard`](crate::Standard)
//! algebra and every existing backend that supports the chosen real scalar type.

use std::collections::{HashMap, HashSet};

use num_complex::Complex;
use num_traits::Float;

use crate::algebra::{Scalar, Standard};
use crate::backend::{Backend, BackendScalar};
use crate::einsum::Einsum;
use crate::tensor::Tensor;

mod tree;

pub use tree::{realify_tree_code, RealifyTreePlan};

/// Constant tensors for realified complex arithmetic.
///
/// Data is column-major. The rank-3 tensors have shape `[2, 2, 2]`; matrices have
/// shape `[2, 2]`; vectors have shape `[2]`.
pub mod constants {
    /// Multiplication vertex `M`: `result[c] = Σ_ab x[a] y[b] M[a,b,c]`.
    ///
    /// Shape `[2, 2, 2]`; flat index is `a + 2*b + 4*c`.
    pub const M_DATA: [f64; 8] = [1.0, 0.0, 0.0, -1.0, 0.0, 1.0, 1.0, 0.0];

    /// Fully symmetric conjugated-output multiplication tensor `𝒞 = M·Z`.
    ///
    /// This is exported for tests and graphical identities. Runtime realification
    /// inserts [`M_DATA`] instead, which has the trailing `Z` pre-absorbed.
    pub const C_DATA: [f64; 8] = [1.0, 0.0, 0.0, -1.0, 0.0, -1.0, -1.0, 0.0];

    /// First input factor of the rank-3 multiplication decomposition.
    ///
    /// Shape `[2, 3]`; columns are the linear forms `x0`, `x1`, and `x0 + x1`.
    pub const U_DATA: [f64; 6] = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0];

    /// Second input factor. Complex multiplication uses the same forms for both inputs.
    pub const V_DATA: [f64; 6] = U_DATA;

    /// Output factor of the rank-3 multiplication decomposition.
    ///
    /// Shape `[2, 3]`; `Re = p0 - p1`, `Im = p2 - p0 - p1`.
    pub const W_DATA: [f64; 6] = [1.0, -1.0, -1.0, -1.0, 0.0, 1.0];

    /// Conjugation matrix `Z = diag(1, -1)`.
    pub const Z_DATA: [f64; 4] = [1.0, 0.0, 0.0, -1.0];

    /// Real-unit vector `e0 = [1, 0]`, which pins an extra leg to the real axis.
    pub const E0_DATA: [f64; 2] = [1.0, 0.0];

    /// Phase rotation matrix for multiplication by `exp(i*phi)`.
    ///
    /// Data is column-major for `[[cos(phi), -sin(phi)], [sin(phi), cos(phi)]]`.
    pub fn r_phi(phi: f64) -> [f64; 4] {
        let (sin_phi, cos_phi) = phi.sin_cos();
        [cos_phi, sin_phi, -sin_phi, cos_phi]
    }
}

/// Declares one host-side input of the source complex network.
///
/// `T` is the real scalar type of the realified network (`f32` or `f64`). Real
/// inputs are uploaded unchanged; complex inputs are converted from column-major
/// `Complex<T>` storage into `[Re block; Im block]` storage with one trailing
/// dimension-2 axis.
#[derive(Clone, Copy, Debug)]
pub enum RealifyInput<'a, T> {
    /// Already-real tensor: passes through unchanged and gets no extra leg.
    Real {
        /// Column-major tensor data.
        data: &'a [T],
        /// Tensor shape matching the source einsum input labels.
        shape: &'a [usize],
    },
    /// Complex tensor with optional conjugation applied during conversion.
    Complex {
        /// Column-major complex tensor data.
        data: &'a [Complex<T>],
        /// Tensor shape matching the source einsum input labels.
        shape: &'a [usize],
        /// If true, negate the imaginary block during conversion.
        conjugate: bool,
    },
}

impl<T> RealifyInput<'_, T> {
    fn shape(&self) -> &[usize] {
        match self {
            RealifyInput::Real { shape, .. } | RealifyInput::Complex { shape, .. } => shape,
        }
    }

    fn is_complex(&self) -> bool {
        matches!(self, RealifyInput::Complex { .. })
    }
}

/// Output descriptor for a realified network.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RealifiedOutput {
    /// No complex inputs: the result is the plain real einsum result.
    Real,
    /// The result carries a trailing dimension-2 axis: index 0 is Re, index 1 is Im.
    ReImAxis,
}

/// Result of the pure topology transform.
///
/// This is labels and sizes only. Tensor data conversion is deliberately separate
/// so callers can plan once and upload real tensors to whatever backend they use.
pub struct RealifyPlan {
    /// The realified einsum spec, before optimization.
    pub einsum: Einsum<usize>,
    /// Number of inserted multiplication vertices.
    pub num_mul_vertices: usize,
    /// Positions of inserted multiplication vertices in the new tensor list.
    pub mul_vertex_positions: Vec<usize>,
    /// Whether the transformed output has a trailing Re/Im axis.
    pub output: RealifiedOutput,
}

/// Pure complex-to-real topology transform.
///
/// `is_complex[k]` says whether input `k` gets an appended Re/Im leg. Fresh labels
/// start at `max(ixs ∪ iy ∪ size_dict.keys()) + 1`; every extra and merge label has
/// size 2 in the returned [`Einsum`].
pub fn realify_code(
    ixs: &[Vec<usize>],
    iy: &[usize],
    size_dict: &HashMap<usize, usize>,
    is_complex: &[bool],
) -> RealifyPlan {
    assert_eq!(
        ixs.len(),
        is_complex.len(),
        "is_complex length {} must match number of inputs {}",
        is_complex.len(),
        ixs.len()
    );

    let complex_count = is_complex.iter().filter(|&&complex| complex).count();
    if complex_count == 0 {
        return RealifyPlan {
            einsum: Einsum::new(ixs.to_vec(), iy.to_vec(), size_dict.clone()),
            num_mul_vertices: 0,
            mul_vertex_positions: Vec::new(),
            output: RealifiedOutput::Real,
        };
    }

    let fresh_count = complex_count
        .checked_mul(2)
        .and_then(|count| count.checked_sub(1))
        .expect("too many complex inputs to allocate fresh labels");
    let fresh_labels = allocate_fresh_labels(ixs, iy, size_dict, fresh_count);
    let mut fresh_labels = fresh_labels.into_iter();

    let mut realified_ixs = Vec::with_capacity(ixs.len() + complex_count.saturating_sub(1));
    let mut realified_sizes = size_dict.clone();
    let mut complex_legs = Vec::new();

    for (ix, &complex) in ixs.iter().zip(is_complex.iter()) {
        let mut new_ix = ix.clone();
        if complex {
            let extra_label = fresh_labels
                .next()
                .expect("fresh label preallocation underflow");
            new_ix.push(extra_label);
            realified_sizes.insert(extra_label, 2);
            complex_legs.push(extra_label);
        }
        realified_ixs.push(new_ix);
    }

    let mut realified_iy = iy.to_vec();
    let mut mul_vertex_positions = Vec::new();
    let output = match complex_legs.as_slice() {
        [] => RealifiedOutput::Real,
        [only] => {
            realified_iy.push(*only);
            RealifiedOutput::ReImAxis
        }
        [first, rest @ ..] => {
            let mut carried = *first;
            for &next_complex_leg in rest {
                let merged = fresh_labels
                    .next()
                    .expect("fresh label preallocation underflow");
                realified_sizes.insert(merged, 2);
                realified_ixs.push(vec![carried, next_complex_leg, merged]);
                mul_vertex_positions.push(realified_ixs.len() - 1);
                carried = merged;
            }
            realified_iy.push(carried);
            RealifiedOutput::ReImAxis
        }
    };

    debug_assert!(fresh_labels.next().is_none());

    RealifyPlan {
        einsum: Einsum::new(realified_ixs, realified_iy, realified_sizes),
        num_mul_vertices: mul_vertex_positions.len(),
        mul_vertex_positions,
        output,
    }
}

fn allocate_fresh_labels(
    ixs: &[Vec<usize>],
    iy: &[usize],
    size_dict: &HashMap<usize, usize>,
    count: usize,
) -> Vec<usize> {
    debug_assert!(count > 0);
    let mut used: HashSet<usize> = ixs
        .iter()
        .flatten()
        .chain(iy.iter())
        .chain(size_dict.keys())
        .copied()
        .collect();
    let mut next_high = used
        .iter()
        .copied()
        .max()
        .and_then(|label| label.checked_add(1));
    let mut next_hole = 0usize;
    let mut labels = Vec::with_capacity(count);

    while labels.len() < count {
        let label = if let Some(high) = next_high {
            next_high = high.checked_add(1);
            high
        } else {
            while used.contains(&next_hole) {
                next_hole = next_hole
                    .checked_add(1)
                    .expect("not enough usize labels to realify network");
            }
            let hole = next_hole;
            next_hole = next_hole.checked_add(1).unwrap_or(next_hole);
            hole
        };
        assert!(
            used.insert(label),
            "not enough usize labels to realify network"
        );
        labels.push(label);
    }

    labels
}

/// Convert one complex tensor to realified column-major storage.
///
/// The output is `[all Re entries in source order; all Im entries in source order]`.
/// If `conjugate` is true, the imaginary block is negated while being written.
pub fn realify_data<T>(data: &[Complex<T>], conjugate: bool) -> Vec<T>
where
    T: Scalar + Float,
{
    let mut out = Vec::with_capacity(data.len() * 2);
    out.extend(data.iter().map(|value| value.re));

    if conjugate {
        out.extend(data.iter().map(|value| -value.im));
    } else {
        out.extend(data.iter().map(|value| value.im));
    }

    out
}

/// Build the real multiplication vertex tensor on a backend.
pub fn mul_vertex_tensor<T, B>(backend: B) -> Tensor<T, B>
where
    T: Scalar + Float,
    B: Backend,
{
    let data = constants::M_DATA.map(cast_f64::<T>);
    Tensor::from_data_with_backend(&data, &[2, 2, 2], backend)
}

/// Build the `U` (= `V`) and `W` tensors for a factorized tree merge.
///
/// Both tensors have shape `[2, 3]`. A merge uses the returned `U` tensor twice,
/// once at each input position, and the `W` tensor once.
pub fn merge_factor_tensors<T, B>(backend: B) -> (Tensor<T, B>, Tensor<T, B>)
where
    T: Scalar + Float,
    B: Backend,
{
    let u = constants::U_DATA.map(cast_f64::<T>);
    let w = constants::W_DATA.map(cast_f64::<T>);
    (
        Tensor::from_data_with_backend(&u, &[2, 3], backend.clone()),
        Tensor::from_data_with_backend(&w, &[2, 3], backend),
    )
}

/// One-shot realified einsum on a chosen backend.
///
/// This plans the topology, converts host input data, uploads all real tensors,
/// greedily optimizes the transformed network, executes it with `Standard<T>`, and
/// returns the raw real result plus the output descriptor. For real-only networks
/// the output descriptor is [`RealifiedOutput::Real`] and no trailing axis is added.
pub fn realify_einsum<T, B>(
    inputs: &[RealifyInput<'_, T>],
    ixs: &[Vec<usize>],
    iy: &[usize],
    backend: B,
) -> (Tensor<T, B>, RealifiedOutput)
where
    T: Scalar + Float + BackendScalar<B>,
    B: Backend,
{
    let size_dict = infer_size_dict(inputs, ixs);
    let is_complex: Vec<bool> = inputs.iter().map(RealifyInput::is_complex).collect();
    let plan = realify_code(ixs, iy, &size_dict, &is_complex);
    let output = plan.output;

    let mut tensors = Vec::with_capacity(inputs.len() + plan.num_mul_vertices);
    for input in inputs {
        match input {
            RealifyInput::Real { data, shape } => {
                assert_data_len(data, shape);
                tensors.push(Tensor::from_data_with_backend(data, shape, backend.clone()));
            }
            RealifyInput::Complex {
                data,
                shape,
                conjugate,
            } => {
                assert_data_len(data, shape);
                let real_data = realify_data(data, *conjugate);
                let mut real_shape = shape.to_vec();
                real_shape.push(2);
                tensors.push(Tensor::from_data_with_backend(
                    &real_data,
                    &real_shape,
                    backend.clone(),
                ));
            }
        }
    }

    for _ in 0..plan.num_mul_vertices {
        tensors.push(mul_vertex_tensor::<T, B>(backend.clone()));
    }

    let tensor_refs: Vec<&Tensor<T, B>> = tensors.iter().collect();
    let mut einsum = plan.einsum;
    einsum.optimize_greedy();
    (einsum.execute::<Standard<T>, T, B>(&tensor_refs), output)
}

fn infer_size_dict<T>(inputs: &[RealifyInput<'_, T>], ixs: &[Vec<usize>]) -> HashMap<usize, usize> {
    assert_eq!(
        inputs.len(),
        ixs.len(),
        "number of inputs {} must match number of index specs {}",
        inputs.len(),
        ixs.len()
    );

    let mut size_dict = HashMap::new();
    for (input, ix) in inputs.iter().zip(ixs.iter()) {
        let shape = input.shape();
        assert_eq!(
            shape.len(),
            ix.len(),
            "index count {} must match tensor rank {}",
            ix.len(),
            shape.len()
        );

        for (&label, &size) in ix.iter().zip(shape.iter()) {
            if let Some(existing) = size_dict.insert(label, size) {
                assert_eq!(
                    existing, size,
                    "inconsistent size for label {label}: {existing} vs {size}"
                );
            }
        }
    }
    size_dict
}

fn assert_data_len<T>(data: &[T], shape: &[usize]) {
    let expected = shape.iter().product::<usize>();
    assert_eq!(
        data.len(),
        expected,
        "data length {} must match shape {:?} product {}",
        data.len(),
        shape,
        expected
    );
}

/// Split a tensor with a trailing Re/Im axis into host-side real and imaginary blocks.
///
/// The input shape must end in dimension 2. Returned vectors have the original
/// output shape with that trailing axis removed.
pub fn split_re_im<T, B>(result: &Tensor<T, B>) -> (Vec<T>, Vec<T>)
where
    T: Scalar,
    B: Backend,
{
    assert_eq!(
        result.shape().last().copied(),
        Some(2),
        "realified complex result must have trailing dimension 2, got shape {:?}",
        result.shape()
    );

    let data = result.to_vec();
    let block_len = data.len() / 2;
    (data[..block_len].to_vec(), data[block_len..].to_vec())
}

/// Download a tensor with a trailing Re/Im axis and reconstruct complex values.
///
/// Unlike [`split_re_im`], this allocates one additional complex output rather than
/// two separate split-output vectors. Device execution remains real-valued until
/// this call.
pub fn recover_complex<T, B>(result: &Tensor<T, B>) -> Vec<Complex<T>>
where
    T: Scalar,
    B: Backend,
{
    assert_eq!(
        result.shape().last().copied(),
        Some(2),
        "realified complex result must have trailing dimension 2, got shape {:?}",
        result.shape()
    );

    let data = result.to_vec();
    let block_len = data.len() / 2;
    data[..block_len]
        .iter()
        .copied()
        .zip(data[block_len..].iter().copied())
        .map(|(re, im)| Complex::new(re, im))
        .collect()
}

fn cast_f64<T: Float>(value: f64) -> T {
    num_traits::cast(value).expect("realify constants are representable as f32/f64")
}

#[cfg(test)]
mod tests;

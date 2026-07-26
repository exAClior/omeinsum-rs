use super::format::{slice_column_major, BenchmarkNetwork, TreeNode};
use num_complex::{Complex32, Complex64};
use num_traits::{One, Zero};
use omeco::{EinCode, NestedEinsum};
use omeinsum::algebra::Scalar;
use omeinsum::realify::{merge_factor_tensors, mul_vertex_tensor};
use omeinsum::{
    realify_code, realify_tree_code, Backend, BackendScalar, Einsum, RealifiedOutput, Standard,
    Tensor,
};
use std::collections::HashMap;

type RealValue<B> = (Tensor<f32, B>, Option<Tensor<f32, B>>);

pub fn assignments(network: &BenchmarkNetwork) -> Vec<HashMap<usize, usize>> {
    (0..network.assignment_count)
        .map(|mut n| {
            let mut a = HashMap::new();
            for &cut in network.cuts.iter().rev() {
                let extent = network.size_dict[&cut];
                a.insert(cut, n % extent);
                n /= extent;
            }
            a
        })
        .collect()
}

pub fn real_cache<B: Backend + Clone>(
    network: &BenchmarkNetwork,
    backend: B,
) -> Vec<Vec<RealValue<B>>>
where
    f32: BackendScalar<B>,
{
    network
        .tensors
        .iter()
        .zip(&network.eincode.input_indices)
        .map(|(t, ix)| {
            local_assignments(network, ix)
                .iter()
                .map(|a| {
                    let (re, shape) = slice_column_major(&t.data_re, &t.shape, ix, a);
                    let r = Tensor::from_data_with_backend(&re, &shape, backend.clone());
                    let im = t.structurally_complex.then(|| {
                        let (v, _) = slice_column_major(&t.data_im, &t.shape, ix, a);
                        Tensor::from_data_with_backend(&v, &shape, backend.clone())
                    });
                    (r, im)
                })
                .collect()
        })
        .collect()
}

/// Prepared static real executor. Source tensors are cached per local slice;
/// generated multiplication factors are shared by every physical assignment.
pub struct StaticReal<B: Backend> {
    einsum: Einsum<usize>,
    source_cache: Vec<Vec<Tensor<f32, B>>>,
    factors: Vec<Tensor<f32, B>>,
    output: RealifiedOutput,
}

pub fn static_factorized<B: Backend + Clone>(
    network: &BenchmarkNetwork,
    backend: B,
) -> StaticReal<B>
where
    f32: BackendScalar<B>,
{
    let source_tree = network.contraction_order.to_nested();
    let is_complex = network
        .tensors
        .iter()
        .map(|tensor| tensor.structurally_complex)
        .collect::<Vec<_>>();
    let plan = realify_tree_code(
        &source_tree,
        &network.eincode.input_indices,
        &network.eincode.output_indices,
        &physical_size_dict(network),
        &is_complex,
    );
    let mut factors = Vec::with_capacity(plan.factor_vertex_positions.len() * 3);
    for &(u_position, v_position, w_position) in &plan.factor_vertex_positions {
        let source_count = network.tensors.len();
        assert_eq!(u_position, source_count + factors.len());
        assert_eq!(v_position, u_position + 1);
        assert_eq!(w_position, v_position + 1);
        let (u, w) = merge_factor_tensors::<f32, B>(backend.clone());
        factors.push(u.clone());
        factors.push(u);
        factors.push(w);
    }
    StaticReal {
        einsum: plan.einsum,
        source_cache: static_source_cache(network, backend),
        factors,
        output: plan.output,
    }
}

/// Build the deliberately dense, tree-following baseline used to expose the 4×
/// binary temporary. This is benchmark-only; production callers should use
/// [`static_factorized`].
pub fn static_dense<B: Backend + Clone>(network: &BenchmarkNetwork, backend: B) -> StaticReal<B>
where
    f32: BackendScalar<B>,
{
    let is_complex = network
        .tensors
        .iter()
        .map(|tensor| tensor.structurally_complex)
        .collect::<Vec<_>>();
    let (einsum, factor_positions, output) = dense_tree_code(network, &is_complex);
    let factors = factor_positions
        .iter()
        .map(|_| mul_vertex_tensor::<f32, B>(backend.clone()))
        .collect();
    StaticReal {
        einsum,
        source_cache: static_source_cache(network, backend),
        factors,
        output,
    }
}

pub fn solve_static<B: Backend + Clone>(
    network: &BenchmarkNetwork,
    prepared: &StaticReal<B>,
    backend: &B,
) -> Complex32
where
    f32: BackendScalar<B>,
{
    let outputs = assignments(network)
        .iter()
        .map(|assignment| {
            let source = select_static_leaves(network, &prepared.source_cache, assignment);
            let mut refs = source.iter().collect::<Vec<_>>();
            refs.extend(prepared.factors.iter());
            prepared.einsum.execute::<Standard<f32>, f32, B>(&refs)
        })
        .collect::<Vec<_>>();
    backend.synchronize();

    let mut re = Kahan::default();
    let mut im = Kahan::default();
    for output in outputs {
        match prepared.output {
            RealifiedOutput::Real => {
                assert_eq!(output.numel(), 1, "benchmark requires scalar output");
                re.add(output.to_vec()[0] as f64);
            }
            RealifiedOutput::ReImAxis => {
                assert_eq!(output.shape(), &[2], "benchmark requires scalar output");
                let values = output.to_vec();
                re.add(values[0] as f64);
                im.add(values[1] as f64);
            }
        }
    }
    Complex32::new(re.sum as f32, im.sum as f32)
}

fn static_source_cache<B: Backend + Clone>(
    network: &BenchmarkNetwork,
    backend: B,
) -> Vec<Vec<Tensor<f32, B>>>
where
    f32: BackendScalar<B>,
{
    network
        .tensors
        .iter()
        .zip(&network.eincode.input_indices)
        .map(|(tensor, ix)| {
            local_assignments(network, ix)
                .iter()
                .map(|assignment| {
                    let (mut data, mut shape) =
                        slice_column_major(&tensor.data_re, &tensor.shape, ix, assignment);
                    if tensor.structurally_complex {
                        let (imaginary, imaginary_shape) =
                            slice_column_major(&tensor.data_im, &tensor.shape, ix, assignment);
                        debug_assert_eq!(shape, imaginary_shape);
                        data.extend(imaginary);
                        shape.push(2);
                    }
                    Tensor::from_data_with_backend(&data, &shape, backend.clone())
                })
                .collect()
        })
        .collect()
}

fn select_static_leaves<B: Backend + Clone>(
    network: &BenchmarkNetwork,
    cache: &[Vec<Tensor<f32, B>>],
    assignment: &HashMap<usize, usize>,
) -> Vec<Tensor<f32, B>> {
    cache
        .iter()
        .zip(&network.eincode.input_indices)
        .map(|(leaf, labels)| leaf[local_index(network, labels, assignment)].clone())
        .collect()
}

fn physical_size_dict(network: &BenchmarkNetwork) -> HashMap<usize, usize> {
    let mut sizes = network.size_dict.clone();
    for &cut in &network.cuts {
        sizes.insert(cut, 1);
    }
    sizes
}

struct DenseBuilt {
    tree: NestedEinsum<usize>,
    skeleton_ix: Vec<usize>,
    output_ix: Vec<usize>,
    green: Option<usize>,
}

struct DenseBuilder<'a, I>
where
    I: Iterator<Item = usize>,
{
    source_ixs: &'a [Vec<usize>],
    leaf_greens: &'a [Option<usize>],
    realified_ixs: &'a mut Vec<Vec<usize>>,
    factor_positions: &'a mut Vec<usize>,
    merge_greens: &'a mut I,
}

impl<I> DenseBuilder<'_, I>
where
    I: Iterator<Item = usize>,
{
    fn build(&mut self, node: &TreeNode) -> DenseBuilt {
        match node {
            TreeNode::Leaf { tensor_index } => DenseBuilt {
                tree: NestedEinsum::leaf(*tensor_index),
                skeleton_ix: self.source_ixs[*tensor_index].clone(),
                output_ix: self.realified_ixs[*tensor_index].clone(),
                green: self.leaf_greens[*tensor_index],
            },
            TreeNode::Node {
                args,
                input_indices,
                output_indices,
            } => {
                assert_eq!(args.len(), 2);
                assert_eq!(input_indices.len(), 2);
                let left = self.build(&args[0]);
                let right = self.build(&args[1]);
                assert_eq!(left.skeleton_ix, input_indices[0]);
                assert_eq!(right.skeleton_ix, input_indices[1]);
                match (left.green, right.green) {
                    (None, None) => dense_pass(left, right, output_indices),
                    (Some(green), None) | (None, Some(green)) => {
                        dense_ride(left, right, output_indices, green)
                    }
                    (Some(left_green), Some(right_green)) => {
                        self.merge(left, right, output_indices, left_green, right_green)
                    }
                }
            }
        }
    }

    fn merge(
        &mut self,
        left: DenseBuilt,
        right: DenseBuilt,
        iy: &[usize],
        left_green: usize,
        right_green: usize,
    ) -> DenseBuilt {
        let output_green = self
            .merge_greens
            .next()
            .expect("dense merge label preallocation underflow");
        let mut product_ix = iy.to_vec();
        product_ix.push(left_green);
        product_ix.push(right_green);
        let product = nested_binary(
            left.tree,
            left.output_ix,
            right.tree,
            right.output_ix,
            &product_ix,
        );

        let m_ix = vec![left_green, right_green, output_green];
        let m_position = self.realified_ixs.len();
        self.realified_ixs.push(m_ix.clone());
        self.factor_positions.push(m_position);
        let mut output_ix = iy.to_vec();
        output_ix.push(output_green);
        DenseBuilt {
            tree: nested_binary(
                product,
                product_ix,
                NestedEinsum::leaf(m_position),
                m_ix,
                &output_ix,
            ),
            skeleton_ix: iy.to_vec(),
            output_ix,
            green: Some(output_green),
        }
    }
}

fn dense_pass(left: DenseBuilt, right: DenseBuilt, iy: &[usize]) -> DenseBuilt {
    DenseBuilt {
        tree: nested_binary(left.tree, left.output_ix, right.tree, right.output_ix, iy),
        skeleton_ix: iy.to_vec(),
        output_ix: iy.to_vec(),
        green: None,
    }
}

fn dense_ride(left: DenseBuilt, right: DenseBuilt, iy: &[usize], green: usize) -> DenseBuilt {
    let mut output_ix = iy.to_vec();
    output_ix.push(green);
    DenseBuilt {
        tree: nested_binary(
            left.tree,
            left.output_ix,
            right.tree,
            right.output_ix,
            &output_ix,
        ),
        skeleton_ix: iy.to_vec(),
        output_ix,
        green: Some(green),
    }
}

fn nested_binary(
    left: NestedEinsum<usize>,
    left_ix: Vec<usize>,
    right: NestedEinsum<usize>,
    right_ix: Vec<usize>,
    iy: &[usize],
) -> NestedEinsum<usize> {
    NestedEinsum::node(
        vec![left, right],
        EinCode::new(vec![left_ix, right_ix], iy.to_vec()),
    )
}

fn dense_tree_code(
    network: &BenchmarkNetwork,
    is_complex: &[bool],
) -> (Einsum<usize>, Vec<usize>, RealifiedOutput) {
    let source_ixs = &network.eincode.input_indices;
    let source_iy = &network.eincode.output_indices;
    let source_tree = network.contraction_order.to_nested();
    if !is_complex.iter().any(|&complex| complex) {
        let mut einsum = Einsum::new(
            source_ixs.clone(),
            source_iy.clone(),
            physical_size_dict(network),
        );
        einsum.set_contraction_tree(source_tree);
        return (einsum, Vec::new(), RealifiedOutput::Real);
    }

    // Reuse the production allocator and leaf conversion layout. Only the dense
    // merge wiring differs from the legacy cascade it returns.
    let allocation = realify_code(
        source_ixs,
        source_iy,
        &physical_size_dict(network),
        is_complex,
    );
    let source_count = source_ixs.len();
    let mut realified_ixs = allocation.einsum.ixs[..source_count].to_vec();
    let leaf_greens = source_ixs
        .iter()
        .zip(&realified_ixs)
        .zip(is_complex)
        .map(|((source, realified), &complex)| complex.then(|| realified[source.len()]))
        .collect::<Vec<_>>();
    let mut merge_greens = allocation
        .mul_vertex_positions
        .iter()
        .map(|&position| allocation.einsum.ixs[position][2]);
    let mut factor_positions = Vec::new();
    let built = DenseBuilder {
        source_ixs,
        leaf_greens: &leaf_greens,
        realified_ixs: &mut realified_ixs,
        factor_positions: &mut factor_positions,
        merge_greens: &mut merge_greens,
    }
    .build(&network.contraction_order);
    assert!(merge_greens.next().is_none());

    let mut iy = source_iy.clone();
    iy.push(built.green.expect("complex tree must have green output"));
    let mut einsum = Einsum::new(realified_ixs, iy, allocation.einsum.size_dict);
    einsum.set_contraction_tree(built.tree);
    (einsum, factor_positions, RealifiedOutput::ReImAxis)
}

fn real_walk<B: Backend + Clone>(node: &TreeNode, leaves: &[RealValue<B>]) -> RealValue<B>
where
    f32: BackendScalar<B>,
{
    match node {
        TreeNode::Leaf { tensor_index } => leaves[*tensor_index].clone(),
        TreeNode::Node {
            args,
            input_indices,
            output_indices,
        } => {
            let (ar, ai) = real_walk(&args[0], leaves);
            let (br, bi) = real_walk(&args[1], leaves);
            let c = |x: &Tensor<f32, B>, y: &Tensor<f32, B>| {
                x.contract_binary::<Standard<f32>>(
                    y,
                    &input_indices[0],
                    &input_indices[1],
                    output_indices,
                )
            };
            match (ai, bi) {
                (None, None) => (c(&ar, &br), None),
                (Some(ai), None) => {
                    let re = c(&ar, &br);
                    let im = c(&ai, &br);
                    (re, Some(im))
                }
                (None, Some(bi)) => {
                    let re = c(&ar, &br);
                    let im = c(&ar, &bi);
                    (re, Some(im))
                }
                (Some(ai), Some(bi)) => {
                    let asum = ar.linear_combination(&ai, 1.0);
                    let bsum = br.linear_combination(&bi, 1.0);
                    let p1 = c(&asum, &bsum);
                    let p2 = c(&ar, &br);
                    let p3 = c(&ai, &bi);
                    let re = p2.linear_combination(&p3, -1.0);
                    let im = p1
                        .linear_combination(&p2, -1.0)
                        .linear_combination(&p3, -1.0);
                    (re, Some(im))
                }
            }
        }
    }
}

pub fn solve_real<B: Backend + Clone>(
    network: &BenchmarkNetwork,
    cache: &[Vec<RealValue<B>>],
    backend: &B,
) -> Complex32
where
    f32: BackendScalar<B>,
{
    let outputs = assignments(network)
        .iter()
        .map(|assignment| {
            let leaves = select_real_leaves(network, cache, assignment);
            real_walk(&network.contraction_order, &leaves)
        })
        .collect::<Vec<_>>();
    backend.synchronize();
    let mut re = Kahan::default();
    let mut im = Kahan::default();
    for (r, i) in outputs {
        assert_eq!(
            r.numel(),
            1,
            "benchmark reduction currently requires scalar output"
        );
        re.add(r.to_vec()[0] as f64);
        if let Some(i) = i {
            im.add(i.to_vec()[0] as f64)
        }
    }
    Complex32::new(re.sum as f32, im.sum as f32)
}

pub fn native_cache<T, B: Backend + Clone, F: Fn(f32, f32) -> T>(
    network: &BenchmarkNetwork,
    backend: B,
    make: F,
) -> Vec<Vec<Tensor<T, B>>>
where
    T: Scalar + BackendScalar<B>,
{
    network
        .tensors
        .iter()
        .zip(&network.eincode.input_indices)
        .map(|(t, ix)| {
            let data = t
                .data_re
                .iter()
                .zip(&t.data_im)
                .map(|(&r, &i)| make(r, i))
                .collect::<Vec<_>>();
            local_assignments(network, ix)
                .iter()
                .map(|a| {
                    let (v, shape) = slice_column_major(&data, &t.shape, ix, a);
                    Tensor::from_data_with_backend(&v, &shape, backend.clone())
                })
                .collect()
        })
        .collect()
}
fn native_walk<T, B: Backend + Clone>(node: &TreeNode, leaves: &[Tensor<T, B>]) -> Tensor<T, B>
where
    T: Scalar + Zero + One + PartialEq + BackendScalar<B>,
{
    match node {
        TreeNode::Leaf { tensor_index } => leaves[*tensor_index].clone(),
        TreeNode::Node {
            args,
            input_indices,
            output_indices,
        } => native_walk(&args[0], leaves).contract_binary::<Standard<T>>(
            &native_walk(&args[1], leaves),
            &input_indices[0],
            &input_indices[1],
            output_indices,
        ),
    }
}
pub fn solve_native<T, B: Backend + Clone>(
    network: &BenchmarkNetwork,
    cache: &[Vec<Tensor<T, B>>],
    backend: &B,
) -> Complex32
where
    T: Scalar + Zero + One + PartialEq + Into<Complex32> + BackendScalar<B>,
{
    let out = assignments(network)
        .iter()
        .map(|assignment| {
            let leaves = select_native_leaves(network, cache, assignment);
            native_walk(&network.contraction_order, &leaves)
        })
        .collect::<Vec<_>>();
    backend.synchronize();
    let (mut re, mut im) = (Kahan::default(), Kahan::default());
    for x in out {
        assert_eq!(x.numel(), 1);
        let c: Complex32 = x.to_vec()[0].into();
        re.add(c.re as f64);
        im.add(c.im as f64)
    }
    Complex32::new(re.sum as f32, im.sum as f32)
}

// W4: c64 mirrors of native_cache/solve_native for the f64 reference runs.
// They read the artifact's f64 payloads and never downcast; the f32 path
// above is untouched.
pub fn native_cache_c64<T, B: Backend + Clone, F: Fn(f64, f64) -> T>(
    network: &BenchmarkNetwork,
    backend: B,
    make: F,
) -> Vec<Vec<Tensor<T, B>>>
where
    T: Scalar + BackendScalar<B>,
{
    network
        .tensors
        .iter()
        .zip(&network.eincode.input_indices)
        .map(|(t, ix)| {
            let re = t.data_re_f64.as_ref().expect("missing c64 payload");
            let im = t.data_im_f64.as_ref().expect("missing c64 payload");
            let data = re
                .iter()
                .zip(im)
                .map(|(&r, &i)| make(r, i))
                .collect::<Vec<_>>();
            local_assignments(network, ix)
                .iter()
                .map(|a| {
                    let (v, shape) = slice_column_major(&data, &t.shape, ix, a);
                    Tensor::from_data_with_backend(&v, &shape, backend.clone())
                })
                .collect()
        })
        .collect()
}

pub fn solve_native_c64<T, B: Backend + Clone>(
    network: &BenchmarkNetwork,
    cache: &[Vec<Tensor<T, B>>],
    backend: &B,
) -> Complex64
where
    T: Scalar + Zero + One + PartialEq + Into<Complex64> + BackendScalar<B>,
{
    let out = assignments(network)
        .iter()
        .map(|assignment| {
            let leaves = select_native_leaves(network, cache, assignment);
            native_walk(&network.contraction_order, &leaves)
        })
        .collect::<Vec<_>>();
    backend.synchronize();
    let (mut re, mut im) = (Kahan::default(), Kahan::default());
    for x in out {
        assert_eq!(x.numel(), 1);
        let c: Complex64 = x.to_vec()[0].into();
        re.add(c.re);
        im.add(c.im)
    }
    Complex64::new(re.sum, im.sum)
}

fn local_assignments(network: &BenchmarkNetwork, labels: &[usize]) -> Vec<HashMap<usize, usize>> {
    let cuts = network
        .cuts
        .iter()
        .copied()
        .filter(|cut| labels.contains(cut))
        .collect::<Vec<_>>();
    let count = cuts.iter().map(|cut| network.size_dict[cut]).product();
    (0..count)
        .map(|mut n| {
            let mut assignment = HashMap::new();
            for cut in cuts.iter().rev() {
                assignment.insert(*cut, n % network.size_dict[cut]);
                n /= network.size_dict[cut];
            }
            assignment
        })
        .collect()
}

fn local_index(
    network: &BenchmarkNetwork,
    labels: &[usize],
    assignment: &HashMap<usize, usize>,
) -> usize {
    network
        .cuts
        .iter()
        .filter(|cut| labels.contains(cut))
        .fold(0, |index, cut| {
            index * network.size_dict[cut] + assignment[cut]
        })
}

fn select_real_leaves<B: Backend + Clone>(
    network: &BenchmarkNetwork,
    cache: &[Vec<RealValue<B>>],
    assignment: &HashMap<usize, usize>,
) -> Vec<RealValue<B>> {
    cache
        .iter()
        .zip(&network.eincode.input_indices)
        .map(|(leaf, labels)| leaf[local_index(network, labels, assignment)].clone())
        .collect()
}

fn select_native_leaves<T: Scalar, B: Backend + Clone>(
    network: &BenchmarkNetwork,
    cache: &[Vec<Tensor<T, B>>],
    assignment: &HashMap<usize, usize>,
) -> Vec<Tensor<T, B>> {
    cache
        .iter()
        .zip(&network.eincode.input_indices)
        .map(|(leaf, labels)| leaf[local_index(network, labels, assignment)].clone())
        .collect()
}
#[derive(Default)]
struct Kahan {
    sum: f64,
    c: f64,
}
impl Kahan {
    fn add(&mut self, x: f64) {
        let y = x - self.c;
        let t = self.sum + y;
        self.c = (t - self.sum) - y;
        self.sum = t
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use omeinsum::Cpu;
    #[test]
    fn sliced_native_equals_tree_real() {
        let n = BenchmarkNetwork {
            format: "omeinsum-yao-benchmark-v2".into(),
            source_format: "x".into(),
            source_mode: "x".into(),
            optimizer: Default::default(),
            slicer: Default::default(),
            eincode: super::super::format::BenchmarkEinCode {
                input_indices: vec![vec![0], vec![0]],
                output_indices: vec![],
            },
            tensors: vec![
                super::super::format::BenchmarkTensor {
                    shape: vec![2],
                    data_re: vec![1., 2.],
                    data_im: vec![0., 0.],
                    structurally_complex: false,
                },
                super::super::format::BenchmarkTensor {
                    shape: vec![2],
                    data_re: vec![3., 4.],
                    data_im: vec![1., -1.],
                    structurally_complex: true,
                },
            ],
            size_dict: HashMap::from([(0, 2)]),
            contraction_order: TreeNode::Node {
                args: vec![
                    TreeNode::Leaf { tensor_index: 0 },
                    TreeNode::Leaf { tensor_index: 1 },
                ],
                input_indices: vec![vec![0], vec![0]],
                output_indices: vec![],
            },
            cuts: vec![0],
            assignment_count: 2,
            complexity: Default::default(),
            tree_real_audit: Default::default(),
        };
        let rc = real_cache(&n, Cpu);
        let nc = native_cache(&n, Cpu, Complex32::new);
        let factorized = static_factorized(&n, Cpu);
        let dense = static_dense(&n, Cpu);
        assert_eq!(solve_real(&n, &rc, &Cpu), Complex32::new(11., -1.));
        assert_eq!(solve_native(&n, &nc, &Cpu), solve_real(&n, &rc, &Cpu));
        assert_eq!(
            solve_static(&n, &factorized, &Cpu),
            solve_real(&n, &rc, &Cpu)
        );
        assert_eq!(solve_static(&n, &dense, &Cpu), solve_real(&n, &rc, &Cpu));
    }

    #[test]
    fn cache_counts_are_products_of_only_each_leafs_cuts() {
        let mut n = test_network(
            vec![vec![0], vec![1], vec![2]],
            vec![vec![2], vec![3], vec![5]],
            vec![0, 1],
            HashMap::from([(0, 2), (1, 3), (2, 5)]),
        );
        n.assignment_count = 6;
        let native = native_cache(&n, Cpu, Complex32::new);
        let real = real_cache(&n, Cpu);
        assert_eq!(
            native.iter().map(Vec::len).collect::<Vec<_>>(),
            vec![2, 3, 1]
        );
        assert_eq!(real.iter().map(Vec::len).collect::<Vec<_>>(), vec![2, 3, 1]);
    }

    #[test]
    fn sliced_multidimensional_complex_gauss_matches_native_values() {
        let shape = vec![2, 3, 4];
        let re_a = (0..24).map(|i| i as f32 * 0.25 + 1.0).collect::<Vec<_>>();
        let im_a = (0..24).map(|i| i as f32 * -0.1 + 0.5).collect::<Vec<_>>();
        let re_b = (0..24).map(|i| i as f32 * 0.15 - 0.75).collect::<Vec<_>>();
        let im_b = (0..24).map(|i| i as f32 * 0.2 + 0.25).collect::<Vec<_>>();
        let mut n = test_network(
            vec![vec![0, 1, 2], vec![0, 1, 2]],
            vec![shape.clone(), shape],
            vec![1],
            HashMap::from([(0, 2), (1, 3), (2, 4)]),
        );
        n.assignment_count = 3;
        n.tensors[0].data_re = re_a;
        n.tensors[0].data_im = im_a;
        n.tensors[1].data_re = re_b;
        n.tensors[1].data_im = im_b;
        n.tensors
            .iter_mut()
            .for_each(|t| t.structurally_complex = true);
        let real = solve_real(&n, &real_cache(&n, Cpu), &Cpu);
        let native = solve_native(&n, &native_cache(&n, Cpu, Complex32::new), &Cpu);
        let factorized = solve_static(&n, &static_factorized(&n, Cpu), &Cpu);
        let dense = solve_static(&n, &static_dense(&n, Cpu), &Cpu);
        assert!(real.re != 0.0 && real.im != 0.0);
        for (name, actual) in [
            ("dispatch", real),
            ("static-factorized", factorized),
            ("static-dense", dense),
        ] {
            assert!(
                (actual - native).norm() <= 2e-4 * native.norm().max(1.0),
                "{name}={actual:?}, native={native:?}"
            );
        }
    }

    fn test_network(
        labels: Vec<Vec<usize>>,
        shapes: Vec<Vec<usize>>,
        cuts: Vec<usize>,
        size_dict: HashMap<usize, usize>,
    ) -> BenchmarkNetwork {
        let tensors = shapes
            .iter()
            .map(|shape| {
                let len = shape.iter().product();
                super::super::format::BenchmarkTensor {
                    shape: shape.clone(),
                    data_re: vec![1.0; len],
                    data_im: vec![0.0; len],
                    structurally_complex: false,
                }
            })
            .collect::<Vec<_>>();
        let contraction_order = if tensors.len() == 2 {
            TreeNode::Node {
                args: vec![
                    TreeNode::Leaf { tensor_index: 0 },
                    TreeNode::Leaf { tensor_index: 1 },
                ],
                input_indices: labels.clone(),
                output_indices: vec![],
            }
        } else {
            TreeNode::Leaf { tensor_index: 0 }
        };
        BenchmarkNetwork {
            format: "omeinsum-yao-benchmark-v2".into(),
            source_format: "x".into(),
            source_mode: "x".into(),
            optimizer: Default::default(),
            slicer: Default::default(),
            eincode: super::super::format::BenchmarkEinCode {
                input_indices: labels,
                output_indices: vec![],
            },
            tensors,
            size_dict,
            contraction_order,
            cuts,
            assignment_count: 1,
            complexity: Default::default(),
            tree_real_audit: Default::default(),
        }
    }
}

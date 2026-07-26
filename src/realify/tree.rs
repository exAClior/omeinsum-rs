//! Tree-following complex-to-real topology conversion.
//!
//! Unlike the dense cascade built by [`super::realify_code`], this transform starts
//! from an already chosen complex contraction tree and preserves that order. A
//! complex/complex merge is expanded into the rank-3 factorization of complex
//! multiplication, so the ordinary binary executor performs Gauss's three-product
//! schedule without a custom kernel.

use std::collections::HashMap;

use omeco::{EinCode, NestedEinsum};

use super::{allocate_fresh_labels, RealifiedOutput};
use crate::einsum::Einsum;

/// Result of tree-following realification.
///
/// Original inputs keep their positions. Each merge appends three factor leaves in
/// `U, V, W` order; [`factor_vertex_positions`](Self::factor_vertex_positions)
/// records those positions. The generated contraction tree is already installed on
/// [`einsum`](Self::einsum) and must not be re-optimized.
pub struct RealifyTreePlan {
    /// Flat real einsum specification with its generated contraction tree installed.
    pub einsum: Einsum<usize>,
    /// Positions of the `U`, `V`, and `W` leaves for every complex/complex merge.
    pub factor_vertex_positions: Vec<(usize, usize, usize)>,
    /// Whether the output has a trailing Re/Im axis.
    pub output: RealifiedOutput,
}

struct BuiltNode {
    tree: NestedEinsum<usize>,
    skeleton_ix: Vec<usize>,
    output_ix: Vec<usize>,
    green: Option<usize>,
}

struct Builder<'a, I>
where
    I: Iterator<Item = usize>,
{
    source_ixs: &'a [Vec<usize>],
    leaf_greens: &'a [Option<usize>],
    realified_ixs: &'a mut Vec<Vec<usize>>,
    realified_sizes: &'a mut HashMap<usize, usize>,
    factor_vertex_positions: &'a mut Vec<(usize, usize, usize)>,
    fresh_labels: &'a mut I,
    leaf_counts: &'a mut [usize],
}

impl<I> Builder<'_, I>
where
    I: Iterator<Item = usize>,
{
    fn build(&mut self, source: &NestedEinsum<usize>) -> BuiltNode {
        match source {
            NestedEinsum::Leaf { tensor_index } => {
                assert!(
                    *tensor_index < self.source_ixs.len(),
                    "contraction tree leaf {tensor_index} is out of range for {} inputs",
                    self.source_ixs.len()
                );
                self.leaf_counts[*tensor_index] += 1;
                BuiltNode {
                    tree: NestedEinsum::leaf(*tensor_index),
                    skeleton_ix: self.source_ixs[*tensor_index].clone(),
                    output_ix: self.realified_ixs[*tensor_index].clone(),
                    green: self.leaf_greens[*tensor_index],
                }
            }
            NestedEinsum::Node { args, eins } => {
                assert_eq!(args.len(), 2, "tree realification requires a binary tree");
                assert_eq!(
                    eins.ixs.len(),
                    2,
                    "tree realification requires two input index lists per node"
                );

                let left = self.build(&args[0]);
                let right = self.build(&args[1]);
                assert_eq!(
                    eins.ixs[0], left.skeleton_ix,
                    "left child output does not match archived node input"
                );
                assert_eq!(
                    eins.ixs[1], right.skeleton_ix,
                    "right child output does not match archived node input"
                );

                match (left.green, right.green) {
                    (None, None) => self.pass(left, right, &eins.iy),
                    (Some(green), None) => self.ride(left, right, &eins.iy, green),
                    (None, Some(green)) => self.ride(left, right, &eins.iy, green),
                    (Some(left_green), Some(right_green)) => {
                        self.merge(left, right, &eins.iy, left_green, right_green)
                    }
                }
            }
        }
    }

    fn pass(&self, left: BuiltNode, right: BuiltNode, iy: &[usize]) -> BuiltNode {
        let output_ix = iy.to_vec();
        BuiltNode {
            tree: binary_node(
                left.tree,
                left.output_ix,
                right.tree,
                right.output_ix,
                &output_ix,
            ),
            skeleton_ix: iy.to_vec(),
            output_ix,
            green: None,
        }
    }

    fn ride(&self, left: BuiltNode, right: BuiltNode, iy: &[usize], green: usize) -> BuiltNode {
        let mut output_ix = iy.to_vec();
        output_ix.push(green);
        BuiltNode {
            tree: binary_node(
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

    fn merge(
        &mut self,
        left: BuiltNode,
        right: BuiltNode,
        iy: &[usize],
        left_green: usize,
        right_green: usize,
    ) -> BuiltNode {
        let rank = self
            .fresh_labels
            .next()
            .expect("fresh label preallocation underflow at merge rank");
        let output_green = self
            .fresh_labels
            .next()
            .expect("fresh label preallocation underflow at merge output");
        self.realified_sizes.insert(rank, 3);
        self.realified_sizes.insert(output_green, 2);

        let u_ix = vec![left_green, rank];
        let u_position = self.realified_ixs.len();
        self.realified_ixs.push(u_ix.clone());
        let v_ix = vec![right_green, rank];
        let v_position = self.realified_ixs.len();
        self.realified_ixs.push(v_ix.clone());
        let w_ix = vec![output_green, rank];
        let w_position = self.realified_ixs.len();
        self.realified_ixs.push(w_ix.clone());
        self.factor_vertex_positions
            .push((u_position, v_position, w_position));

        // The source binary step would normalize each operand before contracting:
        // repeated labels become a diagonal and labels private to one operand are
        // reduced. The inserted U/V nodes must do that normalization now, otherwise
        // a repeated label would illegally appear twice in their output shape.
        let mut left_rank_ix =
            normalized_operand_skeleton(&left.skeleton_ix, &right.skeleton_ix, iy);
        left_rank_ix.push(rank);
        let left_rank_tree = binary_node(
            left.tree,
            left.output_ix,
            NestedEinsum::leaf(u_position),
            u_ix,
            &left_rank_ix,
        );

        let mut right_rank_ix =
            normalized_operand_skeleton(&right.skeleton_ix, &left.skeleton_ix, iy);
        right_rank_ix.push(rank);
        let right_rank_tree = binary_node(
            right.tree,
            right.output_ix,
            NestedEinsum::leaf(v_position),
            v_ix,
            &right_rank_ix,
        );

        let mut product_ix = iy.to_vec();
        product_ix.push(rank);
        let product_tree = binary_node(
            left_rank_tree,
            left_rank_ix,
            right_rank_tree,
            right_rank_ix,
            &product_ix,
        );

        let mut output_ix = iy.to_vec();
        output_ix.push(output_green);
        let tree = binary_node(
            product_tree,
            product_ix,
            NestedEinsum::leaf(w_position),
            w_ix,
            &output_ix,
        );

        BuiltNode {
            tree,
            skeleton_ix: iy.to_vec(),
            output_ix,
            green: Some(output_green),
        }
    }
}

fn normalized_operand_skeleton(ix: &[usize], other: &[usize], output: &[usize]) -> Vec<usize> {
    let other = other
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>();
    let output = output
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>();
    let mut seen = std::collections::HashSet::new();
    ix.iter()
        .copied()
        .filter(|label| (other.contains(label) || output.contains(label)) && seen.insert(*label))
        .collect()
}

fn binary_node(
    left: NestedEinsum<usize>,
    left_ix: Vec<usize>,
    right: NestedEinsum<usize>,
    right_ix: Vec<usize>,
    output_ix: &[usize],
) -> NestedEinsum<usize> {
    NestedEinsum::node(
        vec![left, right],
        EinCode::new(vec![left_ix, right_ix], output_ix.to_vec()),
    )
}

/// Realify an einsum while following an archived complex contraction tree.
///
/// A node with no complex child is copied (`pass`); one complex child carries its
/// dim-2 leg through (`ride`); two complex children are combined by a four-node
/// `U/V/W` factorization (`merge`). Every generated merge has its own size-3 rank
/// label. The returned [`Einsum`] already contains the generated tree.
///
/// This is Gauss's three-product algorithm. For floating-point inputs its linear
/// forms can overflow or lose cancellation in cases where the dense four-product
/// multiplication remains finite. Use [`super::realify_code`] when that wider
/// numerical range matters more than the factorized schedule.
///
/// # Panics
///
/// Panics when `is_complex` does not match `ixs`, when the archived tree is not a
/// valid binary tree over every source input exactly once, or when archived node
/// input labels do not match their child outputs.
pub fn realify_tree_code(
    tree: &NestedEinsum<usize>,
    ixs: &[Vec<usize>],
    iy: &[usize],
    size_dict: &HashMap<usize, usize>,
    is_complex: &[bool],
) -> RealifyTreePlan {
    assert_eq!(
        ixs.len(),
        is_complex.len(),
        "is_complex length {} must match number of inputs {}",
        is_complex.len(),
        ixs.len()
    );

    let complex_count = is_complex.iter().filter(|&&complex| complex).count();
    if complex_count == 0 {
        let mut einsum = Einsum::new(ixs.to_vec(), iy.to_vec(), size_dict.clone());
        einsum.set_contraction_tree(tree.clone());
        validate_tree_metadata(tree, ixs);
        return RealifyTreePlan {
            einsum,
            factor_vertex_positions: Vec::new(),
            output: RealifiedOutput::Real,
        };
    }

    let merge_count = complex_count - 1;
    let fresh_count = complex_count
        .checked_add(
            merge_count
                .checked_mul(2)
                .expect("too many merges to allocate fresh labels"),
        )
        .expect("too many complex inputs to allocate fresh labels");
    let fresh_labels = allocate_fresh_labels(ixs, iy, size_dict, fresh_count);
    let mut fresh_labels = fresh_labels.into_iter();
    let mut realified_ixs = ixs.to_vec();
    let mut realified_sizes = size_dict.clone();
    let mut leaf_greens = vec![None; ixs.len()];

    for (tensor_index, &complex) in is_complex.iter().enumerate() {
        if complex {
            let green = fresh_labels
                .next()
                .expect("fresh label preallocation underflow at complex leaf");
            realified_ixs[tensor_index].push(green);
            realified_sizes.insert(green, 2);
            leaf_greens[tensor_index] = Some(green);
        }
    }

    let mut factor_vertex_positions = Vec::with_capacity(merge_count);
    let mut leaf_counts = vec![0usize; ixs.len()];
    let built = Builder {
        source_ixs: ixs,
        leaf_greens: &leaf_greens,
        realified_ixs: &mut realified_ixs,
        realified_sizes: &mut realified_sizes,
        factor_vertex_positions: &mut factor_vertex_positions,
        fresh_labels: &mut fresh_labels,
        leaf_counts: &mut leaf_counts,
    }
    .build(tree);

    for (tensor_index, count) in leaf_counts.into_iter().enumerate() {
        assert_eq!(
            count, 1,
            "contraction tree must reference tensor {tensor_index} exactly once, found {count} leaves"
        );
    }
    assert_eq!(
        factor_vertex_positions.len(),
        merge_count,
        "a binary tree with {complex_count} complex leaves must contain {merge_count} complex merges"
    );
    assert!(
        fresh_labels.next().is_none(),
        "fresh label preallocation overflow"
    );

    let output_green = built
        .green
        .expect("a tree containing complex leaves must have a complex root");
    let mut realified_iy = iy.to_vec();
    realified_iy.push(output_green);
    // The archived root may emit the same labels in a different order from the
    // enclosing Einsum output. Preserve that distinction: `execute` performs the
    // final permutation/reduction after walking the installed tree.
    let mut einsum = Einsum::new(realified_ixs, realified_iy, realified_sizes);
    einsum.set_contraction_tree(built.tree);
    RealifyTreePlan {
        einsum,
        factor_vertex_positions,
        output: RealifiedOutput::ReImAxis,
    }
}

fn validate_tree_metadata(tree: &NestedEinsum<usize>, source_ixs: &[Vec<usize>]) {
    fn walk(
        tree: &NestedEinsum<usize>,
        source_ixs: &[Vec<usize>],
        counts: &mut [usize],
    ) -> Vec<usize> {
        match tree {
            NestedEinsum::Leaf { tensor_index } => {
                assert!(
                    *tensor_index < counts.len(),
                    "contraction tree leaf {tensor_index} is out of range for {} inputs",
                    counts.len()
                );
                counts[*tensor_index] += 1;
                source_ixs[*tensor_index].clone()
            }
            NestedEinsum::Node { args, eins } => {
                assert_eq!(args.len(), 2, "tree realification requires a binary tree");
                assert_eq!(
                    eins.ixs.len(),
                    2,
                    "tree realification requires two input index lists per node"
                );
                let child_outputs = args
                    .iter()
                    .map(|arg| walk(arg, source_ixs, counts))
                    .collect::<Vec<_>>();
                assert_eq!(
                    eins.ixs, child_outputs,
                    "archived node inputs do not match child outputs"
                );
                eins.iy.clone()
            }
        }
    }

    let mut counts = vec![0usize; source_ixs.len()];
    walk(tree, source_ixs, &mut counts);
    for (tensor_index, count) in counts.into_iter().enumerate() {
        assert_eq!(
            count, 1,
            "contraction tree must reference tensor {tensor_index} exactly once, found {count} leaves"
        );
    }
}

#[cfg(test)]
mod tests;

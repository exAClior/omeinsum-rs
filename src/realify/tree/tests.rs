use std::collections::{HashMap, HashSet};

use omeco::{EinCode, NestedEinsum};

use super::*;
use crate::backend::Cpu;
use crate::realify::{constants, merge_factor_tensors};

fn node(
    left: NestedEinsum<usize>,
    left_ix: &[usize],
    right: NestedEinsum<usize>,
    right_ix: &[usize],
    iy: &[usize],
) -> NestedEinsum<usize> {
    NestedEinsum::node(
        vec![left, right],
        EinCode::new(vec![left_ix.to_vec(), right_ix.to_vec()], iy.to_vec()),
    )
}

fn pass_ride_merge_tree() -> NestedEinsum<usize> {
    let pass = node(
        NestedEinsum::leaf(1),
        &[1, 2],
        NestedEinsum::leaf(2),
        &[2, 3],
        &[1, 3],
    );
    let ride = node(NestedEinsum::leaf(0), &[0, 1], pass, &[1, 3], &[0, 3]);
    node(ride, &[0, 3], NestedEinsum::leaf(3), &[3, 4], &[0, 4])
}

#[test]
fn factorized_constants_reconstruct_dense_m_exactly() {
    for a in 0..2 {
        for b in 0..2 {
            for c in 0..2 {
                let actual = (0..3)
                    .map(|k| {
                        constants::U_DATA[a + 2 * k]
                            * constants::V_DATA[b + 2 * k]
                            * constants::W_DATA[c + 2 * k]
                    })
                    .sum::<f64>();
                assert_eq!(actual, constants::M_DATA[a + 2 * b + 4 * c]);
            }
        }
    }
}

#[test]
fn merge_factor_tensors_have_expected_shapes_and_values() {
    let (u, w) = merge_factor_tensors::<f32, Cpu>(Cpu);
    assert_eq!(u.shape(), &[2, 3]);
    assert_eq!(w.shape(), &[2, 3]);
    assert_eq!(u.to_vec(), constants::U_DATA.map(|x| x as f32));
    assert_eq!(w.to_vec(), constants::W_DATA.map(|x| x as f32));
}

#[test]
fn tree_transform_maps_pass_ride_and_merge_nodes() {
    let ixs = vec![vec![0, 1], vec![1, 2], vec![2, 3], vec![3, 4]];
    let sizes = (0..=4).map(|label| (label, 2)).collect();
    let plan = realify_tree_code(
        &pass_ride_merge_tree(),
        &ixs,
        &[0, 4],
        &sizes,
        &[true, false, false, true],
    );

    assert_eq!(
        plan.einsum.ixs,
        vec![
            vec![0, 1, 5],
            vec![1, 2],
            vec![2, 3],
            vec![3, 4, 6],
            vec![5, 7],
            vec![6, 7],
            vec![8, 7],
        ]
    );
    assert_eq!(plan.einsum.iy, vec![0, 4, 8]);
    assert_eq!(plan.einsum.size_dict[&5], 2);
    assert_eq!(plan.einsum.size_dict[&6], 2);
    assert_eq!(plan.einsum.size_dict[&7], 3);
    assert_eq!(plan.einsum.size_dict[&8], 2);
    assert_eq!(plan.factor_vertex_positions, vec![(4, 5, 6)]);
    assert_eq!(plan.output, RealifiedOutput::ReImAxis);

    let generated_greens = HashSet::from([5, 6, 8]);
    fn assert_one_green(tree: &NestedEinsum<usize>, generated_greens: &HashSet<usize>) {
        if let NestedEinsum::Node { args, eins } = tree {
            assert!(
                eins.iy
                    .iter()
                    .filter(|label| generated_greens.contains(label))
                    .count()
                    <= 1,
                "node output {:?} carries multiple green labels",
                eins.iy
            );
            for arg in args {
                assert_one_green(arg, generated_greens);
            }
        }
    }
    assert_one_green(
        plan.einsum.contraction_tree().expect("installed tree"),
        &generated_greens,
    );
}

#[test]
fn all_real_tree_transform_is_an_identity_with_installed_tree() {
    let ixs = vec![vec![0, 1], vec![1, 2]];
    let tree = node(
        NestedEinsum::leaf(0),
        &ixs[0],
        NestedEinsum::leaf(1),
        &ixs[1],
        &[0, 2],
    );
    let sizes = HashMap::from([(0, 2), (1, 3), (2, 4)]);
    let plan = realify_tree_code(&tree, &ixs, &[0, 2], &sizes, &[false, false]);

    assert_eq!(plan.einsum.ixs, ixs);
    assert_eq!(plan.einsum.iy, vec![0, 2]);
    assert_eq!(plan.einsum.size_dict, sizes);
    assert!(plan.factor_vertex_positions.is_empty());
    assert_eq!(plan.output, RealifiedOutput::Real);
    assert!(plan.einsum.contraction_tree().is_some());
}

#[test]
#[should_panic(expected = "archived node inputs do not match child outputs")]
fn all_real_transform_rejects_inconsistent_archived_metadata() {
    let ixs = vec![vec![0], vec![1]];
    let malformed = node(
        NestedEinsum::leaf(0),
        &[1],
        NestedEinsum::leaf(1),
        &[0],
        &[0, 1],
    );
    let _ = realify_tree_code(
        &malformed,
        &ixs,
        &[0, 1],
        &HashMap::from([(0, 2), (1, 2)]),
        &[false, false],
    );
}

#[test]
fn one_complex_leaf_rides_without_factor_vertices() {
    let ixs = vec![vec![0, 1], vec![1, 2]];
    let tree = node(
        NestedEinsum::leaf(0),
        &ixs[0],
        NestedEinsum::leaf(1),
        &ixs[1],
        &[0, 2],
    );
    let plan = realify_tree_code(
        &tree,
        &ixs,
        &[2, 0],
        &HashMap::from([(0, 2), (1, 3), (2, 4)]),
        &[false, true],
    );

    assert_eq!(plan.einsum.ixs, vec![vec![0, 1], vec![1, 2, 3]]);
    assert_eq!(plan.einsum.iy, vec![2, 0, 3]);
    assert!(plan.factor_vertex_positions.is_empty());
}

#[test]
fn generated_labels_avoid_sparse_and_archived_cut_labels() {
    let ixs = vec![vec![0], vec![0]];
    let tree = node(
        NestedEinsum::leaf(0),
        &ixs[0],
        NestedEinsum::leaf(1),
        &ixs[1],
        &[],
    );
    let sizes = HashMap::from([(0, 2), (100, 2), (10_000, 7)]);
    let plan = realify_tree_code(&tree, &ixs, &[], &sizes, &[true, true]);

    let generated = plan
        .einsum
        .size_dict
        .keys()
        .copied()
        .filter(|label| !sizes.contains_key(label))
        .collect::<HashSet<_>>();
    assert_eq!(generated, HashSet::from([10_001, 10_002, 10_003, 10_004]));
    assert!(!generated.contains(&100));
    assert_eq!(plan.einsum.size_dict[&10_003], 3);
}

#[test]
fn complex_unary_tree_keeps_archived_leaf_and_appends_output_axis() {
    let ixs = vec![vec![0, 0]];
    let plan = realify_tree_code(
        &NestedEinsum::leaf(0),
        &ixs,
        &[],
        &HashMap::from([(0, 3)]),
        &[true],
    );

    assert_eq!(plan.einsum.ixs, vec![vec![0, 0, 1]]);
    assert_eq!(plan.einsum.iy, vec![1]);
    assert!(matches!(
        plan.einsum.contraction_tree(),
        Some(NestedEinsum::Leaf { tensor_index: 0 })
    ));
}

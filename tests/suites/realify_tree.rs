use approx::assert_relative_eq;
use num_complex::Complex64;
use omeco::{EinCode, NestedEinsum};
use omeinsum::{cost_and_gradient, recover_complex, Cpu, Einsum, Standard, Tensor};

use super::realify_support::{self, *};

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

fn mixed_pass_ride_merge_tree() -> NestedEinsum<usize> {
    let pass = node(
        NestedEinsum::leaf(1),
        &[1, 2],
        NestedEinsum::leaf(2),
        &[2, 3],
        &[1, 3],
    );
    let left_ride = node(NestedEinsum::leaf(0), &[0, 1], pass, &[1, 3], &[0, 3]);
    let right_ride = node(
        NestedEinsum::leaf(3),
        &[3, 4],
        NestedEinsum::leaf(4),
        &[4, 5],
        &[3, 5],
    );
    // Deliberately emit root labels in a different order from the enclosing
    // Einsum output; the engine must apply the final permutation after the tree.
    node(left_ride, &[0, 3], right_ride, &[3, 5], &[0, 5])
}

#[test]
fn tree_realification_matches_native_with_pass_ride_merge_and_permuted_output() {
    let ixs = vec![vec![0, 1], vec![1, 2], vec![2, 3], vec![3, 4], vec![4, 5]];
    let iy = vec![5, 0];
    let inputs = vec![
        OwnedInput::Complex {
            data: patterned_complex(2 * 3, 1),
            shape: vec![2, 3],
            conjugate: false,
        },
        OwnedInput::Real {
            data: patterned_real(3 * 2, 2),
            shape: vec![3, 2],
        },
        OwnedInput::Real {
            data: patterned_real(2 * 3, 3),
            shape: vec![2, 3],
        },
        OwnedInput::Complex {
            data: patterned_complex(3 * 2, 4),
            shape: vec![3, 2],
            conjugate: false,
        },
        OwnedInput::Real {
            data: patterned_real(2 * 2, 5),
            shape: vec![2, 2],
        },
    ];
    let tree = mixed_pass_ride_merge_tree();

    let native = execute_native_tree(&inputs, &ixs, &iy, tree.clone());
    let (actual, output) = execute_tree_realified(&inputs, &ixs, &iy, &tree);
    let (cascade, cascade_output) = execute_realified(&inputs, &ixs, &iy, Execution::Greedy);

    assert_shape_matches_output(&actual, output, native.shape());
    let recovered = realify_support::recover_complex(&actual, output);
    assert_complex_vec_close(&recovered, &native.to_vec(), 1e-10);
    assert_complex_vec_close(
        &recovered,
        &realify_support::recover_complex(&cascade, cascade_output),
        1e-10,
    );
}

#[test]
fn deep_merges_and_conjugation_match_native_sandwich() {
    let psi = vec![
        Complex64::new(1.0, 2.0),
        Complex64::new(-0.5, 0.25),
        Complex64::new(3.0, -1.0),
    ];
    let hermitian = vec![
        Complex64::new(2.0, 0.0),
        Complex64::new(1.0, -1.0),
        Complex64::new(0.0, 2.0),
        Complex64::new(1.0, 1.0),
        Complex64::new(3.0, 0.0),
        Complex64::new(4.0, -0.5),
        Complex64::new(0.0, -2.0),
        Complex64::new(4.0, 0.5),
        Complex64::new(5.0, 0.0),
    ];
    let inputs = vec![
        OwnedInput::Complex {
            data: psi.clone(),
            shape: vec![3],
            conjugate: true,
        },
        OwnedInput::Complex {
            data: hermitian,
            shape: vec![3, 3],
            conjugate: false,
        },
        OwnedInput::Complex {
            data: psi,
            shape: vec![3],
            conjugate: false,
        },
    ];
    let ixs = vec![vec![0], vec![0, 1], vec![1]];
    let left = node(
        NestedEinsum::leaf(0),
        &ixs[0],
        NestedEinsum::leaf(1),
        &ixs[1],
        &[1],
    );
    let tree = node(left, &[1], NestedEinsum::leaf(2), &ixs[2], &[]);

    let native = execute_native_tree(&inputs, &ixs, &[], tree.clone());
    let (plan, tensors) = prepare_tree_realified(&inputs, &ixs, &[], &tree);
    let refs = tensors.iter().collect::<Vec<_>>();
    let actual = plan.einsum.execute::<Standard<f64>, f64, Cpu>(&refs);
    let recovered = recover_complex(&actual);

    assert_eq!(plan.factor_vertex_positions.len(), 2);
    assert_eq!(actual.shape(), &[2]);
    assert_complex_vec_close(&recovered, &native.to_vec(), 1e-10);
    assert_relative_eq!(recovered[0].im, 0.0, epsilon = 1e-11);
}

#[test]
fn sliced_skeleton_axis_of_extent_one_matches_native() {
    let ixs = vec![vec![0, 1, 2], vec![0, 1, 2]];
    let inputs = vec![
        OwnedInput::Complex {
            data: patterned_complex(2 * 3, 21),
            shape: vec![2, 1, 3],
            conjugate: false,
        },
        OwnedInput::Complex {
            data: patterned_complex(2 * 3, 22),
            shape: vec![2, 1, 3],
            conjugate: false,
        },
    ];
    let tree = node(
        NestedEinsum::leaf(0),
        &ixs[0],
        NestedEinsum::leaf(1),
        &ixs[1],
        &[],
    );

    let native = execute_native_tree(&inputs, &ixs, &[], tree.clone());
    let (actual, output) = execute_tree_realified(&inputs, &ixs, &[], &tree);

    assert_eq!(actual.shape(), &[2]);
    assert_complex_vec_close(
        &realify_support::recover_complex(&actual, output),
        &native.to_vec(),
        1e-10,
    );
}

#[test]
fn merge_normalizes_repeated_and_private_source_labels() {
    let ixs = vec![vec![0, 0, 1], vec![0]];
    let inputs = vec![
        OwnedInput::Complex {
            data: patterned_complex(2 * 2 * 3, 31),
            shape: vec![2, 2, 3],
            conjugate: false,
        },
        OwnedInput::Complex {
            data: patterned_complex(2, 32),
            shape: vec![2],
            conjugate: false,
        },
    ];
    let tree = node(
        NestedEinsum::leaf(0),
        &ixs[0],
        NestedEinsum::leaf(1),
        &ixs[1],
        &[],
    );

    let native = execute_native_tree(&inputs, &ixs, &[], tree.clone());
    let (actual, output) = execute_tree_realified(&inputs, &ixs, &[], &tree);

    assert_eq!(actual.shape(), &[2]);
    assert_complex_vec_close(
        &realify_support::recover_complex(&actual, output),
        &native.to_vec(),
        1e-10,
    );
}

#[test]
fn installed_tree_autodiff_matches_native_complex_and_finite_differences() {
    let ixs = vec![vec![0], vec![0]];
    let inputs = vec![
        OwnedInput::Complex {
            data: vec![Complex64::new(0.75, -0.5), Complex64::new(-1.25, 0.5)],
            shape: vec![2],
            conjugate: false,
        },
        OwnedInput::Complex {
            data: vec![Complex64::new(1.5, 0.25), Complex64::new(-0.25, -2.0)],
            shape: vec![2],
            conjugate: false,
        },
    ];
    let tree = node(
        NestedEinsum::leaf(0),
        &ixs[0],
        NestedEinsum::leaf(1),
        &ixs[1],
        &[],
    );

    let native_tensors = inputs
        .iter()
        .map(OwnedInput::native_tensor)
        .collect::<Vec<_>>();
    let native_refs = native_tensors.iter().collect::<Vec<_>>();
    let mut native_code = Einsum::new(ixs.clone(), vec![], infer_sizes(&inputs, &ixs));
    native_code.set_contraction_tree(tree.clone());
    let (native_result, native_gradients) =
        cost_and_gradient::<Standard<Complex64>, _, _>(&native_code, &native_refs, None);

    let (plan, tensors) = prepare_tree_realified(&inputs, &ixs, &[], &tree);
    let refs = tensors.iter().collect::<Vec<_>>();
    let seed = Tensor::<f64, Cpu>::from_data(&[1.0, 0.0], &[2]);
    let (actual, gradients) =
        cost_and_gradient::<Standard<f64>, _, _>(&plan.einsum, &refs, Some(&seed));

    assert_complex_vec_close(&recover_complex(&actual), &native_result.to_vec(), 1e-12);
    for input_index in 0..inputs.len() {
        // `backward.rs` returns the holomorphic complex derivative. A gradient of
        // the real objective Re(y) is its conjugate when represented as one complex
        // number, which is exactly what the two real channels encode.
        let expected = native_gradients[input_index]
            .to_vec()
            .into_iter()
            .map(|value| value.conj())
            .collect::<Vec<_>>();
        assert_complex_vec_close(&recover_complex(&gradients[input_index]), &expected, 1e-10);
    }

    let base = tensors[0].to_vec();
    let epsilon = 1e-6;
    for entry in 0..base.len() {
        let mut plus = base.clone();
        let mut minus = base.clone();
        plus[entry] += epsilon;
        minus[entry] -= epsilon;
        let finite_difference = (real_loss(&plan.einsum, &tensors, &plus)
            - real_loss(&plan.einsum, &tensors, &minus))
            / (2.0 * epsilon);
        assert_relative_eq!(
            gradients[0].to_vec()[entry],
            finite_difference,
            epsilon = 1e-8,
            max_relative = 1e-8
        );
    }
}

fn real_loss(einsum: &Einsum<usize>, tensors: &[Tensor<f64, Cpu>], first: &[f64]) -> f64 {
    let replacement = Tensor::<f64, Cpu>::from_data(first, tensors[0].shape());
    let refs = tensors
        .iter()
        .enumerate()
        .map(|(index, tensor)| if index == 0 { &replacement } else { tensor })
        .collect::<Vec<_>>();
    einsum.execute::<Standard<f64>, f64, Cpu>(&refs).to_vec()[0]
}

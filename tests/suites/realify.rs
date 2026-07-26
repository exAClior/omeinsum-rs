use approx::assert_relative_eq;
use num_complex::Complex64;
use omeinsum::realify::mul_vertex_tensor;
use omeinsum::{
    einsum, einsum_with_grad, realify_code, realify_data, realify_einsum, Cpu, RealifiedOutput,
    Standard, Tensor,
};

use super::realify_support::*;

#[test]
fn exact_integer_matmul_matches_design_vector() {
    let a = vec![
        Complex64::new(1.0, 2.0),
        Complex64::new(0.0, -1.0),
        Complex64::new(3.0, 0.0),
        Complex64::new(2.0, -1.0),
    ];
    let b = vec![
        Complex64::new(2.0, 0.0),
        Complex64::new(1.0, -1.0),
        Complex64::new(0.0, 1.0),
        Complex64::new(4.0, 0.0),
    ];
    let inputs = vec![
        OwnedInput::Complex {
            data: a,
            shape: vec![2, 2],
            conjugate: false,
        },
        OwnedInput::Complex {
            data: b,
            shape: vec![2, 2],
            conjugate: false,
        },
    ];
    let ixs = vec![vec![0, 1], vec![1, 2]];
    let iy = vec![0, 2];
    let expected_realified = vec![5.0, 1.0, 10.0, 9.0, 1.0, -5.0, 1.0, -4.0];

    let (result, output) = execute_realified(&inputs, &ixs, &iy, Execution::Unoptimized);

    assert_eq!(output, RealifiedOutput::ReImAxis);
    assert_eq!(result.shape(), &[2, 2, 2]);
    assert_eq!(result.to_vec(), expected_realified);

    let borrowed = borrowed_inputs(&inputs);
    let (greedy_result, greedy_output) = realify_einsum(&borrowed, &ixs, &iy, Cpu);
    assert_eq!(greedy_output, RealifiedOutput::ReImAxis);
    assert_eq!(greedy_result.to_vec(), expected_realified);
}

#[test]
fn mixed_network_matches_native_for_unoptimized_greedy_and_treesa() {
    let ixs = vec![
        vec![0, 1],
        vec![1, 2],
        vec![2, 3],
        vec![3, 4],
        vec![4, 5],
        vec![5, 0],
    ];
    let iy = vec![3, 0];
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
        OwnedInput::Complex {
            data: patterned_complex(2 * 3, 3),
            shape: vec![2, 3],
            conjugate: false,
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
        OwnedInput::Complex {
            data: patterned_complex(2 * 2, 6),
            shape: vec![2, 2],
            conjugate: true,
        },
    ];

    for execution in [Execution::Unoptimized, Execution::Greedy, Execution::TreeSa] {
        assert_realified_matches_native(&inputs, &ixs, &iy, execution, 1e-10);
    }

    let native_greedy = execute_native(&inputs, &ixs, &iy, Execution::Greedy);
    let borrowed = borrowed_inputs(&inputs);
    let (public_result, public_output) = realify_einsum(&borrowed, &ixs, &iy, Cpu);
    assert_shape_matches_output(&public_result, public_output, native_greedy.shape());
    assert_complex_vec_close(
        &recover_complex(&public_result, public_output),
        &native_greedy.to_vec(),
        1e-10,
    );
}

#[test]
fn all_real_network_is_identity_transform() {
    let a_data = patterned_real(2 * 3, 1);
    let b_data = patterned_real(3 * 4, 2);
    let c_data = patterned_real(4 * 2, 3);
    let inputs = vec![
        OwnedInput::Real {
            data: a_data.clone(),
            shape: vec![2, 3],
        },
        OwnedInput::Real {
            data: b_data.clone(),
            shape: vec![3, 4],
        },
        OwnedInput::Real {
            data: c_data.clone(),
            shape: vec![4, 2],
        },
    ];
    let ixs = vec![vec![0, 1], vec![1, 2], vec![2, 3]];
    let iy = vec![3, 0];
    let borrowed = borrowed_inputs(&inputs);

    let (realified, output) = realify_einsum(&borrowed, &ixs, &iy, Cpu);

    let a = Tensor::<f64, Cpu>::from_data(&a_data, &[2, 3]);
    let b = Tensor::<f64, Cpu>::from_data(&b_data, &[3, 4]);
    let c = Tensor::<f64, Cpu>::from_data(&c_data, &[4, 2]);
    let expected = einsum::<Standard<f64>, _, _>(
        &[&a, &b, &c],
        &[ixs[0].as_slice(), ixs[1].as_slice(), ixs[2].as_slice()],
        &iy,
    );

    assert_eq!(output, RealifiedOutput::Real);
    assert_eq!(realified.shape(), expected.shape());
    assert_eq!(realified.to_vec(), expected.to_vec());
}

#[test]
fn single_complex_input_partial_trace_keeps_one_re_im_axis() {
    let inputs = vec![OwnedInput::Complex {
        data: patterned_complex(2 * 3 * 3 * 2, 7),
        shape: vec![2, 3, 3, 2],
        conjugate: false,
    }];
    let ixs = vec![vec![0, 1, 1, 2]];
    let iy = vec![2, 0];

    assert_realified_matches_native(&inputs, &ixs, &iy, Execution::Unoptimized, 1e-12);
    assert_realified_matches_native(&inputs, &ixs, &iy, Execution::Greedy, 1e-12);
}

#[test]
fn single_empty_complex_input_preserves_output_and_reduction_semantics() {
    let inputs = vec![OwnedInput::Complex {
        data: vec![],
        shape: vec![0],
        conjugate: false,
    }];
    let ixs = vec![vec![0]];

    let (retained, retained_output) = realify_einsum(&borrowed_inputs(&inputs), &ixs, &[0], Cpu);
    assert_eq!(retained_output, RealifiedOutput::ReImAxis);
    assert_eq!(retained.shape(), &[0, 2]);
    assert!(retained.to_vec().is_empty());

    let (reduced, reduced_output) = realify_einsum(&borrowed_inputs(&inputs), &ixs, &[], Cpu);
    assert_eq!(reduced_output, RealifiedOutput::ReImAxis);
    assert_eq!(reduced.shape(), &[2]);
    assert_eq!(reduced.to_vec(), vec![0.0, 0.0]);
}

#[test]
fn realified_execution_keeps_large_source_and_generated_labels_distinct() {
    let contracted = usize::MAX - 1;
    let retained = u32::MAX as usize;
    let inputs = vec![
        OwnedInput::Complex {
            data: vec![
                Complex64::new(1.0, 1.0),
                Complex64::new(2.0, -1.0),
                Complex64::new(3.0, 0.5),
                Complex64::new(4.0, 2.0),
            ],
            shape: vec![2, 2],
            conjugate: false,
        },
        OwnedInput::Complex {
            data: vec![Complex64::new(2.0, -1.0), Complex64::new(-0.5, 3.0)],
            shape: vec![2],
            conjugate: false,
        },
    ];
    let ixs = vec![vec![retained, contracted], vec![contracted]];

    assert_realified_matches_native(&inputs, &ixs, &[retained], Execution::Greedy, 1e-12);
}

#[test]
fn scalar_output_network_returns_re_im_vector() {
    let inputs = vec![
        OwnedInput::Complex {
            data: patterned_complex(4, 1),
            shape: vec![2, 2],
            conjugate: false,
        },
        OwnedInput::Complex {
            data: patterned_complex(4, 2),
            shape: vec![2, 2],
            conjugate: false,
        },
        OwnedInput::Complex {
            data: patterned_complex(4, 3),
            shape: vec![2, 2],
            conjugate: false,
        },
    ];
    let ixs = vec![vec![0, 1], vec![1, 2], vec![2, 0]];
    let iy = vec![];

    let native = execute_native(&inputs, &ixs, &iy, Execution::Greedy);
    let (realified, output) = execute_realified(&inputs, &ixs, &iy, Execution::Greedy);

    assert_eq!(native.shape(), &[] as &[usize]);
    assert_eq!(output, RealifiedOutput::ReImAxis);
    assert_eq!(realified.shape(), &[2]);
    assert_complex_vec_close(
        &recover_complex(&realified, output),
        &native.to_vec(),
        1e-10,
    );
}

#[test]
fn tdvp_shaped_binary_contraction_matches_native_complex_cpu() {
    let chi = 32;
    let mpo_bond_dim = 9;
    let inputs = vec![
        OwnedInput::Complex {
            data: patterned_complex(chi * mpo_bond_dim * chi, 11),
            shape: vec![chi, mpo_bond_dim, chi],
            conjugate: false,
        },
        OwnedInput::Complex {
            data: patterned_complex(chi * 2 * chi, 12),
            shape: vec![chi, 2, chi],
            conjugate: false,
        },
    ];
    let ixs = vec![vec![0, 1, 2], vec![0, 3, 4]];
    let iy = vec![1, 2, 3, 4];

    assert_realified_matches_native(&inputs, &ixs, &iy, Execution::Greedy, 1e-8);
}

#[test]
fn hermitian_sandwich_conjugates_bra_and_matches_native() {
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
    let iy = vec![];
    let native = execute_native(&inputs, &ixs, &iy, Execution::Greedy);
    let (realified, output) = execute_realified(&inputs, &ixs, &iy, Execution::Greedy);
    let actual = recover_complex(&realified, output);

    assert_complex_vec_close(&actual, &native.to_vec(), 1e-10);
    assert_relative_eq!(actual[0].im, 0.0, epsilon = 1e-12, max_relative = 1e-12);
}

#[test]
fn backward_on_realified_scalar_network_matches_finite_differences() {
    let a_complex = vec![Complex64::new(0.75, -0.5), Complex64::new(-1.25, 0.5)];
    let b_complex = vec![Complex64::new(1.5, 0.25), Complex64::new(-0.25, -2.0)];
    let ixs = vec![vec![0], vec![0]];
    let iy = vec![];
    let inputs = vec![
        OwnedInput::Complex {
            data: a_complex.clone(),
            shape: vec![2],
            conjugate: false,
        },
        OwnedInput::Complex {
            data: b_complex.clone(),
            shape: vec![2],
            conjugate: false,
        },
    ];
    let is_complex = vec![true, true];
    let mut plan = realify_code(&ixs, &iy, &infer_sizes(&inputs, &ixs), &is_complex);
    let a_data = realify_data(&a_complex, false);
    let b_data = realify_data(&b_complex, false);
    let a = Tensor::<f64, Cpu>::from_data(&a_data, &[2, 2]);
    let b = Tensor::<f64, Cpu>::from_data(&b_data, &[2, 2]);
    let m = mul_vertex_tensor::<f64, Cpu>(Cpu);
    let tensor_refs = vec![&a, &b, &m];
    let ix_refs: Vec<&[usize]> = plan.einsum.ixs.iter().map(Vec::as_slice).collect();

    let (result, grad_fn) =
        einsum_with_grad::<Standard<f64>, _, _>(&tensor_refs, &ix_refs, &plan.einsum.iy);
    let grad_output = Tensor::<f64, Cpu>::from_data(&[1.0, 0.0], &[2]);
    let grads = grad_fn.backward::<Standard<f64>>(&grad_output, &tensor_refs);

    assert_eq!(result.shape(), &[2]);
    assert_eq!(grads.len(), 3);
    let analytic = grads[0].to_vec();
    let epsilon = 1e-6;
    for entry in [0usize, 1, 2] {
        let mut plus = a_data.clone();
        let mut minus = a_data.clone();
        plus[entry] += epsilon;
        minus[entry] -= epsilon;
        let finite_difference = (loss_re(&plus, &b_data, &mut plan)
            - loss_re(&minus, &b_data, &mut plan))
            / (2.0 * epsilon);
        assert_relative_eq!(
            analytic[entry],
            finite_difference,
            epsilon = 1e-7,
            max_relative = 1e-7,
        );
    }
}

fn loss_re(a_data: &[f64], b_data: &[f64], plan: &mut omeinsum::RealifyPlan) -> f64 {
    let a = Tensor::<f64, Cpu>::from_data(a_data, &[2, 2]);
    let b = Tensor::<f64, Cpu>::from_data(b_data, &[2, 2]);
    let m = mul_vertex_tensor::<f64, Cpu>(Cpu);
    let refs = vec![&a, &b, &m];
    plan.einsum.optimize_greedy();
    plan.einsum
        .execute::<Standard<f64>, f64, Cpu>(&refs)
        .to_vec()[0]
}

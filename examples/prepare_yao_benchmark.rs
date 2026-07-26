//! Archive an optimized and sliced original complex Yao network.
#[path = "support/yao_benchmark.rs"]
mod format;
use format::*;
use omeco::{
    contraction_complexity, green_pipeline, optimize_code, slice_code, EinCode, GreenAnnealer,
    TreeSA, TreeSASlicer,
};
use std::{collections::HashMap, fs::File, io::BufReader};

struct Args {
    input: String,
    output: String,
    optimizer: String,
    optimizer_sc: f64,
    slicer_sc: f64,
    max: usize,
    opt_profile: String,
    opt_trials: usize,
    opt_iters: usize,
    slice_trials: usize,
    slice_iters: usize,
    anneal_steps: usize,
    study_out: Option<String>,
    // W4: artifact scalar dtype. "f32" downcasts the yao-TN's f64 gate data
    // (benchmark default); "c64" keeps f64 for the reference runs.
    dtype: String,
    // W4: reuse an existing artifact's archived tree/cuts byte-identically
    // instead of re-optimizing (same-plan rule for the c64 reference set).
    import_tree: Option<String>,
}
fn args() -> Args {
    let mut a = std::env::args().skip(1);
    let input = a
        .next()
        .expect("usage: prepare_yao_benchmark INPUT OUTPUT [options]");
    let output = a.next().expect("missing OUTPUT");
    let mut x = Args {
        input,
        output,
        // treesa: one-stage TreeSA (v3 policy). green-sa: TreeSA init + the
        // paper's second-stage green-aware anneal (W2, omeco::GreenAnnealer).
        optimizer: "treesa".into(),
        // The optimizer's sc target shapes the tree search; it must not exclude
        // good high-sc trees (the paper's Table-1 trees reach sc = 35). Memory
        // is enforced by the slicer below, not by the optimizer.
        optimizer_sc: 40.0,
        // Per-slice peak: 2^26 f32 elements = 256 MiB (tree-real 2x -> ~512 MiB).
        slicer_sc: 26.0,
        max: 1 << 24,
        opt_profile: "fast".into(),
        opt_trials: 1,
        opt_iters: 20,
        slice_trials: 1,
        slice_iters: 5,
        // Paper's anneal length (greensa.jl); polish pass uses half.
        anneal_steps: 600_000,
        study_out: None,
        dtype: "f32".into(),
        import_tree: None,
    };
    while let Some(f) = a.next() {
        let v = a.next().unwrap_or_else(|| panic!("{f} requires a value"));
        match f.as_str() {
            "--sc-target" => {
                eprintln!("warning: --sc-target constrains BOTH optimizer and slicer; prefer --optimizer-sc-target and --slicer-sc-target");
                x.optimizer_sc = v.parse().unwrap();
                x.slicer_sc = x.optimizer_sc;
            }
            "--optimizer-sc-target" => x.optimizer_sc = v.parse().unwrap(),
            "--slicer-sc-target" => x.slicer_sc = v.parse().unwrap(),
            "--max-assignments" => x.max = v.parse().unwrap(),
            "--optimizer-profile" => x.opt_profile = v,
            "--optimizer-trials" => x.opt_trials = v.parse().unwrap(),
            "--optimizer-iters" => x.opt_iters = v.parse().unwrap(),
            "--slicer-trials" => x.slice_trials = v.parse().unwrap(),
            "--slicer-iters" => x.slice_iters = v.parse().unwrap(),
            "--optimizer" => x.optimizer = v,
            "--anneal-steps" => x.anneal_steps = v.parse().unwrap(),
            "--study-out" => x.study_out = Some(v),
            "--dtype" => x.dtype = v,
            "--import-tree" => x.import_tree = Some(v),
            _ => panic!("unknown option {f}"),
        }
    }
    x
}
fn label(x: &str) -> usize {
    x.parse()
        .unwrap_or_else(|e| panic!("invalid label {x}: {e}"))
}
fn column_major<T: Copy>(data: &[T], shape: &[usize]) -> Vec<T> {
    assert_eq!(data.len(), shape.iter().product::<usize>());
    (0..data.len())
        .map(|mut ci| {
            let mut c = Vec::new();
            for &n in shape {
                c.push(ci % n);
                ci /= n;
            }
            c.iter().zip(shape).fold(0, |i, (&v, &n)| i * n + v)
        })
        .map(|i| data[i])
        .collect()
}
fn metric(x: omeco::ContractionComplexity) -> Complexity {
    Complexity {
        log2_flops: x.tc,
        log2_peak_elements: x.sc,
        log2_readwrites: x.rwc,
    }
}

/// Dump the fig-pipe three-pass dataset for one circuit (W2). All costs come
/// from a single TreeSA start, so the four passes are directly comparable.
fn write_green_study(
    path: &str,
    a: &Args,
    code: &EinCode<usize>,
    sizes: &HashMap<usize, usize>,
    is_complex: &[bool],
    outcome: &omeco::GreenPipeOutcome<usize>,
    annealed: &omeco::ContractionComplexity,
) {
    let start_complexity = contraction_complexity(&outcome.start, sizes, &code.ixs);
    // Cross-check the annealer's (m, r) bookkeeping against audit_tree on the
    // exported unsliced tree (W2 spec test iii, campaign side).
    let audit = audit_tree(
        &TreeNode::from_nested(&outcome.full_anneal),
        is_complex,
        sizes,
        annealed.sc,
    );
    let base = outcome.full_anneal_pass_volume
        + outcome.full_anneal_ride_volume
        + outcome.full_anneal_merge_volume;
    let annealer_m = outcome.full_anneal_merge_volume / base;
    let annealer_r = outcome.full_anneal_ride_volume / base;
    let fraction_diff = (audit.m - annealer_m)
        .abs()
        .max((audit.r - annealer_r).abs());
    assert!(
        fraction_diff < 1e-9,
        "annealer/audit fraction mismatch: {fraction_diff}"
    );
    let log2 = |x: f64| x.log2();
    let best = outcome.best_real_cost;
    let study = serde_json::json!({
        "format": "green-sa-study-v1",
        "circuit": std::path::Path::new(&a.input)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(""),
        "tensors": code.num_tensors(),
        "labels": code.unique_labels().len(),
        "structurally_complex": is_complex.iter().filter(|&&c| c).count(),
        "factors": {"merge": 3.0, "ride": 2.0},
        "schedule": {"nsteps": a.anneal_steps, "t0": 1.0, "t1": 0.005, "seeds": [42, 7, 2026]},
        "polish_schedule": {"nsteps": a.anneal_steps / 2, "t0": 0.03, "t1": 0.002, "seeds": [42, 7, 2026]},
        "initializer": {
            "algorithm": "omeco::TreeSA",
            "profile": a.opt_profile,
            "ntrials": a.opt_trials,
            "niters": a.opt_iters,
            "sc_target": a.optimizer_sc,
            "seed_base": 42
        },
        "costs": {
            "green_blind": outcome.green_blind_cost,
            "best_real": best,
            "convert_only": outcome.convert_only_cost,
            "polished": outcome.polished_cost,
            "full_anneal": outcome.full_anneal_cost,
            "start_volume": outcome.start_volume
        },
        "costs_log2": {
            "green_blind": log2(outcome.green_blind_cost),
            "best_real": log2(best),
            "convert_only": log2(outcome.convert_only_cost),
            "polished": log2(outcome.polished_cost),
            "full_anneal": log2(outcome.full_anneal_cost),
            "start_volume": log2(outcome.start_volume)
        },
        "ratios_vs_best_real": {
            "green_blind": outcome.green_blind_cost / best,
            "convert_only": outcome.convert_only_cost / best,
            "polished": outcome.polished_cost / best,
            "full_anneal": outcome.full_anneal_cost / best
        },
        "full_anneal_fractions": {
            "pass": outcome.full_anneal_pass_volume / base,
            "ride": outcome.full_anneal_ride_volume / base,
            "merge": outcome.full_anneal_merge_volume / base
        },
        "audit_crosscheck": {
            "m": audit.m,
            "r": audit.r,
            "predicted_overhead": audit.predicted_overhead,
            "annealer_m": annealer_m,
            "annealer_r": annealer_r,
            "fraction_abs_diff": fraction_diff
        },
        "start_tree": {"tc": start_complexity.tc, "sc": start_complexity.sc},
        "annealed_tree": {"tc": annealed.tc, "sc": annealed.sc},
        "seconds": {
            "treesa_init": outcome.seconds.0,
            "blind_anneal": outcome.seconds.1,
            "aware_anneal": outcome.seconds.2,
            "polish": outcome.seconds.3
        }
    });
    serde_json::to_writer_pretty(File::create(path).unwrap(), &study).unwrap();
}

fn main() {
    let a = args();
    let source: YaoNetwork = serde_json::from_reader(BufReader::new(File::open(&a.input).unwrap()))
        .expect("invalid Yao JSON");
    assert_eq!(source.format, "yao-tn-v1");
    let ixs = source
        .eincode
        .input_indices
        .iter()
        .map(|x| x.iter().map(|s| label(s)).collect())
        .collect::<Vec<Vec<_>>>();
    let iy = source
        .eincode
        .output_indices
        .iter()
        .map(|s| label(s))
        .collect::<Vec<_>>();
    let sizes = source
        .size_dict
        .iter()
        .map(|(k, v)| (label(k), *v))
        .collect::<HashMap<_, _>>();
    assert!(
        matches!(a.dtype.as_str(), "f32" | "c64"),
        "--dtype must be f32 or c64"
    );
    let c64 = a.dtype == "c64";
    assert_eq!(ixs.len(), source.tensors.len());
    let mut tensors = Vec::new();
    for (i, t) in source.tensors.iter().enumerate() {
        let n = t.shape.iter().product::<usize>();
        assert_eq!(t.shape, ixs[i].iter().map(|l| sizes[l]).collect::<Vec<_>>());
        assert_eq!(t.data_re.len(), n);
        assert_eq!(t.data_im.len(), n);
        tensors.push(BenchmarkTensor {
            shape: t.shape.clone(),
            data_re: if c64 {
                Vec::new()
            } else {
                column_major(
                    &t.data_re.iter().map(|x| *x as f32).collect::<Vec<_>>(),
                    &t.shape,
                )
            },
            data_im: if c64 {
                Vec::new()
            } else {
                column_major(
                    &t.data_im.iter().map(|x| *x as f32).collect::<Vec<_>>(),
                    &t.shape,
                )
            },
            data_re_f64: c64.then(|| column_major(&t.data_re, &t.shape)),
            data_im_f64: c64.then(|| column_major(&t.data_im, &t.shape)),
            structurally_complex: t.data_im.iter().any(|x| *x != 0.0),
        });
    }
    // W4: --import-tree loads the reference artifact and reuses its archived
    // tree/cuts; optimization and slicing below are skipped.
    let imported: Option<BenchmarkNetwork> = a.import_tree.as_ref().map(|path| {
        let mut d =
            serde_json::Deserializer::from_reader(BufReader::new(File::open(path).unwrap()));
        d.disable_recursion_limit();
        let r: BenchmarkNetwork =
            serde::Deserialize::deserialize(&mut d).expect("invalid reference artifact");
        assert_eq!(r.format, "omeinsum-yao-benchmark-v2");
        assert_eq!(
            r.eincode.input_indices, ixs,
            "--import-tree artifact's input labels do not match the input network"
        );
        assert_eq!(
            r.eincode.output_indices, iy,
            "--import-tree artifact's output labels do not match the input network"
        );
        eprintln!("--import-tree {path}: reusing archived tree/cuts; optimizer and slicer flags ignored");
        r
    });
    let code = EinCode::new(ixs.clone(), iy.clone());
    let optimizer = match a.opt_profile.as_str() {
        "default" => TreeSA::default(),
        "fast" => TreeSA::fast(),
        profile => panic!("unknown optimizer profile {profile}"),
    }
    .with_ntrials(a.opt_trials)
    .with_niters(a.opt_iters)
    .with_sc_target(a.optimizer_sc);
    let is_complex: Vec<bool> = tensors.iter().map(|t| t.structurally_complex).collect();
    let (original, study) = if let Some(r) = &imported {
        (r.contraction_order.to_nested(), None)
    } else {
        match a.optimizer.as_str() {
        "treesa" => (
            optimize_code(&code, &sizes, &optimizer).expect("TreeSA produced no tree"),
            None,
        ),
        // W2: TreeSA init -> the paper's second-stage green-aware anneal
        // (full 3-seed x 600k-step pipeline). Also produces the fig-pipe
        // three-pass dataset from the same TreeSA start.
        "green-sa" => {
            let annealer = GreenAnnealer::default()
                .with_nsteps(a.anneal_steps)
                .with_initializer(optimizer.clone());
            let outcome = green_pipeline(&code, &sizes, &is_complex, &annealer)
                .expect("green pipeline produced no tree");
            (outcome.full_anneal.clone(), Some(outcome))
        }
        other => panic!("unknown optimizer {other}"),
        }
    };
    if a.study_out.is_some() && study.is_none() {
        panic!("--study-out requires --optimizer green-sa (not available with --import-tree)");
    }
    let unsliced = contraction_complexity(&original, &sizes, &ixs);
    if let (Some(path), Some(outcome)) = (&a.study_out, &study) {
        write_green_study(path, &a, &code, &sizes, &is_complex, outcome, &unsliced);
    }
    let (sliced_nested, mut cuts) = if let Some(r) = &imported {
        (original.clone(), r.cuts.clone())
    } else {
        let slicer = TreeSASlicer::fast()
            .with_ntrials(a.slice_trials)
            .with_niters(a.slice_iters)
            .with_sc_target(a.slicer_sc);
        let sliced = slice_code(&original, &sizes, &slicer, &ixs).expect("TreeSASlicer failed");
        let mut cuts = sliced.slicing;
        cuts.sort_unstable();
        cuts.dedup();
        (sliced.eins, cuts)
    };
    cuts.sort_unstable();
    cuts.dedup();
    for c in &cuts {
        assert!(sizes.contains_key(c), "invalid physical cut {c}");
        assert!(!iy.contains(c), "cannot cut output label {c}");
    }
    let assignments = cuts
        .iter()
        .try_fold(1usize, |n, c| n.checked_mul(sizes[c]))
        .expect("assignment count overflow");
    assert!(
        assignments <= a.max,
        "assignment cap exceeded: {assignments} > {}",
        a.max
    );
    let tree = TreeNode::from_nested(&sliced_nested);
    let leaves = tree.validate(tensors.len()).expect("invalid tree");
    assert_eq!(
        leaves.len(),
        tensors.len(),
        "tree does not contain every tensor"
    );
    if let Some(r) = &imported {
        assert_eq!(
            assignments, r.assignment_count,
            "imported cuts do not reproduce the reference assignment count"
        );
    }
    let adjusted = cut_sizes(&sizes, &cuts);
    let per = contraction_complexity(&sliced_nested, &adjusted, &ixs);
    let total = Complexity {
        log2_flops: per.tc + (assignments as f64).log2(),
        log2_peak_elements: per.sc,
        log2_readwrites: per.rwc + (assignments as f64).log2(),
    };
    let mut audit = audit_tree(
        &tree,
        &tensors
            .iter()
            .map(|t| t.structurally_complex)
            .collect::<Vec<_>>(),
        &adjusted,
        per.sc,
    );
    audit.native_resident_input_cache_elements = tensors
        .iter()
        .map(|t| t.shape.iter().product::<usize>())
        .sum();
    audit.native_resident_input_cache_bytes =
        audit.native_resident_input_cache_elements * if c64 { 16 } else { 8 };
    audit.tree_real_resident_input_cache_elements = tensors
        .iter()
        .map(|t| t.shape.iter().product::<usize>() * (1 + usize::from(t.structurally_complex)))
        .sum();
    audit.tree_real_resident_input_cache_bytes =
        audit.tree_real_resident_input_cache_elements * if c64 { 8 } else { 4 };
    let optimizer_cfg = imported
        .as_ref()
        .map(|r| r.optimizer.clone())
        .unwrap_or(OptimizerConfig {
            algorithm: match a.optimizer.as_str() {
                "green-sa" => "omeco::GreenSA(TreeSA-init)".into(),
                _ => "omeco::TreeSA".into(),
            },
            version: "0.3.0".into(),
            ntrials: a.opt_trials,
            niters: a.opt_iters,
            sc_target: a.optimizer_sc,
            seed_base: 42,
            anneal_steps: if a.optimizer == "green-sa" {
                a.anneal_steps
            } else {
                0
            },
            anneal_t0: if a.optimizer == "green-sa" { 1.0 } else { 0.0 },
            anneal_t1: if a.optimizer == "green-sa" { 0.005 } else { 0.0 },
            anneal_seeds: if a.optimizer == "green-sa" {
                vec![42, 7, 2026]
            } else {
                vec![]
            },
            merge_factor: if a.optimizer == "green-sa" { 3.0 } else { 0.0 },
            ride_factor: if a.optimizer == "green-sa" { 2.0 } else { 0.0 },
        });
    let slicer_cfg = imported
        .as_ref()
        .map(|r| r.slicer.clone())
        .unwrap_or(SlicerConfig {
            algorithm: "omeco::TreeSASlicer".into(),
            ntrials: a.slice_trials,
            niters: a.slice_iters,
            sc_target: a.slicer_sc,
            optimization_ratio: 1.0,
            seed_base: 42,
        });
    let artifact = BenchmarkNetwork {
        format: "omeinsum-yao-benchmark-v2".into(),
        source_format: source.format,
        source_mode: source.mode,
        dtype: a.dtype.clone(),
        optimizer: optimizer_cfg,
        slicer: slicer_cfg,
        eincode: BenchmarkEinCode {
            input_indices: ixs,
            output_indices: iy,
        },
        tensors,
        size_dict: sizes,
        contraction_order: tree,
        cuts,
        assignment_count: assignments,
        complexity: ComplexityReport {
            physical_unsliced: metric(unsliced),
            physical_per_slice: metric(per),
            physical_total: total,
        },
        tree_real_audit: audit,
    };
    artifact.validate().expect("prepared artifact is invalid");
    serde_json::to_writer(File::create(&a.output).unwrap(), &artifact).unwrap();
    println!(
        "prepared={} cuts={} assignments={}",
        a.output,
        artifact
            .cuts
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(","),
        assignments
    );
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn layout() {
        assert_eq!(
            column_major(&[0, 1, 2, 3, 4, 5], &[2, 3]),
            vec![0, 3, 1, 4, 2, 5]
        );
    }
}

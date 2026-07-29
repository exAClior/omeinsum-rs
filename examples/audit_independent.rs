//! W6: independent-optimization half of the paper's `tab:audit` (campaign TODO W6).
//!
//! Loads one prepared benchmark artifact and asks the question the same-tree
//! audit cannot answer: does an optimizer set loose on the *realified* einsum
//! find the cost the law predicts? The artifact's complex einsum is exported
//! twice —
//!
//!   * tree-free cascade (`realify_code`): the canonical realified einsum;
//!     the optimizer has never seen its structure.
//!   * tree-following (`realify_tree_code` on the archived tree): merges
//!     expanded into Gauss-3M factor leaves where the archived tree merges
//!     complex subtrees. Its installed tree is evaluated as the
//!     same-structure reference; its flat code is then re-optimized as a
//!     second independent run.
//!
//! Both flat codes are optimized with the complex sibling's TreeSA policy
//! (default profile, 10 trials x 50 iters, sc-target 40; omeco derives each
//! trial's RNG seed from 42) and measured with `omeco::contraction_complexity`,
//! which also yields the first exact realified read-write numbers.
//!
//! Topology and complexity arithmetic only: tensor data is never touched.
#[path = "support/yao_benchmark.rs"]
mod format;
use format::*;
use omeinsum::{realify_code, realify_tree_code};
use omeco::{contraction_complexity, optimize_code, NestedEinsum, TreeSA};
use std::{collections::HashMap, fs::File, io::BufReader, time::Instant};

struct Args {
    input: String,
    output: String,
    profile: String,
    trials: usize,
    iters: usize,
    sc_target: f64,
}
fn args() -> Args {
    let mut a = std::env::args().skip(1);
    let input = a.next().expect("usage: audit_independent ARTIFACT OUT [options]");
    let output = a.next().expect("missing OUT");
    let mut x = Args {
        input,
        output,
        profile: "default".into(),
        trials: 10,
        iters: 50,
        sc_target: 40.0,
    };
    while let Some(f) = a.next() {
        let v = a.next().unwrap_or_else(|| panic!("{f} requires a value"));
        match f.as_str() {
            "--optimizer-profile" => x.profile = v,
            "--optimizer-trials" => x.trials = v.parse().unwrap(),
            "--optimizer-iters" => x.iters = v.parse().unwrap(),
            "--optimizer-sc-target" => x.sc_target = v.parse().unwrap(),
            _ => panic!("unknown option {f}"),
        }
    }
    x
}

fn metric(x: omeco::ContractionComplexity) -> Complexity {
    Complexity {
        log2_flops: x.tc,
        log2_peak_elements: x.sc,
        log2_readwrites: x.rwc,
    }
}

fn leaf_count(tree: &NestedEinsum<usize>) -> usize {
    match tree {
        NestedEinsum::Leaf { .. } => 1,
        NestedEinsum::Node { args, .. } => args.iter().map(leaf_count).sum(),
    }
}

/// Every label carried by the code must have a known extent.
fn assert_sizes(ixs: &[Vec<usize>], iy: &[usize], sizes: &HashMap<usize, usize>, what: &str) {
    for label in ixs.iter().flatten().chain(iy.iter()) {
        assert!(sizes.contains_key(label), "{what} label {label} has no extent");
    }
}

fn optimize_and_measure(
    ixs: &[Vec<usize>],
    iy: &[usize],
    sizes: &HashMap<usize, usize>,
    optimizer: &TreeSA,
) -> (Complexity, f64) {
    assert_sizes(ixs, iy, sizes, "exported code");
    let code = omeco::EinCode::new(ixs.to_vec(), iy.to_vec());
    let start = Instant::now();
    let tree = optimize_code(&code, sizes, optimizer).expect("TreeSA produced no tree");
    let seconds = start.elapsed().as_secs_f64();
    assert_eq!(
        leaf_count(&tree),
        ixs.len(),
        "optimized tree does not cover every realified tensor"
    );
    (metric(contraction_complexity(&tree, sizes, ixs)), seconds)
}

/// Deviation of an independently optimized log2 cost from the same-tree value:
/// linear percentage, the units of the paper's "<1%" prediction.
fn deviation_pct(independent: f64, same_tree: f64) -> f64 {
    ((independent - same_tree).exp2() - 1.0) * 100.0
}

fn run(network: &BenchmarkNetwork, optimizer: &TreeSA) -> serde_json::Value {
    network.validate().expect("artifact is invalid");
    assert!(
        network.cuts.is_empty(),
        "W6 audits the unsliced plan; sliced artifacts are out of scope"
    );
    let ixs = &network.eincode.input_indices;
    let iy = &network.eincode.output_indices;
    let sizes = &network.size_dict;
    let is_complex: Vec<bool> = network.tensors.iter().map(|t| t.structurally_complex).collect();
    let complex_leaves = is_complex.iter().filter(|&&c| c).count();

    // Complex side, recomputed from the archived tree (self-contained: the
    // stored physical_unsliced block is cross-checked, not trusted).
    let tree = network.contraction_order.to_nested();
    let complex = metric(contraction_complexity(&tree, sizes, ixs));
    let stored = &network.complexity.physical_unsliced;
    for (recomputed, stored) in [
        (complex.log2_flops, stored.log2_flops),
        (complex.log2_peak_elements, stored.log2_peak_elements),
        (complex.log2_readwrites, stored.log2_readwrites),
    ] {
        assert!(
            (recomputed - stored).abs() < 1e-9,
            "artifact complexity block disagrees with its archived tree"
        );
    }

    // Same-tree realified values: the law applied to the archived tree.
    let audit = audit_tree(
        &network.contraction_order,
        &is_complex,
        sizes,
        complex.log2_peak_elements,
    );
    let same_tc = complex.log2_flops + audit.predicted_overhead.log2();
    let same_sc = complex.log2_peak_elements + 1.0; // elements; bytes match complex64

    // Export 1: tree-free cascade. The genuinely independent run.
    let free = realify_code(ixs, iy, sizes, &is_complex);
    let (free_opt, free_seconds) = optimize_and_measure(
        &free.einsum.ixs,
        &free.einsum.iy,
        &free.einsum.size_dict,
        optimizer,
    );

    // Export 2: tree-following. The installed tree is the same-structure
    // reference; its flat code is re-optimized as a second independent run.
    let follow = realify_tree_code(&tree, ixs, iy, sizes, &is_complex);
    let installed = follow
        .einsum
        .contraction_tree()
        .expect("tree-following export installs its tree");
    assert_eq!(
        leaf_count(installed),
        follow.einsum.ixs.len(),
        "installed tree does not cover every realified tensor"
    );
    let follow_installed = metric(contraction_complexity(
        installed,
        &follow.einsum.size_dict,
        &follow.einsum.ixs,
    ));
    let (follow_opt, follow_seconds) = optimize_and_measure(
        &follow.einsum.ixs,
        &follow.einsum.iy,
        &follow.einsum.size_dict,
        optimizer,
    );

    let deltas = |ind: &Complexity| {
        serde_json::json!({
            "dtc_log2": ind.log2_flops - same_tc,
            "tc_deviation_pct": deviation_pct(ind.log2_flops, same_tc),
            "dsc_log2_vs_elements_plus1": ind.log2_peak_elements - same_sc,
            "drwc_log2_vs_complex": ind.log2_readwrites - complex.log2_readwrites,
        })
    };
    serde_json::json!({
        "format": "audit-independent-v1",
        "network": {
            "tensors": ixs.len(),
            "structurally_complex": complex_leaves,
            "labels": sizes.len(),
            "cuts": network.cuts.len(),
        },
        "complex_tree": {
            "tc": complex.log2_flops,
            "sc": complex.log2_peak_elements,
            "rwc": complex.log2_readwrites,
        },
        "same_tree_real": {
            "m": audit.m,
            "r": audit.r,
            "predicted_overhead": audit.predicted_overhead,
            "tc": same_tc,
            "sc_elements": same_sc,
            "rwc": null,
            "note": "same-tree realified rwc does not exist; this is the gap W6 closes",
        },
        "tree_free": {
            "tensors": free.einsum.ixs.len(),
            "mul_vertices": free.num_mul_vertices,
            "optimized": {
                "tc": free_opt.log2_flops,
                "sc": free_opt.log2_peak_elements,
                "rwc": free_opt.log2_readwrites,
                "seconds": free_seconds,
            },
        },
        "tree_following": {
            "tensors": follow.einsum.ixs.len(),
            "factor_leaves": follow.factor_vertex_positions.len() * 3,
            "installed": {
                "tc": follow_installed.log2_flops,
                "sc": follow_installed.log2_peak_elements,
                "rwc": follow_installed.log2_readwrites,
                "note": "same-structure reference: archived tree with Gauss-3M merge factorization",
            },
            "optimized": {
                "tc": follow_opt.log2_flops,
                "sc": follow_opt.log2_peak_elements,
                "rwc": follow_opt.log2_readwrites,
                "seconds": follow_seconds,
            },
        },
        "deltas_vs_same_tree": {
            "tree_free": deltas(&free_opt),
            "tree_following_optimized": deltas(&follow_opt),
            "tree_following_installed": deltas(&follow_installed),
        },
    })
}

fn main() {
    let a = args();
    let network: BenchmarkNetwork =
        serde_json::from_reader(BufReader::new(File::open(&a.input).unwrap()))
            .expect("invalid benchmark artifact");
    let optimizer = match a.profile.as_str() {
        "default" => TreeSA::default(),
        "fast" => TreeSA::fast(),
        profile => panic!("unknown optimizer profile {profile}"),
    }
    .with_ntrials(a.trials)
    .with_niters(a.iters)
    .with_sc_target(a.sc_target);
    let mut report = run(&network, &optimizer);
    report["circuit"] = serde_json::json!(std::path::Path::new(&a.input)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(""));
    report["policy"] = serde_json::json!({
        "algorithm": "omeco::TreeSA",
        "profile": a.profile,
        "ntrials": a.trials,
        "niters": a.iters,
        "sc_target": a.sc_target,
        "seed_base": 42,
        "note": "omeco derives each trial's RNG seed from 42; identical to the v5 artifact policy",
    });
    serde_json::to_writer_pretty(File::create(&a.output).unwrap(), &report).unwrap();
    let d = &report["deltas_vs_same_tree"];
    println!(
        "audited={} tree_free dtc={:+.4} ({:+.2}%) tree_following dtc={:+.4} ({:+.2}%)",
        a.input,
        d["tree_free"]["dtc_log2"],
        d["tree_free"]["tc_deviation_pct"],
        d["tree_following_optimized"]["dtc_log2"],
        d["tree_following_optimized"]["tc_deviation_pct"],
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two-tensor network, one contraction node. Leaf 0 carries [0,1], leaf 1
    /// carries [1], output [0]; sizes 2 and 3.
    fn tiny(complex_leaf0: bool) -> BenchmarkNetwork {
        let elements = 6;
        BenchmarkNetwork {
            format: "omeinsum-yao-benchmark-v2".into(),
            source_format: "yao-tn-v1".into(),
            source_mode: "overlap".into(),
            optimizer: Default::default(),
            slicer: Default::default(),
            eincode: BenchmarkEinCode {
                input_indices: vec![vec![0, 1], vec![1]],
                output_indices: vec![0],
            },
            tensors: vec![
                BenchmarkTensor {
                    shape: vec![2, 3],
                    data_re: vec![0.0; elements],
                    data_im: if complex_leaf0 {
                        vec![1.0; elements]
                    } else {
                        vec![0.0; elements]
                    },
                    structurally_complex: complex_leaf0,
                },
                BenchmarkTensor {
                    shape: vec![3],
                    data_re: vec![0.0; 3],
                    data_im: vec![0.0; 3],
                    structurally_complex: false,
                },
            ],
            size_dict: HashMap::from([(0, 2), (1, 3)]),
            contraction_order: TreeNode::Node {
                args: vec![
                    TreeNode::Leaf { tensor_index: 0 },
                    TreeNode::Leaf { tensor_index: 1 },
                ],
                input_indices: vec![vec![0, 1], vec![1]],
                output_indices: vec![0],
            },
            cuts: vec![],
            assignment_count: 1,
            complexity: ComplexityReport {
                // omeco convention: sc counts input tensors too (leaf 0 has
                // 6 elements), rwc sums child outputs + node output (6+3+2).
                physical_unsliced: Complexity {
                    log2_flops: 6f64.log2(),
                    log2_peak_elements: 6f64.log2(),
                    log2_readwrites: (11f64).log2(),
                },
                physical_per_slice: Default::default(),
                physical_total: Default::default(),
            },
            tree_real_audit: Default::default(),
        }
    }

    fn policy() -> TreeSA {
        TreeSA::default().with_ntrials(2).with_niters(5).with_sc_target(40.0)
    }

    #[test]
    fn all_real_exports_are_identity() {
        let report = run(&tiny(false), &policy());
        let c = &report["complex_tree"];
        // No complex leaves: predicted overhead is exactly 1, both exports
        // reproduce the complex einsum, and every measured cost matches.
        assert_eq!(report["same_tree_real"]["predicted_overhead"], 1.0);
        assert_eq!(report["tree_free"]["mul_vertices"], 0);
        assert_eq!(report["tree_following"]["factor_leaves"], 0);
        for path in [
            &report["tree_free"]["optimized"]["tc"],
            &report["tree_following"]["optimized"]["tc"],
            &report["tree_following"]["installed"]["tc"],
        ] {
            assert!((path.as_f64().unwrap() - c["tc"].as_f64().unwrap()).abs() < 1e-12);
        }
    }

    #[test]
    fn single_ride_matches_law_exactly() {
        let report = run(&tiny(true), &policy());
        // One ride node: m = 0, r = 1, overhead = 2, tc_R = tc_C + 1.
        assert_eq!(report["same_tree_real"]["m"], 0.0);
        assert_eq!(report["same_tree_real"]["r"], 1.0);
        assert_eq!(report["same_tree_real"]["predicted_overhead"], 2.0);
        let same_tc = report["same_tree_real"]["tc"].as_f64().unwrap();
        let complex_tc = report["complex_tree"]["tc"].as_f64().unwrap();
        assert!((same_tc - complex_tc - 1.0).abs() < 1e-12);
        // One complex leaf: the tree-free export appends a single size-2 leg
        // and needs no mul vertex; the tree-following export has no merge and
        // therefore no factor leaves, and its installed tree is the same
        // single node riding the green leg: tc matches the law exactly.
        assert_eq!(report["tree_free"]["mul_vertices"], 0);
        assert_eq!(report["tree_following"]["factor_leaves"], 0);
        let installed_tc = report["tree_following"]["installed"]["tc"].as_f64().unwrap();
        assert!((installed_tc - same_tc).abs() < 1e-12);
        // The output gained exactly one size-2 Re/Im axis.
        let d = &report["deltas_vs_same_tree"];
        assert!(d["tree_free"]["tc_deviation_pct"].as_f64().unwrap().is_finite());
    }
}

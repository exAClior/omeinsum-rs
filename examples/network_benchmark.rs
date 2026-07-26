//! Execute every physical slice of a v2 Yao benchmark artifact.
#[cfg(all(feature = "cuda", feature = "ascend"))]
compile_error!("select at most one accelerator feature");
#[path = "support/yao_benchmark.rs"]
mod format;
#[path = "support/benchmark_runner.rs"]
mod runner;
use format::BenchmarkNetwork;
use num_complex::{Complex32, Complex64};
#[cfg(all(feature = "cuda", not(feature = "ascend")))]
use omeinsum::backend::CudaComplex;
#[cfg(all(feature = "ascend", not(feature = "cuda")))]
use omeinsum::Ascend as Device;
#[cfg(not(any(feature = "cuda", feature = "ascend")))]
use omeinsum::Cpu as Device;
#[cfg(all(feature = "cuda", not(feature = "ascend")))]
use omeinsum::Cuda as Device;
use omeinsum::{Backend, Cpu};
use std::{fs::File, io::BufReader, time::Instant};

struct Args {
    path: String,
    representation: String,
    warmup: usize,
    repeats: usize,
    check: bool,
    dtype: String,
}
fn args() -> Args {
    let mut a = std::env::args().skip(1);
    let path = a
        .next()
        .unwrap_or_else(|| "benches/network_small.json".into());
    let mut x = Args {
        path,
        representation: "tree-real".into(),
        warmup: 3,
        repeats: 20,
        check: false,
        dtype: "f32".into(),
    };
    while let Some(f) = a.next() {
        match f.as_str() {
            "--check-cpu" => x.check = true,
            "--representation" => x.representation = a.next().expect("missing representation"),
            "--warmup" => x.warmup = a.next().unwrap().parse().unwrap(),
            "--repeats" => x.repeats = a.next().unwrap().parse().unwrap(),
            "--dtype" => x.dtype = a.next().expect("missing dtype"),
            _ => panic!("unknown option {f}"),
        }
    }
    assert!(x.repeats > 0);
    assert!(matches!(x.dtype.as_str(), "f32" | "c64"), "--dtype must be f32 or c64");
    x
}
#[cfg(any(feature = "cuda", feature = "ascend"))]
fn device() -> Device {
    Device::new().expect("failed to initialize device 0")
}
#[cfg(not(any(feature = "cuda", feature = "ascend")))]
fn device() -> Device {
    Cpu
}
fn cpu_reference(n: &BenchmarkNetwork) -> Complex32 {
    let mut unsliced = n.clone();
    unsliced.cuts.clear();
    unsliced.assignment_count = 1;
    let c = runner::native_cache(&unsliced, Cpu, Complex32::new);
    runner::solve_native(&unsliced, &c, &Cpu)
}

fn main() {
    let a = args();
    let mut d = serde_json::Deserializer::from_reader(BufReader::new(File::open(&a.path).unwrap()));
    d.disable_recursion_limit();
    let n: BenchmarkNetwork = serde::Deserialize::deserialize(&mut d).expect("invalid artifact");
    assert_eq!(n.format, "omeinsum-yao-benchmark-v2");
    n.validate().expect("artifact validation failed");
    let dev = device();
    // W4: c64 reference path (native-complex only; tree-real/static stay f32).
    if a.dtype == "c64" {
        run_c64(&a, &n, &dev);
        return;
    }
    let run: Box<dyn Fn() -> Complex32> = match a.representation.as_str() {
        "tree-real" | "dispatch" => {
            let cache = runner::real_cache(&n, dev.clone());
            dev.synchronize();
            let run_network = n.clone();
            let run_device = dev.clone();
            Box::new(move || runner::solve_real(&run_network, &cache, &run_device))
        }
        "static-factorized" => {
            let prepared = runner::static_factorized(&n, dev.clone());
            dev.synchronize();
            let run_network = n.clone();
            let run_device = dev.clone();
            Box::new(move || runner::solve_static(&run_network, &prepared, &run_device))
        }
        "static-dense" => {
            let prepared = runner::static_dense(&n, dev.clone());
            dev.synchronize();
            let run_network = n.clone();
            let run_device = dev.clone();
            Box::new(move || runner::solve_static(&run_network, &prepared, &run_device))
        }
        "native-complex" => native_run(&n, &dev),
        _ => panic!(
            "representation must be native-complex, tree-real/dispatch, static-factorized, or static-dense"
        ),
    };
    let initial = run();
    let expected = a.check.then(|| cpu_reference(&n));
    if let Some(e) = expected {
        let err = (initial - e).norm();
        assert!(
            err <= 1e-4 + 1e-4 * e.norm(),
            "CPU check failed: error={err}"
        );
    }
    for _ in 0..a.warmup {
        std::hint::black_box(run());
    }
    let mut samples = Vec::new();
    for _ in 0..a.repeats {
        let t = Instant::now();
        std::hint::black_box(run());
        samples.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let mut sorted = samples.clone();
    sorted.sort_by(f64::total_cmp);
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let median = if sorted.len() % 2 == 0 {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
    } else {
        sorted[sorted.len() / 2]
    };
    let audit = &n.tree_real_audit;
    let u = &n.complexity.physical_unsliced;
    let p = &n.complexity.physical_per_slice;
    let total = &n.complexity.physical_total;
    println!("backend={} network={} representation={} cuts={} assignments={} warmup={} repeats={} scope=synchronized-full-slice-solve",Device::name(),a.path,a.representation,n.cuts.iter().map(|x|x.to_string()).collect::<Vec<_>>().join(","),n.assignment_count,a.warmup,a.repeats);
    println!("complexity unsliced_tc={:.6} unsliced_sc={:.6} unsliced_rwc={:.6} per_slice_tc={:.6} per_slice_sc={:.6} per_slice_rwc={:.6} total_tc={:.6} total_sc={:.6} total_rwc={:.6}",u.log2_flops,u.log2_peak_elements,u.log2_readwrites,p.log2_flops,p.log2_peak_elements,p.log2_readwrites,total.log2_flops,total.log2_peak_elements,total.log2_readwrites);
    println!("tree_real pass={} ride={} merge={} pass_volume={} ride_volume={} merge_volume={} base_volume={} real_volume={} m={} r={} predicted_overhead={} native_contraction_sc_elements={} native_contraction_sc_bytes={} tree_real_logical_sc_elements={} tree_real_logical_sc_bytes={} native_resident_input_cache_elements={} native_resident_input_cache_bytes={} tree_real_resident_input_cache_elements={} tree_real_resident_input_cache_bytes={}",audit.pass_nodes,audit.ride_nodes,audit.merge_nodes,audit.pass_volume,audit.ride_volume,audit.merge_volume,audit.base_volume,audit.real_volume,audit.m,audit.r,audit.predicted_overhead,audit.native_contraction_sc_elements,audit.native_contraction_sc_bytes,audit.tree_real_logical_sc_elements,audit.tree_real_logical_sc_bytes,audit.native_resident_input_cache_elements,audit.native_resident_input_cache_bytes,audit.tree_real_resident_input_cache_elements,audit.tree_real_resident_input_cache_bytes);
    println!("result_values={:.12e},{:.12e}", initial.re, initial.im);
    println!(
        "wall_clock_ms mean={mean:.6} median={median:.6} min={:.6} max={:.6}",
        sorted[0],
        sorted[sorted.len() - 1]
    );
    println!(
        "samples_ms={}",
        samples
            .iter()
            .map(|x| format!("{x:.6}"))
            .collect::<Vec<_>>()
            .join(",")
    );
}

#[cfg(not(feature = "ascend"))]
fn native_run<'a>(n: &'a BenchmarkNetwork, d: &'a Device) -> Box<dyn Fn() -> Complex32 + 'a> {
    #[cfg(feature = "cuda")]
    let cache = runner::native_cache(n, d.clone(), CudaComplex::new);
    #[cfg(not(feature = "cuda"))]
    let cache = runner::native_cache(n, d.clone(), Complex32::new);
    Box::new(move || runner::solve_native(n, &cache, d))
}
#[cfg(feature = "ascend")]
fn native_run<'a>(_: &'a BenchmarkNetwork, _: &'a Device) -> Box<dyn Fn() -> Complex32 + 'a> {
    panic!("native-complex is unsupported on Ascend")
}

// W4: c64 reference execution (native-complex only). Prints result_values at
// full f64 precision; the f32 main path is untouched.
#[cfg(not(feature = "ascend"))]
fn run_c64(a: &Args, n: &BenchmarkNetwork, dev: &Device) {
    assert_eq!(
        a.representation, "native-complex",
        "--dtype c64 supports native-complex only"
    );
    #[cfg(feature = "cuda")]
    let cache = runner::native_cache_c64(n, dev.clone(), CudaComplex::<f64>::new);
    #[cfg(not(feature = "cuda"))]
    let cache = runner::native_cache_c64(n, dev.clone(), Complex64::new);
    dev.synchronize();
    let run = || runner::solve_native_c64(n, &cache, dev);
    let initial = run();
    if a.check {
        let mut unsliced = n.clone();
        unsliced.cuts.clear();
        unsliced.assignment_count = 1;
        let c = runner::native_cache_c64(&unsliced, Cpu, Complex64::new);
        let e = runner::solve_native_c64(&unsliced, &c, &Cpu);
        let err = (initial - e).norm();
        assert!(
            err <= 1e-12 + 1e-12 * e.norm(),
            "CPU c64 check failed: error={err}"
        );
    }
    for _ in 0..a.warmup {
        std::hint::black_box(run());
    }
    let mut samples = Vec::new();
    for _ in 0..a.repeats {
        let t = Instant::now();
        std::hint::black_box(run());
        samples.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let mut sorted = samples.clone();
    sorted.sort_by(f64::total_cmp);
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let median = if sorted.len() % 2 == 0 {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
    } else {
        sorted[sorted.len() / 2]
    };
    let audit = &n.tree_real_audit;
    let u = &n.complexity.physical_unsliced;
    let p = &n.complexity.physical_per_slice;
    let total = &n.complexity.physical_total;
    println!("backend={} network={} representation={} cuts={} assignments={} warmup={} repeats={} scope=synchronized-full-slice-solve",Device::name(),a.path,a.representation,n.cuts.iter().map(|x|x.to_string()).collect::<Vec<_>>().join(","),n.assignment_count,a.warmup,a.repeats);
    println!("complexity unsliced_tc={:.6} unsliced_sc={:.6} unsliced_rwc={:.6} per_slice_tc={:.6} per_slice_sc={:.6} per_slice_rwc={:.6} total_tc={:.6} total_sc={:.6} total_rwc={:.6}",u.log2_flops,u.log2_peak_elements,u.log2_readwrites,p.log2_flops,p.log2_peak_elements,p.log2_readwrites,total.log2_flops,total.log2_peak_elements,total.log2_readwrites);
    println!("tree_real pass={} ride={} merge={} pass_volume={} ride_volume={} merge_volume={} base_volume={} real_volume={} m={} r={} predicted_overhead={} native_contraction_sc_elements={} native_contraction_sc_bytes={} tree_real_logical_sc_elements={} tree_real_logical_sc_bytes={} native_resident_input_cache_elements={} native_resident_input_cache_bytes={} tree_real_resident_input_cache_elements={} tree_real_resident_input_cache_bytes={}",audit.pass_nodes,audit.ride_nodes,audit.merge_nodes,audit.pass_volume,audit.ride_volume,audit.merge_volume,audit.base_volume,audit.real_volume,audit.m,audit.r,audit.predicted_overhead,audit.native_contraction_sc_elements,audit.native_contraction_sc_bytes,audit.tree_real_logical_sc_elements,audit.tree_real_logical_sc_bytes,audit.native_resident_input_cache_elements,audit.native_resident_input_cache_bytes,audit.tree_real_resident_input_cache_elements,audit.tree_real_resident_input_cache_bytes);
    println!("result_values={:.17e},{:.17e}", initial.re, initial.im);
    println!(
        "wall_clock_ms mean={mean:.6} median={median:.6} min={:.6} max={:.6}",
        sorted[0],
        sorted[sorted.len() - 1]
    );
    println!(
        "samples_ms={}",
        samples
            .iter()
            .map(|x| format!("{x:.6}"))
            .collect::<Vec<_>>()
            .join(",")
    );
}

#[cfg(feature = "ascend")]
fn run_c64(_: &Args, _: &BenchmarkNetwork, _: &Device) {
    panic!("native-complex is unsupported on Ascend")
}

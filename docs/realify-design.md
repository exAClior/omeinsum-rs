# Realify: Complex → Real Tensor Network Conversion

**Status:** core implemented (M0–M2), with CPU and Ascend benchmark harnesses,
M4 CLI support, final-only complex recovery, and an M5 feature-gated Ascend test.
The 2026-07-22 Ascend study establishes correctness and the performance crossover.
**M6 implemented (2026-07-23), GPU/Ascend campaign complete (2026-07-24):**
tree-following realification with the rank-3-factorized multiplication vertex — the
ComplexTN.jl construction ported to Rust. The library transform, installed-tree
execution and AD, `--realify-tree` CLI, and four-mode benchmark harness are verified.
Matched A800/Ascend measurements show that `3 : 3 : 4` predicts core multiplication
arithmetic, not wall time: the `4/3` dense/factorized ratio appears on the
contraction-dominated Ascend case, but not on CUDA (§5 M6d).
**Goal:** contract complex-valued tensor networks on backends without native complex
support (today: Ascend, which is f32-only; also CUDA builds without cuTENSOR) by
mechanically rewriting the network into an equivalent real-valued network, with no
asymptotic FLOP or memory overhead versus native complex contraction.

This document is self-contained: the math is stated with verified constants, the
relevant codebase facts are listed, all design decisions are settled with rationale,
and the work is broken into milestones with test gates. An implementing session
should not need to re-derive anything.

---

## 1. Mathematical background (verified)

Source: *tnet.pdf* (Typst lecture notes), section "Complex Numbers: A Tensor-Network
Perspective", pp. 23–25. Every identity below was re-verified numerically with numpy
to machine precision before being written down here.

### 1.1 Realification of a tensor

ℂ is a 2-dimensional algebra over ℝ. Make that explicit as network structure: each
complex tensor `A` with indices `i1..in` becomes a real tensor `T_A` with **one extra
trailing index of dimension 2**:

```text
(T_A)[i1..in, 0] = Re(A[i1..in])
(T_A)[i1..in, 1] = Im(A[i1..in])
```

In column-major layout (what this crate uses) the extra index being *last* means the
data is simply `[ all Re entries (col-major) ; then all Im entries (col-major) ]`.

- **Conjugation** = applying `Z = diag(1, -1)` to the extra leg. Implementation-wise:
  negate the Im block during conversion. No extra tensor needed.
- **Phase multiplication** `e^{iφ}·A` = rotation `R_φ = [[cosφ, -sinφ],[sinφ, cosφ]]`
  on the extra leg (column-major data `[cosφ, sinφ, -sinφ, cosφ]`). Not needed for
  the core feature; export as a utility constant.

### 1.2 The multiplication vertex

Complex multiplication `(x0 + i·x1)(y0 + i·y1)` is a bilinear map ℝ²×ℝ² → ℝ², i.e. a
rank-3 tensor `M[a,b,c]` with `result[c] = Σ_ab x[a]·y[b]·M[a,b,c]`:

```text
M[:,:,0] = [[1, 0], [0, -1]]   # Re = x0·y0 − x1·y1
M[:,:,1] = [[0, 1], [1,  0]]   # Im = x0·y1 + x1·y0
```

Column-major flat data for shape `[2,2,2]`:

```rust
// index (a,b,c) at position a + 2b + 4c
const M_DATA: [f64; 8] = [1.0, 0.0, 0.0, -1.0,   0.0, 1.0, 1.0, 0.0];
```

The lecture notes use a variant `𝒞 = M·Z` (multiplication with conjugated output),
whose nonzeros are `𝒞_000 = 1`, `𝒞_011 = 𝒞_101 = 𝒞_110 = −1`:

```rust
const C_DATA: [f64; 8] = [1.0, 0.0, 0.0, -1.0,   0.0, -1.0, -1.0, 0.0];
```

`𝒞` is **invariant under any permutation of its three legs** and invariant under
`Z⊗Z⊗Z` (both verified). These symmetries make it the nicer object for *tests and
graphical identities*, but the plain `M` is the vertex we insert at runtime (one
fewer tensor; eq. 26 of the notes is `T_A, T_B, 𝒞, Z` chained — pre-contracting
`𝒞·Z = M` is exactly the same network with the trailing `Z` absorbed).

Matrix-product ground truth (notes eq. 26, rewritten with `M`):

```text
D = A·B   ⟺   (T_D)[i,k,d] = Σ_{j,a,b} (T_A)[i,j,a] · (T_B)[j,k,b] · M[a,b,d]
einsum: "ija,jkb,abd->ikd"
```

### 1.3 Whole-network recipe

Given a complex einsum `(ixs, iy, size_dict)` with input tensors `t_1..t_n`, of which
`m` are genuinely complex:

1. **Realify each complex tensor**: tensor `k` gets a fresh label `a_k` appended to
   its `ix` (dimension 2). Real tensors are left untouched — their extra leg would be
   pinned to `e0 = (1,0)`, and `M · e0 = I₂` (verified), so it cancels exactly.
2. **Insert `m − 1` copies of `M`** as ordinary constant input tensors, merging the
   extra labels pairwise: `M(a_1, a_2 → b_2), M(b_2, a_3 → b_3), …, M(b_{m-1}, a_m → b_m)`.
   Associativity/permutation symmetry (the notes' "cascade rule") guarantees any
   merge tree gives the same result; the contraction-order optimizer remains free to
   schedule the merges anywhere in the tree.
3. **Output**: `iy' = iy ++ [b_m]` (or `++ [a_1]` when `m == 1`). Slicing the result
   at the last axis gives Re (index 0) and Im (index 1). A scalar-output network
   becomes a shape-`[2]` vector `[Re, Im]`.
4. `size_dict' = size_dict ∪ { a_k ↦ 2, b_k ↦ 2 }`.

Special case `m == 0`: the network is already real; the transform is the identity and
the caller must know no trailing axis was added (see `RealifiedOutput` below).

A 3-tensor chain (`ija,jkb,abe,klc,ecf->ilf` ≡ `A·B·B'`) was verified numerically.

### 1.4 Cost

Algebraically, a binary contraction of two realified tensors followed by an
`M`-merge requires the same four real products as a native complex contraction
(Re·Re, Im·Im, Re·Im, Im·Re). Memory is 2× real (same as complex), so
realification has **no asymptotic overhead**. Whether the generic optimizer and
backend realize that ideal schedule is a benchmark question; the constant-factor
risks are:

- a separate rank-3 `M` leaf forces one temporary with two dim-2 legs (4× real data)
  under the engine's binary contraction trees. This is not merely a poor tree: the
  first contraction among `A`, `B`, and `M` leaves two Re/Im labels for every possible
  pair. Removing it requires a fused three-input lowering or specialized kernel, not
  a different binary tree. This is a **cost issue only, never a correctness issue**.
- Gauss's trick (3 real multiplications instead of 4) is a possible later fusion; not
  in scope for v1.

**2026-07-23 revision.** The 4× temporary above is inherent to the *dense* `M` under
binary trees — both pairwise orders through a 3-leg vertex leave two dim-2 legs on
an intermediate. It is **not** inherent to realification: the rank-3 factorization of
`M` (§1.6) expresses exactly the Gauss 3-multiplication schedule as ordinary binary
contractions, with no fused kernel and no three-input lowering. M6 adopts the
factorized vertex for the tree-following path; the dense vertex stays for the v1
cascade and for identity tests. This retires the "fused multiplication-vertex
lowering" and "Gauss 3-mult fusion" stretch bullets in M5 — the factorized static
form achieves both, in graph structure rather than in kernel code.

### 1.5 Exact integer test vector (paste into tests)

```text
A = [[1+2i, 3+0i], [0−1i, 2−1i]]        B = [[2+0i, 0+1i], [1−1i, 4+0i]]
D = A·B = [[5+1i, 10+1i], [1−5i, 9−4i]]

Column-major blocks (shape [2,2,2], Re block then Im block):
T_A data = [1, 0, 3, 2,   2, -1, 0, -1]
T_B data = [2, 1, 0, 4,   0, -1, 1, 0]
T_D data = [5, 1, 10, 9,  1, -5, 1, -4]

Network: ixs = [[0,1,10], [1,2,11], [10,11,12]]  (labels 10,11,12 are the extra legs)
         iy  = [0,2,12]
         tensors = [T_A, T_B, M]
         sizes: 0↦2, 1↦2, 2↦2, 10↦2, 11↦2, 12↦2
```

### 1.6 The factorized multiplication vertex (M6, verified against `M_DATA`)

`M` has tensor rank 3 (Winograd): exactly three rank-1 terms suffice, and two
cannot. The decomposition is Gauss's multiplication algorithm written as tensor
factors — `U`, `V` shape `[2,3]`, `W` shape `[2,3]`, column-major:

```rust
/// U = V = [[1,0,1],[0,1,1]]; the three linear forms x0, x1, x0+x1.
pub const U_DATA: [f64; 6] = [1.0, 0.0,   0.0, 1.0,   1.0, 1.0];
/// W = [[1,-1,0],[-1,-1,1]]; Re = p1 - p2, Im = p3 - p1 - p2.
pub const W_DATA: [f64; 6] = [1.0, -1.0,  -1.0, -1.0,  0.0, 1.0];
```

Identity (unit test, all 8 entries): `M[a,b,c] = Σ_k U[a,k] V[b,k] W[c,k]`, `k` a
fresh label of size 3 per merge site.

A tree node contracting green children `X[…,a]`, `Y[…,b]` over skeleton labels `S`
becomes a 4-step subtree of ordinary binary contractions:

1. `X′ ← contract(X, U)` over `a` — cheap, surface-size;
2. `Y′ ← contract(Y, V)` over `b` — cheap;
3. `Z″ ← contract(X′, Y′)` over `S`, with `k` a **batch** label (in both inputs and
   the output) — exactly 3× the skeleton contraction;
4. `Z ← contract(Z″, W)` over `k`, emitting the node's single dim-2 green leg `c`.

The 3× merge cost is **structural, not optimizer-dependent**: alternative pairwise
schedules of this subtree sum `k` too early and cost 6× the skeleton step, so any
working optimizer keeps the intended order. Total per merge: 3× skeleton +
O(surface). Numerically this is Gauss's algorithm — identical error behavior to the
`real_walk` merge arm and to 3M generally. Its linear forms can overflow or lose
cancellation for extreme finite operands where dense 4M remains finite; the
legacy dense path is the deliberate numerical-range fallback. This is the reason
to keep dense `M` beyond identity tests.

---

## 2. Codebase facts an implementer needs

Repo: `omeinsum` v0.1.1, single crate + `omeinsum-cli` workspace member, edition
2021, column-major tensors throughout. Canonical agent instructions:
`.claude/CLAUDE.md` (read it; `make check` is the pre-PR gate: fmt + clippy +
non-GPU tests).

Key types and where they live:

| Item | Location | Facts that matter here |
|---|---|---|
| `Tensor<T, B>` | `src/tensor/mod.rs` | column-major; `from_data(&[T], &[usize])`, `from_data_with_backend`, `to_vec()`, `permute`, `reshape`, `get(linear)`; `contract_binary::<A>` in `src/tensor/ops.rs` |
| `Scalar` | `src/algebra/mod.rs` | marker trait; impls include `f32, f64, Complex32, Complex64`; requires `bytemuck::Pod` |
| `Standard<T>` | `src/algebra/standard.rs` | the `(+,×)` algebra; realified execution runs `Standard<f64>`/`Standard<f32>` |
| `Einsum<L=usize>` | `src/einsum/engine.rs` | public fields `ixs: Vec<Vec<usize>>`, `iy: Vec<usize>`, `size_dict: HashMap<usize, usize>`; `optimize_greedy()`, `optimize_treesa()`, `set_contraction_tree`, `execute::<A,T,B>(&[&Tensor<T,B>])` |
| one-shot `einsum` | `src/einsum/mod.rs` | infers size_dict from tensors; we construct `Einsum` directly instead |
| `EinBuilder` | `src/einsum/builder.rs` | builder for `Einsum`; not required but keep API style consistent |
| `BackendScalar<B>` | `src/backend/traits.rs` | **Cpu: all scalars. Cuda: f32/f64 always, complex only with cuTENSOR (`cuda` feature). Ascend: `f32` only** ("Milestone A intentionally exposes only CANN's native f32 matmul path") |
| backward | `src/einsum/backward.rs` | gradient tape over the same execute path; realified networks get AD for free since they are ordinary real einsums |

Conventions (from `.claude/CLAUDE.md`):

- unit tests inline under `#[cfg(test)]`; integration suites in `tests/suites/*.rs`
  wired through `tests/main.rs` (do **not** add new top-level `tests/*.rs` crates);
- tests must prove **values**, not shapes;
- topology and tensor data are separate concerns — the design below honors this;
- `num-complex` already has the `bytemuck` feature: `Complex64` is Pod, `#[repr(C)]`
  `{re, im}`, so `bytemuck::cast_slice::<Complex64, f64>` yields interleaved
  `[re0, im0, re1, im1, …]` in column-major element order.

**Trap:** the doc comment on `Tensor::is_contiguous` (`src/tensor/mod.rs`) says
"contiguous in memory (row-major)" — that comment is wrong. The crate is
column-major everywhere: `compute_contiguous_strides` and the unit test
`test_from_data` (strides `[1, 2]` for shape `[2, 3]`) are the ground truth. Trust
the tests, not that comment.

Motivating consumer: `rydbergsim-rs/crates/rydberg-tn` calls omeinsum directly with
`Standard<Complex64>` on CPU (TDVP/GSE contractions; shapes mirrored in
`benches/complex_tdvp.rs`). `yao-rs` (`circuit_to_einsum`) is a second consumer.
Neither should need code changes to *use* the feature — it is a pure omeinsum API.

---

## 3. Design decisions (settled)

- **D1 — Code+data transform, not a new algebra.** Realification is a preprocessing
  step producing an ordinary real einsum. No changes to `Semiring`/`Algebra`, no
  backend-specific code. Every existing optimizer, backend, and the backward tape
  work unchanged. (Rejected alternative: a `RealifiedComplex` algebra type — touches
  every backend kernel, violates "topology and data are separate concerns".)
- **D2 — Insert `M`, export `𝒞`/`Z`/`R_φ` as constants.** `M = 𝒞·Z` is the runtime
  vertex (one tensor per merge). `𝒞`, `Z`, `R_φ`, `E0` are exported for tests and
  identity checks (permutation invariance, `Z⊗Z⊗Z·𝒞 = 𝒞`, `M·e0 = I`).
- **D3 — Extra leg is appended last.** Column-major ⇒ contiguous Re block then Im
  block; conversion from `Complex<T>` storage is a de-interleave; recovery is a
  slice at the last axis. Matches the notes' `cat(real(A), imag(A), dims=n+1)`.
- **D4 — Conjugation is a per-input flag**, applied by negating the Im block during
  conversion. No `Z` tensors in the network. This makes bra-side tensors in
  `⟨ψ|O|ψ⟩` sandwiches share data with ket-side tensors.
- **D5 — Only genuinely complex inputs get an extra leg.** Mixed real/complex
  networks are first-class: real tensors pass through untouched (their `M` would
  collapse to identity). The caller declares which inputs are complex via the input
  enum below; an `is_real` auto-detect (all Im ≈ 0) is deliberately **not** done in
  v1 (silent behavior changes from noisy zeros; revisit later as an explicit opt-in).
- **D6 — Planning is separated from data conversion.** The label/topology transform
  (`RealifyPlan`) is pure and backend-agnostic. Data conversion runs **host-side**
  on slices, *then* uploads via `Tensor::from_data_with_backend`. This is forced:
  `Tensor<Complex64, Ascend>` cannot exist (`BackendScalar` gate), so complex data
  can never touch a real-only backend. CPU-side conversion + real upload is the only
  possible dataflow, and the API should make it the natural one.
- **D7 — Generic over `Complex<T>` with `T ∈ {f32, f64}`.** c64 ↦ f64 network,
  c32 ↦ f32 network. Note for Ascend users: the backend is f32-only today, so c64
  networks targeting Ascend must be explicitly downcast by the caller (precision
  loss is their informed choice); document this, do not silently downcast.
- **D8 — Merge topology in v1 is a left-deep chain** of `M` vertices in input order,
  handed to the optimizer as ordinary inputs. Rationale: simplest correct thing;
  the optimizer can still schedule merges anywhere. M3 found two Re/Im legs in each
  measured binary intermediate; §1.4 proves tree alignment alone cannot reduce that
  count, so a future optimization must fuse multiplication-vertex lowering.
  **2026-07-23 revision:** the §1.4 limitation is specific to the *dense* vertex;
  see D10–D12 for the tree-following factorized path (M6). The cascade remains the
  CLI convenience form; it is saved in practice by re-optimization (the ComplexTN.jl
  benchmark suite measured ≤1% cost difference between convert-only and fully
  re-annealed orders — the optimization landscape is flat). The unsafe direction is
  the reverse one: wiring merges in a fixed order *before* optimization without a
  re-plan (ComplexTN.jl measured a 1555× blow-up on a 16-qubit circuit from
  time-ordered wiring). M6 never pre-wires: it attaches to an already-optimized tree.
- **D9 — Output is explicit about whether a trailing Re/Im axis exists** (see
  `RealifiedOutput`), so `m == 0` (all-real network) is not a silent special case.
- **D10 — Tree-following realification (M6) mirrors an archived contraction tree.**
  Given a `NestedEinsum` already optimized on the complex topology (omeco TreeSA),
  each internal node maps to one of three static patterns: *pass* (neither child
  green: unchanged binary step), *ride* (one child green: the dim-2 leg rides to the
  node's output), *merge* (both green: the 4-step factorized subtree of §1.6).
  Invariant: every intermediate outside the merge subtrees carries at most one
  dim-2 green leg, so its logical payload is at most 2× the real skeleton. The local
  merge subtree also carries a size-3 rank leg. This is the ComplexTN.jl
  `_nested_append` construction, with the dense `𝒞` replaced by the rank-3
  factorization so literal execution hits 3× instead of 4×.
- **D11 — The tree-realified network is executed in the archived order, never
  re-optimized.** The output plan carries its realified `NestedEinsum` and installs
  it via `set_contraction_tree`. Re-optimization is pointless (flat landscape,
  ≤1%) and would only risk disturbing the topology-protected 3× schedule (§1.6).
- **D12 — Factorized vertex for the tree path, dense vertex for the cascade path.**
  Both are exact; they differ in execution cost under binary engines (3× vs 4× per
  merge) and in numerics (Gauss vs 4M error constants). The `real_walk` dispatch
  executor stays as-is: its merge arm *is* the factorized schedule in host code, so
  M6 adds the graph form of the same arithmetic — the value is AD-for-free through
  `backward.rs` (the walker is outside the gradient tape) and a static artifact for
  graph-compiler ingestion, not a flop count change.

---

## 4. Proposed API

New module `src/realify.rs` (single file; split only if it grows past ~600 lines),
re-exported from `lib.rs` as `pub mod realify` plus top-level re-exports of the main
entry points.

```rust
//! src/realify.rs — complex → real tensor network conversion.

use crate::{Backend, BackendScalar, Einsum, Scalar, Tensor};
use num_complex::Complex;

/// Constant tensors (all shape [2,2,2] except Z/E0), column-major data.
pub mod constants {
    /// Multiplication vertex M: result[c] = Σ_ab x[a] y[b] M[a,b,c].
    pub const M_DATA: [f64; 8] = [1.0, 0.0, 0.0, -1.0, 0.0, 1.0, 1.0, 0.0];
    /// Fully symmetric conjugated-output variant 𝒞 = M·Z (tests/identities).
    pub const C_DATA: [f64; 8] = [1.0, 0.0, 0.0, -1.0, 0.0, -1.0, -1.0, 0.0];
    /// Conjugation Z = diag(1, -1).
    pub const Z_DATA: [f64; 4] = [1.0, 0.0, 0.0, -1.0];
    /// Real-unit vector e0; pins the extra leg of a real tensor.
    pub const E0_DATA: [f64; 2] = [1.0, 0.0];
    // plus `pub fn r_phi(phi: f64) -> [f64; 4]` for phase rotations
}

/// Declares one input of the complex network, host-side.
/// `T` is the *real* scalar type of the realified network (f32 or f64).
pub enum RealifyInput<'a, T> {
    /// Already-real tensor: passes through unchanged (no extra leg).
    Real { data: &'a [T], shape: &'a [usize] },
    /// Complex tensor, column-major `Complex<T>` data.
    Complex { data: &'a [Complex<T>], shape: &'a [usize], conjugate: bool },
}

/// Result of the topology transform. Pure labels/sizes — no tensor data.
pub struct RealifyPlan {
    /// The realified einsum spec: original ixs with extra labels appended to
    /// complex inputs, followed by the ixs of the inserted M vertices;
    /// iy' = iy ++ [result_leg] when `output` is `ReImAxis`.
    pub einsum: Einsum<usize>,          // not yet optimized
    /// How many M vertices were appended (== max(m−1, 0)).
    pub num_mul_vertices: usize,
    /// Positions (into the *new* tensor list) of the M vertices.
    pub mul_vertex_positions: Vec<usize>,
    pub output: RealifiedOutput,
}

pub enum RealifiedOutput {
    /// No complex inputs: result is the plain real result, no trailing axis.
    Real,
    /// Result carries a trailing dim-2 axis; index 0 = Re, 1 = Im.
    ReImAxis,
}

/// Pure topology transform. `is_complex[k]` ⇔ input k gets an extra leg.
/// Fresh labels start at max(all labels in ixs/iy/size_dict) + 1.
pub fn realify_code(
    ixs: &[Vec<usize>],
    iy: &[usize],
    size_dict: &std::collections::HashMap<usize, usize>,
    is_complex: &[bool],
) -> RealifyPlan;

/// Host-side data conversion for one complex tensor:
/// interleaved Complex<T> (column-major) → [Re block ; ±Im block], shape ++ [2].
pub fn realify_data<T: Scalar + num_traits::Float>(
    data: &[Complex<T>],
    conjugate: bool,
) -> Vec<T>;

/// One-shot convenience on a chosen backend: plans, converts, uploads,
/// optimizes (greedy), executes with Standard<T>, returns the raw real result
/// plus the output descriptor.
pub fn realify_einsum<T, B>(
    inputs: &[RealifyInput<'_, T>],
    ixs: &[Vec<usize>],
    iy: &[usize],
    backend: B,
) -> (Tensor<T, B>, RealifiedOutput)
where
    T: Scalar + num_traits::Float,
    B: Backend,
    T: BackendScalar<B>;

/// Recover a complex result host-side (CPU use / tests).
/// Splits the trailing axis: returns (re, im) with the original output shape.
pub fn split_re_im<T: Scalar, B: Backend>(
    result: &Tensor<T, B>,
) -> (Vec<T>, Vec<T>);

/// Final-only recovery with one complex output allocation.
pub fn recover_complex<T: Scalar, B: Backend>(
    result: &Tensor<T, B>,
) -> Vec<Complex<T>>;
```

Notes:

- `realify_code` must **not** consume tensor data — keeps it testable in isolation
  and usable by callers (rydberg-tn) that manage their own device tensors.
- `realify_einsum` is deliberately thin sugar over plan + convert + `Einsum::execute`.
  Callers with custom optimization (TreeSA, precomputed trees) use the pieces.
- `realify_data` is a de-interleave loop; use `bytemuck::cast_slice` only as an
  input view (`&[Complex<T>] → &[T]` interleaved), then copy out blocks. Conjugate ⇒
  negate while writing the Im block. No in-place trickery.
- The M vertex tensor for `T = f32` is `M_DATA.map(|x| x as f32)`; keep a small
  helper `mul_vertex_tensor::<T, B>(backend) -> Tensor<T, B>`.

### 4.1 Tree-following API (M6, planned)

```rust
/// Result of the tree-following topology transform. Pure structure — no tensor data.
pub struct RealifyTreePlan {
    /// Flat spec of the realified network (original ixs with green labels appended
    /// to complex inputs, plus U/V/W factor leaves), with the realified
    /// `NestedEinsum` already installed via `set_contraction_tree`.
    pub einsum: Einsum<usize>,
    /// Positions of the inserted factor-tensor leaves, per merge site: (u, v, w).
    pub factor_vertex_positions: Vec<(usize, usize, usize)>,
    pub output: RealifiedOutput,
}

/// Pure tree-following transform (D10). `tree` is the archived optimized
/// contraction tree of the *complex* network; `is_complex[k]` ⇔ input k gets a
/// green leg. Fresh labels (green, size 2; rank, size 3) start at
/// max(all labels) + 1 and must not collide with cut labels in sliced artifacts.
pub fn realify_tree_code(
    tree: &NestedEinsum<usize>,
    ixs: &[Vec<usize>],
    iy: &[usize],
    size_dict: &std::collections::HashMap<usize, usize>,
    is_complex: &[bool],
) -> RealifyTreePlan;

/// The U (= V) and W factor tensors on a backend, mirroring `mul_vertex_tensor`.
pub fn merge_factor_tensors<T, B>(backend: B) -> (Tensor<T, B>, Tensor<T, B>);
```

- Data conversion reuses M1's `realify_data` unchanged (leaf conversion,
  `conjugate` flag); recovery reuses `split_re_im` / `recover_complex`.
- Execution is the stock engine — `plan.einsum.execute::<Standard<T>, _, _>(…)` —
  which walks the installed tree; no new executor, no dispatch code.
- AD is the stock tape: `einsum_with_grad` / `cost_and_gradient` on the plan's flat
  spec. If the tape re-plans internally, gradients are unaffected (order-invariant
  values); assert values, and separately assert the forward schedule follows the
  installed tree on the direct `execute` path.

## 5. Milestones

Each milestone ends green under `make check`. Wire the new integration suite into
`tests/main.rs` as `#[path = "suites/realify.rs"] mod realify;` (M2); the M6 suite
is `suites/realify_tree.rs`.

### M0 — constants + topology transform (no tensor data)

Files: `src/realify.rs` (new), `src/lib.rs` (module + re-exports).

- Implement `constants`, `realify_code`, `RealifyPlan`, `RealifiedOutput`.
- Fresh-label allocation: `max(labels in ixs ∪ iy ∪ size_dict keys) + 1`. Extra
  labels and merge labels all get size 2 in `size_dict`.
- Handle: `m == 0` (identity plan, `RealifiedOutput::Real`), `m == 1` (no M vertex,
  `iy' = iy ++ [a_1]`), `m ≥ 2` (chain of M vertices per §1.3).
- Repeated labels inside one `ix` (diagonals, e.g. `[0,0]`) are untouched — the
  extra label is simply appended; add a unit test asserting the plan for `[[0,0]]`.
- Unit tests (inline): plan shapes for m ∈ {0,1,2,3}; label freshness (collision
  with a sparse size_dict containing a high unused label); `C_DATA` permutation
  invariance and `Z⊗Z⊗Z·𝒞 = 𝒞` checked numerically against `constants`; `M·e0 = I₂`.

### M1 — data conversion + recovery

Files: `src/realify.rs`.

- `realify_data` (+ conjugate), `split_re_im`, `mul_vertex_tensor`.
- Unit tests with exact integers: the §1.5 vectors for `T_A`/`T_B` (including a
  `conjugate: true` case: `T_A*` data = `[1, 0, 3, 2, -2, 1, 0, 1]`); round-trip
  `Complex → realify_data → split` equality.

### M2 — end-to-end execution + oracle suite

Files: `src/realify.rs` (`realify_einsum`), `tests/suites/realify.rs` (new),
`tests/main.rs` (wire-in).

Oracle: the same network contracted natively with `Standard<Complex64>` on `Cpu`.
Tests prove **values** (`approx::assert_relative_eq`, tol 1e-10 for f64):

1. §1.5 matmul with exact integer asserts (no oracle needed).
2. Random 4–6 tensor chains/networks, f64, mixed real/complex inputs, including a
   permuted / partially traced output `iy`.
3. TDVP-shaped contraction copied from `benches/complex_tdvp.rs` (χ = 32 slice).
4. Sandwich `⟨ψ|O|ψ⟩`: bra = same data with `conjugate: true`; assert Im ≈ 0 for
   Hermitian `O` **and** the value matches the oracle (catches sign errors that
   Im-only checks miss).
5. `m == 0`: all-real network returns `RealifiedOutput::Real`, bit-identical to a
   plain real einsum.
6. `m == 1`: single complex tensor, unary-style network (e.g. partial trace).
7. Scalar output network → shape `[2]` `[Re, Im]`.
8. Both unoptimized (`execute_pairwise`) and `optimize_greedy` paths, and once via
   `optimize_treesa` on the 6-tensor network (guards tree-shape assumptions).
9. Backward smoke test: `einsum_with_grad` on a realified scalar network runs and
   gradients match finite differences on 2–3 entries (AD-for-free claim, verified).
   Note `einsum_with_grad` re-infers `size_dict` from tensors and greedy-optimizes
   internally — fine here, since every size (including the dim-2 legs) is inferable
   from the realified tensors; pass the plan's `ixs`/`iy` slices directly.

### M3 — benchmark + docs (implemented and measured)

Files: `benches/realify.rs` (new, criterion, mirror `complex_tdvp.rs` cases),
`Cargo.toml` (`[[bench]]`), this document (update status), `docs/src` book page if
the mdbook lists features.

- `benches/realify.rs` compares native complex CPU and realified CPU. The dedicated
  `realify_ascend_study` example additionally separates preloaded execution from
  end-to-end host conversion, H2D, contraction, one D2H, and `recover_complex`.
- HPC4 job `109227` ran all four TDVP shapes at χ = 32, 64, 128, 256 with five
  warmups and 50 NPU repetitions. Every result had a genuinely nonzero imaginary
  component; maximum absolute error versus native c32 CPU was `1.133e-6` (maximum
  pointwise relative error `1.611e-3`, dominated by near-zero reference entries).
- Speedup ranges over the four shapes, against native c32 CPU on the same node:

  | χ | preloaded Ascend | end-to-end Ascend |
  |---:|---:|---:|
  | 32 | 0.8–1.6× | 0.5–0.9× |
  | 64 | 5.0–10.8× | 2.3–3.7× |
  | 128 | 26.1–76.8× | 7.7–11.6× |
  | 256 | 163.5–424.6× | 16.0–30.3× |

  Thus aggregate conversion, allocation, transfer, and recovery overhead sets the
  crossover between χ=32 and χ=64; this study does not attribute that delta to one
  component. CPU comparisons use ten timed repetitions per cell.
- All 16 optimized trees reported a maximum of two simultaneous Re/Im legs. Per
  §1.4, a binary-tree rewrite cannot make this one. The next kernel optimization is
  fused multiplication-vertex lowering; final recovery was separately reduced from
  two split-output allocations to one additional complex allocation via
  `recover_complex`.
- `runscribe` was unavailable locally and remotely. The session therefore preserves
  exact commands, raw CSV, NPU metadata, stderr, scheduler state, and hashes under
  `.hpc4-domestic/sessions/20260722T133107Z-realify-ascend-performance/`. Binary
  SHA-256: `6d86bdeef8dee9193ebae6b71b6f5159327db0ce83a7417429f0b37dbd26b3ce`.

### M4 — CLI support (implemented)

Files: `omeinsum-cli/src/contract.rs`, `autodiff.rs`, `format.rs`.

- `--realify` is valid for c32/c64 `contract` and `autodiff`: it plans, converts,
  executes a real network, then reassembles complex result and gradient JSON.
- The CLI remains CPU-only; the flag exercises the backend-independent transform
  but does not yet select Ascend or CUDA. A source topology/expression is validated,
  then the transformed network is greedily replanned because inserted multiplication
  vertices add leaves that do not exist in the source contraction tree.
- For autodiff, a complex seed `u+iv` differentiates the real scalar objective
  `u*Re(y) + v*Im(y)`. Omitting the seed for an original scalar output uses `[1,0]`,
  so gradients are for `Re(y)`. Gradients for internal multiplication vertices are
  discarded before writing one complex gradient per source input.

### M6 — tree-following realification with factorized vertex (implemented, 2026-07-23)

Port of the ComplexTN.jl construction (`real_convert` / `_nested_append`), upgraded
with the rank-3 factorized vertex (§1.6, D10–D12). Motivation, in order: (1) the
`real_walk` dispatch executor sits outside the `backward.rs` gradient tape, so the
dispatch path has no autodiff — the static graph differentiates for free;
(2) a static real einsum is the artifact static-graph accelerator compilers ingest;
(3) it enables the decisive three-way hardware comparison — dispatch vs
static-factorized vs static-dense — whose arithmetic prediction is 3 : 3 : 4.

#### M6a — factorized constants + pure tree transform (implemented)

Files: `src/realify.rs` (or `src/realify/tree.rs` if the module splits).

- `constants::{U_DATA, V_DATA, W_DATA}`; unit test the identity
  `M[a,b,c] = Σ_k U[a,k] V[b,k] W[c,k]` against `M_DATA` (all 8 entries, exact).
- `realify_tree_code` on `NestedEinsum<usize>`: pass/ride/merge node mapping (D10);
  fresh-label allocation reusing `allocate_fresh_labels` (green size 2, rank size 3);
  `m == 0` identity plan, `m == 1` single leg, no factor leaves.
- Unit tests (inline): plan shapes on hand-built trees covering all three node
  types; the one-green-leg invariant walked over every internal node; label
  freshness against a sparse `size_dict` with high labels and against archived cut
  labels; `m ∈ {0, 1}` cases.

#### M6b — end-to-end execution + oracle suite (implemented)

Files: `src/realify.rs`, `tests/suites/realify_tree.rs` (new), `tests/main.rs` (wire-in).

Oracle: native `Standard<Complex64>` on `Cpu`; cross-check against the cascade
`realify_einsum` and, in the examples harness, against `real_walk`. Tests prove
values (f64, tol 1e-10): mixed real/complex networks with pass/ride/merge nodes all
present; a `conjugate: true` sandwich; permuted `iy`; scalar output → shape `[2]`;
merges at shallow and deep tree positions; one sliced network with cuts on skeleton
labels only (green/rank labels are never cut).

#### M6c — AD parity (implemented)

- `cost_and_gradient` on a tree-realified scalar network uses the installed tree;
  gradients match finite differences on source entries and the conjugate of
  `backward.rs`'s holomorphic complex derivative, which is the complex representation
  of the real objective gradient. (`einsum_with_grad` deliberately remains a one-shot
  API that constructs and greedily plans a new `Einsum`, so it is not the schedule
  preservation test.)
- Forward `execute` and `cost_and_gradient` both consume the installed tree; structural
  tests prove every merge is the intended four-step subtree.

#### M6d — CLI + three-way benchmark + docs (GPU/Ascend campaign complete)

Files: `omeinsum-cli/src/{contract,autodiff}.rs`, `examples/network_benchmark.rs` +
`examples/support/`, this document (status), NPUBenchmarkData manifest (separate repo change).

- `--realify-tree` for `contract` and `autodiff` consumes the archived topology tree
  or an explicitly parenthesized expression tree (contrast `--realify`, which
  cascades and replans greedily). Implemented.
- The network benchmark example now has *dispatch* (`tree-real` alias),
  *static-factorized*, *static-dense*, and *native-complex* modes. Static-factorized
  uses the public M6 transform; static-dense is a benchmark-local tree transform so
  the diagnostic 4× path does not become public API. Sliced CPU parity tests pass.
- Matched CUDA/Ascend f32 runs used three immutable NPUBenchmarkData artifacts, their
  exact archived trees, 3 warmups, and 10 synchronized full-solve repetitions. Every
  timed configuration passed its CPU-native check. Durable samples and provenance are
  in NPUBenchmarkData `results/m6-static-realification-v1/` at benchmark execution
  commit `b604235ec86b24a0219ace7c469dd3a45b639469`; the tested omeinsum revision is
  `61cd6b9adc6d7ae42aafda67c4979cd54233b403`.
- Median dense/static-factorized ratios were CUDA `0.708`, `0.914` (bimodal;
  inconclusive), `0.780`, and Ascend `0.812`, `0.787`, `1.367` for `test`,
  rectangular-4x4-d16, and bristlecone-48-d16 respectively. Thus the arithmetic
  `4/3` appears only in the contraction-dominated Ascend point: factorization cut
  its dense median from `244.803 ms` to `179.036 ms` (26.9%). CUDA's robust points
  favored dense static execution despite its fourth product.
- Dispatch and static factorization are not wall-time-equivalent. On the substantive
  CUDA cases dispatch beat static factorization; on Ascend static factorization beat
  the current dispatch path. The latter alone required rank-greater-than-8 output
  materialization through the host, so it is correctness evidence and current-backend
  performance, not a lower bound for an all-device implementation. Three artifacts
  do not establish a universal crossover threshold; profiling and another large
  Ascend point are still warranted. A separate CPU performance sweep was not part of
  this hardware request.

### M5 — stretch (each independent, do only when justified)

- **Fused multiplication-vertex lowering**: recognize the `A,B,M` realification
  motif and emit a fused operation that never materializes its two-leg temporary.
  A custom binary tree cannot provide this guarantee (§1.4). *Superseded for new
  work by M6: the factorized vertex (§1.6) reaches 3× in graph structure with no
  kernel work; this bullet would now only benefit the legacy cascade path.*
- **Gauss 3-mult fusion** at binary-contract level (needs kernel-side work; only if
  profiling shows the 4th GEMM matters). *Superseded likewise — §1.6 is the Gauss
  schedule as tensor factors.*
- **`R_φ` / phase utilities** and an explicit-opt-in `detect_real` helper.
- **Ascend integration test** behind the `ascend` feature flag, c32 → f32 network,
  guarded like the CUDA suites. Implemented in `tests/suites/ascend.rs`; it requires
  a CANN-enabled host with an allocated NPU to link and execute.

## 6. Correctness invariants (checklist for review)

- [x] Contracting the realified network with `Standard<T>` equals the native
      `Standard<Complex<T>>` result, Re and Im, on every test network.
- [x] Real inputs are byte-identical pass-throughs (no copy beyond what the engine
      does anyway; no extra leg).
- [x] `conjugate: true` equals conjugating the data first (test both paths once).
- [x] Labels: no collisions; every extra/merge label has size 2 in `size_dict`;
      merge labels appear exactly twice (or once in `iy'` for the last one).
- [x] `m == 0` adds no axis; `m == 1` adds axis but no M vertex.
- [x] Result recovery preserves the engine's output ordering (`iy'` order — the
      trailing extra label stays last through `finalize_ordered_result`; assert on
      values after permuted-`iy` tests, which is where this would break).
- [x] No `unsafe`, no backend-specific code paths in `realify.rs`.

M6 additions (planned):

- [x] `U,V,W → M` identity holds exactly (all 8 entries).
- [x] Every intermediate outside merge subtrees carries ≤ 1 green leg; every merge
      subtree is the §1.6 4-step pattern with its own fresh size-3 rank label.
- [x] Forward execution follows the installed archived tree (no replanning).
- [x] Values match the native complex oracle, the cascade path, and benchmark
      `real_walk`; real-objective gradients match finite differences and the
      conjugated native holomorphic derivative.
- [x] Green and rank labels never collide with archived cut labels; slicing cuts
      only skeleton labels.

## 7. Verification commands

```bash
make check                      # canonical gate: fmt + clippy + non-GPU tests
cargo test --test main realify  # integration suite only
cargo test realify              # + inline unit tests
cargo bench --bench realify     # M3

# M6
cargo test --test main realify_tree          # integration suite
cargo test 'realify::tree' --lib             # inline unit tests
cargo test -p omeinsum-cli realify_tree      # CLI flag cases
cargo test --example network_benchmark       # static/dispatch sliced parity
```

Implementation verification on 2026-07-22:

- `cargo test realify` passed.
- `cargo test --test main realify` passed.
- `cargo check --benches` passed.
- `cargo test -p omeinsum-cli realify` passed (nine M4 contract/autodiff and
  topology-validation cases).
- `cargo check --features ascend --tests` passed locally.
- The feature-gated c32 → f32 Ascend integration test passed on two allocated
  Ascend 910 NPUs in HPC4 Slurm job `109214` (exit `0:0`, empty stderr).
- The full correctness/performance matrix passed in HPC4 job `109227` (exit `0:0`,
  empty application and scheduler stderr, success marker and matching binary hash).
- `make check` passed.

M6 implementation verification on 2026-07-23:

- `cargo test 'realify::tree' --lib` passed (8 transform/identity/validation tests).
- `cargo test --test main realify_tree` passed (5 native/cascade/value/AD/slicing tests).
- `cargo test -p omeinsum-cli realify_tree` passed (6 CLI cases).
- `cargo test --example network_benchmark` passed (6 tests, including sliced parity
  for dispatch, static-factorized, static-dense, and native execution).
- `make check` passed (format, clippy with tropical/parallel, 550 tests plus docs;
  11 known ignored unit tests and 4 ignored doctests).
- Matched final hardware runs completed on A800 (`m6-final-gpu-20260724`) and Ascend
  910 (HPC4 Slurm job `109459`, exit `0:0`) from clean, pinned revisions. All 21 timed
  configurations passed CPU checks. `isPANN/runscribe` was unavailable (upstream URLs
  returned HTTP 404), so the predeclared NPUBenchmarkData manifest/provenance fallback
  captured commits, input hashes, device inventories, commands, logs, and samples.
- The Ascend campaign exposed and verified the rank-greater-than-8 output-permutation
  fallback in `src/backend/ascend/contract.rs`; the final fail-fast run completed all
  dispatch and static modes. See M6d for the scoped performance conclusion.

## 8. References

- *tnet.pdf* pp. 23–25 ("Complex Numbers: A Tensor-Network Perspective"): eq. 23
  (realification), eqs. 24–25 (`𝒞`), eq. 26 (matmul network), graphical properties
  (permutation invariance, conjugation invariance, cascade rule), eq. 27 (backward
  rule `T_B = (T_A ∗ T_{D*})∗` — the Wirtinger conjugations emerge from the real
  network; motivates M2 test 9).
- The notes' own Julia/OMEinsum verification snippet (p. 24) uses
  `ein"ija,jkb,abc,cd->ikd"(T_A, T_B, C, Z)` — equivalent to our `M` form.
- Consumer shapes: `benches/complex_tdvp.rs` (this repo);
  `rydbergsim-rs/crates/rydberg-tn/src/{contraction,gse}.rs` (CPU complex today);
  `yao-rs/src/einsum.rs` (`circuit_to_einsum`, `ArrayD<Complex64>`).
- ComplexTN.jl (M6 source of the construction): `src/real_convert.jl`
  (`_nested_append`, dense `𝒞` variant), `articles/2026-07-23-realified-tn/main.typ`
  (cost law 1+2m+r; flat-landscape result; 1555× fixed-wiring blow-up; Winograd
  rank-3 bound), `benchmarks/paper/{greensa,wallclock}.jl` (the green-aware SA and
  the three-executor race — the instruments behind D8's revision and D11).
- NPUBenchmarkData `experiments/20260724-m6-static-realification-v1.yaml` and
  `results/m6-static-realification-v1/`: immutable M6 campaign definition, complete
  samples, correctness comparisons, scoped conclusions, and provenance. The archived
  artifact format originates in `experiments/20260723-all-circuits-overlap-v1.yaml`.

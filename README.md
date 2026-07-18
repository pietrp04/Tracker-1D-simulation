# 1D Tracker — Reconstruction, Calibration and CPU/GPU Benchmarking

A simulation and reconstruction chain for a **one-dimensional pixel tracker**, developed for the *Digital Data Acquisition Techniques* course (University of Pavia). Charged particles are emitted from a point-like source, cross `N` equally spaced detector layers, and are reconstructed as straight tracks by a combinatorial χ² track finder implemented **both on CPU (OpenMP) and GPU (CUDA)**. The project covers the full pipeline: event generation, a custom binary event format, track finding, clone removal, data-driven source calibration, performance/accuracy evaluation, and ROOT-based visualisation.

---

## Table of contents

- [Physics and geometry](#physics-and-geometry)
- [How the pipeline works](#how-the-pipeline-works)
- [Repository layout](#repository-layout)
- [Build](#build)
- [Executables](#executables)
- [Configuration](#configuration)
- [Key algorithms in detail](#key-algorithms-in-detail)
- [Known caveats](#known-caveats)
- [Next steps](#next-steps)

---

## Physics and geometry

The detector is a stack of `N_layers` one-dimensional layers, each a row of boolean pixels (a pixel is either hit or not). A point-like source sits at `x = 0`, `y = sourceY`, and emits particles whose emission angle `θ` is drawn from a chosen distribution. Each particle travels in a straight line

```
y(x) = sourceY + tan(θ) · x
```

and hits every layer it crosses within the acceptance `0 < y < 1`. A track is described by two parameters: slope `m = tan(θ)` and intercept `q = y(x=0)`. Reconstructing the source position means finding the peak of the reconstructed-`q` distribution (which should sit at `sourceY`).

Default geometry (from `config_new.h`):

| Constant | Value | Meaning |
|---|---|---|
| `N_layers` | 3 | number of layers |
| `l_tracker` | 1 m | layer length (acceptance `[0,1]`) |
| `l_pixel` | 0.0025 m | pixel pitch → 400 pixels/layer |
| `d_s_l` | 1 m | source-to-first-layer distance |
| `dist_interlayers` | 0.3 m | inter-layer spacing → layers at x = 1.0, 1.3, 1.6 m |
| `sourceY` | 0.5 m | true source height |
| `N_tracks` | 100 | particles generated per event |
| `chi2_threshold` | 5 | χ² cut for accepting a track |
| `[q_min, q_max]` | `[sourceY−0.1, sourceY+0.1]` | seed acceptance window |
| `sigma` | `l_pixel/√12` | single-hit resolution |

**Single-hit resolution.** A hit is quantised to the centre of the fired pixel; the true position within the pixel is uniform over one pitch, so `σ_hit = l_pixel/√12 ≈ 0.72 mm`.

**Propagated resolution.** Both `m` and `q` are fixed-coefficient linear combinations of the (quantised) hit positions, so their theoretical uncertainties follow from least-squares error propagation and depend only on the layer geometry:

```
σ_m = σ_hit · √(N / Δ)          σ_q = σ_hit · √(Sxx / Δ)
Sx = Σ x_L,  Sxx = Σ x_L²,  Δ = N·Sxx − Sx²
```

For the default geometry this gives `σ_q ≈ 2.25 mm`, a strong `m`–`q` anticorrelation (`ρ ≈ −0.98`), and drives both the clone-removal tolerances and the MC-matching ellipse. See `truth.cxx`.

---

## How the pipeline works

```
 Source ──► theoretical y per layer ──► acceptance cut (0<y<1)
        ──► pixelisation (Detector) ──► per-pixel Bernoulli noise
        ──► (optional) binary Event Format on disk (Storage)
        ──► raw track finding (CPU or GPU): seed + nearest-middle + χ² fit
        ──► clone removal (RimuoviCloni)
        ──► data-driven calibration: q-histogram ──► robust peak estimate (μ_q, σ_q)
        ──► calibrated selection (FiltraTraccePerQ, μ_q ± 3σ_q)
        ──► evaluation vs MC truth (Mahalanobis ellipse) : efficiency, purity, ghosts
```

Each stage is a separate, reusable module; the three executables assemble these modules for different studies.

---

## Repository layout

**Core library**

| File | Contents |
|---|---|
| `config_new.h` | all global constants + the `USE_GAUSSIAN_ANGLES` switch |
| `source.{h,cxx}` | `Source`: angle generation (uniform/gaussian dispatcher), theoretical trajectories, acceptance, per-pixel noise |
| `detector.{h,cxx}` | `Detector`: position ↔ pixel conversion (`pos2pix`/`pix2pos`) |
| `storage.{h,cxx}` | `Storage`: binary Event Format I/O (writing/reading/packing) |
| `track_finder.h` | shared declarations: `Track`, `QHistogram`, `PeakEstimate`, the finders and the histogram engines |
| `track_finder_cpu.cxx` | raw CPU track finding, `RimuoviCloni`, `FiltraTraccePerQ`, `CpuCleanedQHistogramEngine` |
| `track_finder_gpu.cu` | CUDA kernel + `GpuSingleEventFinder` (persistent-memory GPU finder) |
| `peak_estimate.cxx` | `RobustPeakEstimate`: fit-free source-peak estimator |
| `truth.{h,cxx}` | `CalcolaTolleranzeTeoriche` (theoretical resolutions) and `ValutaTracce` (MC matching) |
| `visualizer.{h,cxx}` | `Visualizer`: all ROOT plotting (calibration, noise scan, accuracy, timing) |

**Executables**

| File | Produces |
|---|---|
| `analisi_sorgente.cxx` | source calibration + efficiency/false-positives vs noise scan |
| `studio_prestazioni.cxx` | accuracy (efficiency & purity) vs N tracks, and CPU-vs-GPU timing vs N tracks |
| `visualizza_apparato.cxx` | apparatus display of one event through the four filtering stages, plus (m,q) phase-space plots; supports **multiple sources** |

**Other**

| File | |
|---|---|
| `Makefile` | builds all three executables |
| `main_en.tex` | project presentation (Beamer, English) |

> Note: `simulazione.cxx` (an earlier all-in-one driver) has been removed; its functionality is split across the three executables above.

---

## Build

**Requirements**

- ROOT 6 (with `root-config` on the `PATH`)
- CUDA toolkit + an NVIDIA GPU (only `studio_prestazioni` links CUDA)
- a C++17 compiler with OpenMP
- The `Makefile` targets Ada Lovelace (`sm_89`); change `ARCH` for your GPU and `CUDA_PATH` if CUDA is not in `/usr/local/cuda`.

**Commands**

```bash
make                    # builds analisi_sorgente, studio_prestazioni, visualizza_apparato
make analisi_sorgente   # single target
make clean
```

`analisi_sorgente` and `visualizza_apparato` do **not** link CUDA (their calibration runs on CPU/OpenMP); only `studio_prestazioni` does, because it benchmarks the GPU finder.

---

## Executables

### `analisi_sorgente`
Runs the data-driven calibration (default 100k events) and the noise scan.

```bash
./analisi_sorgente [N_calib_events] [N_scan_events]
```

Output plots: the calibration q-histogram with the robust-estimate curve (drawn both with and without the curve, and both for clone-cleaned and raw tracks), efficiency-vs-noise and ghosts/event-vs-noise. A companion canvas repeats the efficiency counting **all** generated MC particles in the denominator (see [Known caveats](#known-caveats)).

### `studio_prestazioni`
Two studies vs the number of generated tracks per event.

```bash
./studio_prestazioni [N_eval_events] [N_calib_events] [N_time_events]
```

- **Accuracy** — efficiency and purity, each with two curves: *clone removal only* and *clones + calibrated q selection*. A second canvas repeats efficiency without the acceptance cut in the denominator.
- **Timing** — average per-event time for the raw track search, CPU vs GPU. Clone removal (a shared host step) is deliberately kept **outside** the timer, so the two curves measure the same work. The GPU uses the persistent-memory finder.

### `visualizza_apparato`
Draws one event and walks it through the four selection stages, each on its own canvas:

1. all reconstructed tracks, **no cut on q**;
2. + selection of intercept in `[q_min, q_max]`;
3. + clone removal;
4. + the real calibrated selection (`μ_q ± 3σ_q`).

It also draws the two `(m, q)` phase-space histograms (no cut / calibrated) and, in multi-source mode, the calibration histogram itself. Noise is fixed at 5%.

---

## Configuration

Most knobs live in `config_new.h`. Two switches are worth calling out.

### Angular distribution — one line, applies everywhere

```cpp
const bool USE_GAUSSIAN_ANGLES = false;   // false = uniform, true = gaussian
```

All event generators call `Source::angles()`, which dispatches to the uniform or gaussian generator based on this flag — so a single edit switches every executable. The comparison is **fair at equal σ**: the uniform range is built as `mean ± std·√3`, whose standard deviation is exactly `std`, matching the gaussian `N(mean, std)`. Only the *shape* changes, not the width. (The gaussian has unbounded tails, so it emits some angles beyond the uniform's hard limit; those miss the tracker and are correctly dropped by the acceptance cut, slightly lowering geometric efficiency.)

### Multiple sources — in `visualizza_apparato.cxx`

```cpp
static const bool USE_MULTI_SOURCE = true;
static const std::vector<SourceSpec> MULTI_SOURCES = {
        {0.30f, 40},   // {height y (m), number of tracks}
        {0.50f, 40},
        {0.70f, 40},
};
```

With `false`, `visualizza_apparato` behaves exactly as the original single-source display. With `true`, hits from all listed sources are merged into each event and the calibration automatically switches to a **multi-peak search** (`FindMultiplePeaks`): it finds as many genuinely significant, well-separated peaks as the data supports (validated on width, prominence and Poisson significance against the background), and keeps a track if its `q` falls in the `±3σ` window of *any* peak. The number of peaks found is not forced to equal the number of configured sources — close sources can legitimately merge. When multiple sources fall outside the default `[q_min, q_max]` window, the calibration histogram range and binning are widened/refined so the peaks are actually resolved.

---

## Key algorithms in detail

### Raw track finding (CPU) — `FindRawTracksForEvent`
Per event: anchor on a layer-0 hit, seed on a last-layer hit, snap the nearest hit on each middle layer to the seed line, least-squares fit the three points, keep the track if `χ² < chi2_threshold`. Two optimisations cut the combinatorics from `O(hits₀·hits_last·hits_mid)`:
- the seeds giving `q_seed ∈ [q_min, q_max]` form a contiguous interval (the seed-`q` relation is affine and monotonic in the last-layer hit), found by **binary search** on the sorted last layer;
- the nearest middle-layer hit is found by **binary search** on the sorted middle layers.

### Raw track finding (GPU) — `TrackFinderKernel` + `GpuSingleEventFinder`
Same physics, opposite strategy: **one thread per (layer-0, last-layer) hit pair** (`16×16` blocks), no pruning — the combinatorics is absorbed by the hardware. The event is flattened into a single contiguous array (+ per-layer offsets/sizes) for one host→device transfer. Valid tracks are appended to a shared output buffer with a two-level reduction (a shared-memory atomic per block, then one global atomic per block) to avoid contention. Device buffers are **persistent**: allocated once and reused, removing the per-call `cudaMalloc/cudaFree` overhead that otherwise dominates single-event timing.

### Clone removal — `RimuoviCloni`
The finder can reconstruct the same physical particle several times when a nearby noise hit also passes the χ² cut. Two tracks are merged into one **only if both**: their `(m,q)` agree within `3σ_m`/`3σ_q` (the theoretical resolutions) **and** they share at least one hit. Requiring both avoids merging genuinely distinct near-collinear particles, and avoids merging accidental combinations that reuse the same noise hit. The merged track takes the cluster-average `(m,q)` and the best member's χ².

### Source calibration — `RobustPeakEstimate`
A classic gaussian fit is **avoided on purpose**: pixel quantisation propagated through the linear fit makes the reconstructed-`q` peak *comb-like* (discrete lines separated by exact-zero bins), on which a least-squares/Minuit fit can collapse onto a single tooth (`σ → 0`). Instead a deterministic, fit-free estimator is used:
- **Stage 1** — adaptively aggregate bins until the measured FWHM is physically credible relative to the theoretical `σ_q`; this fixes `σ`.
- **Stage 2** — a fixed-width iterative sigma-clipping that only recentres `μ`; it cannot diverge because the window width never depends on its own result.

### MC matching — `ValutaTracce`
A reconstructed track is matched to the MC particle with the **minimum Mahalanobis distance** in `(m,q)` space (using the full fit covariance, including the `m`–`q` anticorrelation), accepted if `D² ≤ 9` (the 3σ ellipse). Efficiency is computed only over particles **within geometric acceptance**; purity is good-tracks / reconstructed-tracks.

---

## Known caveats

- **Efficiency denominator.** The primary efficiency counts only MC particles that cross all layers within acceptance (particles that miss a layer can't be reconstructed by any algorithm). Companion plots also report efficiency over *all* generated particles, which folds in the geometric acceptance — clearly lower, and the honest number if you care about "fraction of everything emitted that was found".
- **`q_min`/`q_max` and the seed cut.** The seed acceptance window is hard-wired into the mass-production finder. If it is set narrow (e.g. `[0.4, 0.6]`), sources far from `sourceY` are killed at the seed stage and never appear in calibration; widen it (e.g. `[0, 1]`) for multi-source or displaced-source studies.
- **Histogram binning vs `σ_q`.** A real peak must be resolved by the binning: bins should be a fraction of `σ_q`, otherwise a peak collapses into ~1 bin and fails prominence tests. The executables size the calibration binning on the theoretical `σ_q` for this reason.

---

## Next steps

- 2D/3D tracker.
- Layer misalignment.
- Dead space between pixels (non-continuous pixels).
- Hough-transform track finding on FPGA; GPU-vs-FPGA timing and combinatorial-vs-Hough comparison.

---

*Author: Pietro Schiavone — University of Pavia.*

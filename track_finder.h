#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "config_new.h"
#include <vector>
#include <cstdint>
#include <string>

// A reconstructed straight track: slope m, intercept q (at x=0), fit chi2,
// and the hit index used on each layer.
struct Track {
	float m;
	float q;
	float chi2;
	int hit_indices[N_layers];
};

// Clone removal: merges raw tracks that share at least one hit and have
// compatible (m, q) within tolerance, averaging their parameters and keeping
// the best chi2. Shared by the CPU and GPU paths. verbose=false silences the
// per-call log (used inside the mass loops). Defined in track_finder_cpu.cxx.
std::vector<Track> RimuoviCloni(const std::vector<Track>& tracce_grezze,
                                 float tolleranza_m, float tolleranza_q,
                                 const std::string& hardware_label,
                                 bool verbose = true);

// Keep only tracks with q in [q_peak - n_sigma*sigma_q, q_peak + n_sigma*sigma_q].
// Apply AFTER clone removal, with q_peak/sigma_q from the calibration estimate.
std::vector<Track> FiltraTraccePerQ(const std::vector<Track>& tracce_in,
                                     float q_peak, float sigma_q,
                                     float n_sigma = 3.0f);

// ---------------------------------------------------------------------------
// "Mass-production" types: instead of returning every Track (GBs of data for
// millions of events), accumulate a histogram of the reconstructed q, which
// is all that is needed to find the source peak.
// ---------------------------------------------------------------------------
using EventHits  = std::vector<std::vector<float>>;  // [layer][hit] y positions of one event
using EventBatch = std::vector<EventHits>;            // [event][layer][hit]

struct QHistogram {
	std::vector<uint64_t> counts;
	float q_min = 0.0f;
	float q_max = 0.0f;
	int   n_bins = 0;
	uint64_t total_events_processed = 0;
	uint64_t total_tracks_found = 0;

	float bin_width() const { return n_bins > 0 ? (q_max - q_min) / n_bins : 0.0f; }
	float bin_center(int i) const { return q_min + (i + 0.5f) * bin_width(); }
};

// ---------------------------------------------------------------------------
// Peak (mu_q) and width (sigma_q) estimate WITHOUT a non-linear fit.
// A deterministic weighted sigma-clipping around the modal bin: it cannot
// collapse or diverge like a Minuit least-squares fit can when the physical
// sigma is much narrower than the search window. See peak_estimate.cxx.
// ---------------------------------------------------------------------------
struct PeakEstimate {
	float mu = 0.0f;
	float sigma = 0.0f;
	uint64_t entries_in_window = 0; // diagnostic: entries in the final window
	bool valid = false;
};

// initial_sigma_guess: starting width for the first clipping window (e.g. the
// theoretical sigma from CalcolaTolleranzeTeoriche()). n_iter ~6-10 is enough;
// n_sigma_clip=4 is a standard compromise between excluding the background and
// not cutting the real peak's tails.
PeakEstimate RobustPeakEstimate(const QHistogram& h, float initial_sigma_guess,
                                 int n_iter = 8, float n_sigma_clip = 4.0f);

// ---------------------------------------------------------------------------
// Single-event CPU finders.
// find_tracks_cpu_quiet: raw tracks + RimuoviCloni, no prints/timing (use in
//   loops over many events).
// find_tracks_cpu_raw:   only the raw finding (no clone removal, no prints),
//   the exact CPU equivalent of GpuSingleEventFinder::FindTracks, for fair
//   "who finds tracks faster" benchmarks (clone removal is a shared host step
//   kept out of the timer).
// ---------------------------------------------------------------------------
std::vector<Track> find_tracks_cpu_quiet(const EventHits& hits_lette);
std::vector<Track> find_tracks_cpu_raw(const EventHits& hits_lette);

// ---------------------------------------------------------------------------
// CPU calibration engine: for each event it finds raw tracks and histograms
// their q. clean_clones (default true) selects whether RimuoviCloni is
// applied before binning: true reproduces the single-event chain exactly
// (raw tracks + real clone merging); false bins the RAW combinatorial tracks
// as found, with no merging at all -- useful to see the effect of clone
// removal on the calibration peak/background. OpenMP-parallel over the batch.
// ---------------------------------------------------------------------------
class CpuCleanedQHistogramEngine {
public:
	CpuCleanedQHistogramEngine(int n_bins, float q_hist_min, float q_hist_max, bool clean_clones = true);

	void ProcessBatch(const EventBatch& batch);
	QHistogram GetHistogram() const;
	void ResetHistogram();

	double last_batch_ms() const { return last_ms_; }
	uint64_t events_processed() const { return total_events_; }

private:
	int n_bins_;
	float q_min_, q_max_;
	bool clean_clones_;
	std::vector<uint64_t> counts_;
	uint64_t total_events_ = 0;
	double last_ms_ = 0.0;
};

// ---------------------------------------------------------------------------
// GpuSingleEventFinder: single-event GPU finder with PERSISTENT device memory.
// Device buffers are allocated once and reused across calls (growing only if
// an event needs more), and the output buffer is sized on the real case (at
// most hits_L0 * hits_Llast pairs). This removes the fixed per-call
// cudaMalloc/cudaFree + PCIe overhead. Returns RAW tracks (no clone removal),
// so apply RimuoviCloni on the host afterwards, like find_tracks_cpu_quiet.
// ---------------------------------------------------------------------------
class GpuSingleEventFinder {
public:
	explicit GpuSingleEventFinder(size_t initial_track_capacity = 1u << 20);
	~GpuSingleEventFinder();

	GpuSingleEventFinder(const GpuSingleEventFinder&) = delete;
	GpuSingleEventFinder& operator=(const GpuSingleEventFinder&) = delete;

	std::vector<Track> FindTracks(const EventHits& hits_lette);
	double last_ms() const { return last_ms_; }

private:
	void EnsureCapacity(size_t n_hits, size_t n_tracks);

	float* d_y_hits_ = nullptr;
	int*   d_offsets_ = nullptr;
	int*   d_sizes_ = nullptr;
	Track* d_tracks_out_ = nullptr;
	int*   d_track_count_ = nullptr;

	size_t cap_hits_ = 0;
	size_t cap_tracks_ = 0;
	double last_ms_ = 0.0;
};

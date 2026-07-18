#ifndef VISUALIZER_H
#define VISUALIZER_H

#include <vector>
#include <utility> // for std::pair
#include "track_finder.h" 
#include <TApplication.h>

class Visualizer {
public:
    	Visualizer(int argc, char* argv[]);
    	~Visualizer();

	// Noise scan (0% -> 10% by default, n_noise_steps points). For each
	// level: generate events, reconstruct with find_tracks_cpu_quiet (raw
	// tracks + RimuoviCloni), apply the calibrated q-window selection
	// (mu_q +/- 3 sigma_q from the gaussian fit), and evaluate efficiency
	// and false positives against MC truth (ValutaTracce, Mahalanobis
	// ellipse). Draws efficiency-vs-noise and ghosts/event-vs-noise on two
	// panels, PLUS a second, identically laid out canvas where the left
	// panel restates efficiency counting mc_trovati / mc_totali (i.e. NOT
	// excluding MC particles that never entered geometric acceptance from
	// the denominator) and the right panel repeats the same ghosts/event
	// curve (unaffected by that choice).
	void RunNoiseScan(float mean, float std_dev, float mu_q, float sigma_q,
	                   int n_events_per_step = 200,
	                   float noise_min = 0.0f, float noise_max = 0.10f,
	                   int n_noise_steps = 11);

    	// Single-histogram calibration plot: draws the histogram plus, unless
    	// draw_fit_curve=false, the robust-estimate gaussian curve on top. The
    	// robust estimate is always computed (mu_q/sigma_q are always
    	// returned) -- draw_fit_curve only controls whether the red curve is
    	// actually drawn, useful to look at the raw histogram shape on its own.
    	// name_suffix distinguishes ROOT object names (histogram/canvas/TF1)
    	// across multiple calls in the same run, so the same overload can be
    	// reused for different histograms (e.g. clone-cleaned vs raw tracks,
    	// or fit vs no-fit) without ROOT complaining about duplicate names.
    	std::pair<float, float> DrawSourceSearchAndFit(const QHistogram& h,
    	                                                const char* name_suffix = "",
    	                                                const char* title = "Calibrazione: q ricostruita; q ricostruita (m); Conteggi",
    	                                                bool draw_fit_curve = true);

    	// Accuracy vs number of generated tracks per event: two panels
    	// (efficiency left, purity right), each with two curves: "clone
    	// removal only" and "clones + calibrated q selection". All vectors
    	// must have the same length as n_tracks_vals.
    	//
    	// eff_solo_cloni_all_mc/eff_cloni_piu_q_all_mc are OPTIONAL companion
    	// efficiency curves computed WITHOUT excluding MC particles outside
    	// geometric acceptance from the denominator (mc_trovati / mc_totali
    	// instead of mc_trovati / (mc_trovati + mc_persi), see truth.h). When
    	// both are non-empty (same length as n_tracks_vals), a second canvas
    	// is drawn mirroring the first one: left panel is the "all MC" 
    	// efficiency, right panel restates the same purity curves (purity is
    	// unaffected by the acceptance cut, so it is not recomputed).
    	void DrawAccuracyVsTracks(const std::vector<double>& n_tracks_vals,
    	                           const std::vector<double>& eff_solo_cloni,
    	                           const std::vector<double>& eff_cloni_piu_q,
    	                           const std::vector<double>& pur_solo_cloni,
    	                           const std::vector<double>& pur_cloni_piu_q,
    	                           const std::vector<double>& eff_solo_cloni_all_mc = {},
    	                           const std::vector<double>& eff_cloni_piu_q_all_mc = {});

    	// Track-finding time vs number of generated tracks per event: one
    	// plot with both curves overlaid (CPU blue, GPU green) and a legend,
    	// average time per event in ms. No selection on the source q: only
    	// the seed acceptance cut [q_min, q_max] applies, already internal to
    	// the search algorithms.
    	void DrawTimingVsTracks(const std::vector<double>& n_tracks_vals,
    	                         const std::vector<double>& cpu_ms_per_evento,
    	                         const std::vector<double>& gpu_ms_per_evento);

    	void Run();

private:
    	TApplication* m_app;

    	// Internal helper to convert a QHistogram into a ROOT TH1F
    	class TH1F* MakeRootHist(const QHistogram& h, const char* name, const char* title);
};

#endif // VISUALIZER_H

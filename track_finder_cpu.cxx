#include "track_finder.h"
#include "config_new.h"
#include "truth.h"
#include <iostream>
#include <cstdint>
#include <chrono>
#include <cmath>
#include <string>
#include <algorithm>
#include <utility>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

// =============================================================================
// Clone removal (shared by CPU and GPU paths). Merges raw tracks that share
// at least one hit and have compatible (m, q) within tolerance, averaging
// their parameters and keeping the best chi2. O(n^2), fine for one event.
// verbose=false is used inside the mass loops to avoid per-event logging.
// =============================================================================

std::vector<Track> RimuoviCloni(const std::vector<Track>& tracce_grezze, float tolleranza_m, float tolleranza_q, const std::string& hardware_label, bool verbose) {
        if (tracce_grezze.empty()) return {};

        std::vector<Track> tracce_pulite;
        std::vector<bool> processata(tracce_grezze.size(), false);

        for (size_t i = 0; i < tracce_grezze.size(); ++i) {
                if (processata[i]) continue;

                float somma_m = tracce_grezze[i].m;
                float somma_q = tracce_grezze[i].q;
                float miglior_chi2 = tracce_grezze[i].chi2;
                int contatore_cloni = 1;

                processata[i] = true;

                for (size_t j = i + 1; j < tracce_grezze.size(); ++j) {
                        if (processata[j]) continue;

                        float delta_m = std::abs(tracce_grezze[i].m - tracce_grezze[j].m);
                        float delta_q = std::abs(tracce_grezze[i].q - tracce_grezze[j].q);

                        bool parametri_simili = (delta_m < tolleranza_m && delta_q < tolleranza_q);

                        bool condividono_hit = false;
                        for (size_t L = 0; L < N_layers; ++L) {
                                if (tracce_grezze[i].hit_indices[L] == tracce_grezze[j].hit_indices[L]) {
                                        condividono_hit = true;
                                        break;
                                }
                        }

                        if (parametri_simili && condividono_hit) {
                                somma_m += tracce_grezze[j].m;
                                somma_q += tracce_grezze[j].q;

                                if (tracce_grezze[j].chi2 < miglior_chi2) {
                                        miglior_chi2 = tracce_grezze[j].chi2;
                                }

                                contatore_cloni++;
                                processata[j] = true;
                        }
                }

                Track traccia_unita = tracce_grezze[i];
                traccia_unita.m = somma_m / contatore_cloni;
                traccia_unita.q = somma_q / contatore_cloni;
                traccia_unita.chi2 = miglior_chi2;

                tracce_pulite.push_back(traccia_unita);
        }

        if (verbose) {
                std::cout << "\n[" << hardware_label << " - Ghost Suppression]" << std::endl;
                std::cout << "  -> Taglio a 3 Sigma: Delta_M = " << tolleranza_m << ", Delta_Q = " << tolleranza_q << std::endl;
                std::cout << "  -> Tracce trovate : " << tracce_grezze.size() << "  -> Tracce pulite : " << tracce_pulite.size() << std::endl;
        }

        return tracce_pulite;
}

// Keep only tracks whose q falls inside the calibrated window
// [q_peak - n_sigma*sigma_q, q_peak + n_sigma*sigma_q]. Applied AFTER clone
// removal, with q_peak/sigma_q from the calibration peak estimate.
std::vector<Track> FiltraTraccePerQ(const std::vector<Track>& tracce_in, float q_peak, float sigma_q, float n_sigma) {
        std::vector<Track> tracce_selezionate;
        const float q_min_acc = q_peak - (n_sigma * sigma_q);
        const float q_max_acc = q_peak + (n_sigma * sigma_q);
        for (const auto& traccia : tracce_in) {
                if (traccia.q >= q_min_acc && traccia.q <= q_max_acc) {
                        tracce_selezionate.push_back(traccia);
                }
        }
        return tracce_selezionate;
}

// Raw track finding (before clone removal) for ONE event. Shared single
// implementation behind find_tracks_cpu_quiet, find_tracks_cpu_raw and
// CpuCleanedQHistogramEngine. `out` is NOT cleared here so the caller can
// reuse the same (already-allocated) vector across consecutive events.
// Algorithm: anchor = layer 0 hit, seed = last-layer hit; keep seeds whose
// q_seed lands in [q_min,q_max]; snap the nearest hit on each middle layer
// to the seed line; least-squares fit the 3 points; keep if chi2 < threshold.
static void FindRawTracksForEvent(const EventHits& y_hit, std::vector<Track>& out) {
        if (y_hit.size() != N_layers) return;

        const size_t hits_layer_0 = y_hit[0].size();
        const size_t hits_layer_last = y_hit.back().size();
        if (hits_layer_0 == 0 || hits_layer_last == 0) return;

        const float x_end = d_s_l + (N_layers - 1) * dist_interlayers;
        const float delta_x = x_end - d_s_l;

        // Optimisation: at fixed y_first, q_seed(y_last) = a*y_last + b is
        // affine and monotonic in y_last (a = -d_s_l/delta_x, geometry only),
        // so the y_last giving q_seed in [q_min,q_max] form a contiguous
        // interval. Sorting last-layer hits once lets us find that interval by
        // binary search per y_first, turning O(hits0*hitsLast) into roughly
        // O((hits0+hitsLast) log hitsLast + candidates actually found).
        const float a = -d_s_l / delta_x;

        static thread_local std::vector<std::pair<float,int>> sorted_last;
        sorted_last.clear();
        sorted_last.reserve(hits_layer_last);
        for (size_t j = 0; j < hits_layer_last; j++) {
                sorted_last.emplace_back(y_hit.back()[j], (int)j);
        }
        std::sort(sorted_last.begin(), sorted_last.end());

        // Second optimisation: also sort the middle layers once per event so
        // the nearest hit to the predicted position is found by binary search
        // (O(log hits_mid)) instead of a linear scan per surviving candidate.
        static thread_local std::vector<std::vector<std::pair<float,int>>> sorted_mid;
        sorted_mid.resize(N_layers); // no-op if already the right size
        for (size_t L = 1; L + 1 < N_layers; L++) {
                sorted_mid[L].clear();
                sorted_mid[L].reserve(y_hit[L].size());
                for (size_t k = 0; k < y_hit[L].size(); k++) {
                        sorted_mid[L].emplace_back(y_hit[L][k], (int)k);
                }
                std::sort(sorted_mid[L].begin(), sorted_mid[L].end());
        }

        // Safety fallback: if a ~ 0 (degenerate geometry) the useful interval
        // would cover the whole axis, so scan all last-layer hits instead.
        const bool degenerate = (fabsf(a) < 1e-9f);

        for (size_t i = 0; i < hits_layer_0; i++) {
                const float y_first = y_hit[0][i];
                const float b = (x_end / delta_x) * y_first;

                auto it_begin = sorted_last.begin();
                auto it_end   = sorted_last.end();

                if (!degenerate) {
                        const float bound1 = (q_min - b) / a;
                        const float bound2 = (q_max - b) / a;
                        const float lo = std::min(bound1, bound2);
                        const float hi = std::max(bound1, bound2);

                        it_begin = std::lower_bound(sorted_last.begin(), sorted_last.end(),
                                std::make_pair(lo, std::numeric_limits<int>::min()));
                        it_end = std::upper_bound(sorted_last.begin(), sorted_last.end(),
                                std::make_pair(hi, std::numeric_limits<int>::max()));
                }

                for (auto it = it_begin; it != it_end; ++it) {
                        const float y_last = it->first;
                        const size_t j = (size_t)it->second;

                        const float m_seed = (y_last - y_first) / delta_x;
                        const float q_seed = y_last - (m_seed * x_end);

                        // Exact re-check on top of the binary search, to cover the
                        // interval edges where float rounding could include/exclude
                        // an element by a hair.
                        if (q_seed < q_min || q_seed > q_max) continue;

                        float best_y[N_layers];
                        int best_indices[N_layers];

                        best_y[0] = y_first;
                        best_indices[0] = (int)i;
                        best_y[N_layers - 1] = y_last;
                        best_indices[N_layers - 1] = (int)j;

                        bool track_is_valid = true;

                        for (size_t L = 1; L < (N_layers - 1); L++) {
                                const auto& layer_sorted = sorted_mid[L];
                                if (layer_sorted.empty()) { track_is_valid = false; break; }

                                const float x_L = d_s_l + L * dist_interlayers;
                                const float y_theory = (m_seed * x_L) + q_seed;

                                // Nearest hit in a sorted array: check the first element
                                // >= y_theory and the one just before it; one of the two
                                // is necessarily the closest.
                                auto pos = std::lower_bound(layer_sorted.begin(), layer_sorted.end(),
                                        std::make_pair(y_theory, std::numeric_limits<int>::min()));

                                float closest_y = 0.0f;
                                int closest_idx = -1;
                                float min_dist_sq = 1e18f;

                                if (pos != layer_sorted.end()) {
                                        const float diff = pos->first - y_theory;
                                        min_dist_sq = diff * diff;
                                        closest_y = pos->first;
                                        closest_idx = pos->second;
                                }
                                if (pos != layer_sorted.begin()) {
                                        auto prev_it = std::prev(pos);
                                        const float diff = prev_it->first - y_theory;
                                        const float dist_sq = diff * diff;
                                        if (dist_sq < min_dist_sq) {
                                                min_dist_sq = dist_sq;
                                                closest_y = prev_it->first;
                                                closest_idx = prev_it->second;
                                        }
                                }

                                best_y[L] = closest_y;
                                best_indices[L] = closest_idx;
                        }

                        if (!track_is_valid) continue;

                        float Sx = 0.0f, Sxx = 0.0f, Sy = 0.0f, Sxy = 0.0f;
                        for (size_t L = 0; L < N_layers; L++) {
                                const float x_L = d_s_l + L * dist_interlayers;
                                Sx += x_L; Sxx += x_L * x_L;
                                Sy += best_y[L]; Sxy += x_L * best_y[L];
                        }

                        const float N_float = static_cast<float>(N_layers);
                        const float Delta = (N_float * Sxx) - (Sx * Sx);
                        if (fabsf(Delta) < 1e-6f) continue;

                        const float m_fit = ((N_float * Sxy) - (Sx * Sy)) / Delta;
                        const float q_fit = ((Sxx * Sy) - (Sx * Sxy)) / Delta;

                        float chi2_globale = 0.0f;
                        for (size_t L = 0; L < N_layers; L++) {
                                const float x_L = d_s_l + L * dist_interlayers;
                                const float residuo = best_y[L] - ((m_fit * x_L) + q_fit);
                                chi2_globale += (residuo * residuo);
                        }
                        chi2_globale /= (sigma * sigma);

                        if (chi2_globale < chi2_threshold) {
                                Track t;
                                t.m = m_fit;
                                t.q = q_fit;
                                t.chi2 = chi2_globale;
                                for (size_t L = 0; L < N_layers; L++) t.hit_indices[L] = best_indices[L];
                                out.push_back(t);
                        }
                }
        }
}

std::vector<Track> find_tracks_cpu_raw(const EventHits& y_hit) {
        std::vector<Track> raw;
        FindRawTracksForEvent(y_hit, raw);
        return raw;
}

std::vector<Track> find_tracks_cpu_quiet(const EventHits& y_hit) {
        std::vector<Track> valid_tracks;
        FindRawTracksForEvent(y_hit, valid_tracks);
        DefinizioneTagli tagli = CalcolaTolleranzeTeoriche();
        return RimuoviCloni(valid_tracks, tagli.delta_m_cut, tagli.delta_q_cut, "CPU", /*verbose=*/false);
}

// =============================================================================
// CpuCleanedQHistogramEngine: mass-production CPU engine used for source
// calibration. For every event it runs the raw finder, applies the REAL
// clone removal, and histograms the q of the surviving cleaned tracks.
// Parallelised with OpenMP over the events of the batch: each thread fills a
// private histogram and reuses its own buffers, with a single O(n_bins)
// critical section per thread at the end. Clone-removal tolerances depend
// only on geometry, so they are computed ONCE per batch, not per event.
// =============================================================================

CpuCleanedQHistogramEngine::CpuCleanedQHistogramEngine(int n_bins, float q_hist_min, float q_hist_max, bool clean_clones)
        : n_bins_(n_bins), q_min_(q_hist_min), q_max_(q_hist_max), clean_clones_(clean_clones), counts_(n_bins, 0)
{
#ifndef _OPENMP
        std::cerr << "[CpuCleanedQHistogramEngine] Warning: built without OpenMP (-fopenmp), "
                     "will run on a single core.\n";
#endif
}

void CpuCleanedQHistogramEngine::ProcessBatch(const EventBatch& batch) {
        auto t0 = std::chrono::high_resolution_clock::now();

        const int nb = n_bins_;
        const float qlo = q_min_;
        const float bin_w_inv = nb / (q_max_ - q_min_);
        const long n_events = (long)batch.size();
        const bool clean = clean_clones_;

        const DefinizioneTagli tagli = CalcolaTolleranzeTeoriche();

        std::vector<uint64_t> merged(nb, 0);

        #pragma omp parallel
        {
                std::vector<uint64_t> local(nb, 0);
                std::vector<Track> raw_tracks;
                std::vector<Track> cleaned;
                raw_tracks.reserve(64);

                #pragma omp for schedule(dynamic, 16) nowait
                for (long e = 0; e < n_events; e++) {
                        raw_tracks.clear();
                        FindRawTracksForEvent(batch[e], raw_tracks);
                        if (raw_tracks.empty()) continue;

                        // With clean_clones=true, bin the clone-merged tracks (the real
                        // single-event chain). With clean_clones=false, bin the raw
                        // combinatorial tracks directly: no merging at all.
                        const std::vector<Track>* to_bin = &raw_tracks;
                        if (clean) {
                                cleaned = RimuoviCloni(raw_tracks, tagli.delta_m_cut, tagli.delta_q_cut, "calib", /*verbose=*/false);
                                to_bin = &cleaned;
                        }

                        for (const auto& t : *to_bin) {
                                const int bin = (int)((t.q - qlo) * bin_w_inv);
                                if (bin >= 0 && bin < nb) local[bin]++;
                        }
                }

                #pragma omp critical
                {
                        for (int b = 0; b < nb; b++) merged[b] += local[b];
                }
        }

        for (int b = 0; b < nb; b++) counts_[b] += merged[b];
        total_events_ += (uint64_t)n_events;

        auto t1 = std::chrono::high_resolution_clock::now();
        last_ms_ = std::chrono::duration<double, std::milli>(t1 - t0).count();
}

QHistogram CpuCleanedQHistogramEngine::GetHistogram() const {
        QHistogram h;
        h.counts = counts_;
        h.q_min = q_min_;
        h.q_max = q_max_;
        h.n_bins = n_bins_;
        h.total_events_processed = total_events_;
        uint64_t tot = 0;
        for (auto c : counts_) tot += c;
        h.total_tracks_found = tot;
        return h;
}

void CpuCleanedQHistogramEngine::ResetHistogram() {
        std::fill(counts_.begin(), counts_.end(), 0);
        total_events_ = 0;
}

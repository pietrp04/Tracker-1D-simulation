// studio_prestazioni.cxx
// ---------------------------------------------------------------------------
// Final graphs of the project.
//
// GRAPH A - Accuracy vs number of tracks generated per event, two curves:
//   (1) clone removal ONLY (find_tracks_cpu_quiet: raw tracks + RimuoviCloni),
//       evaluated against MC truth;
//   (2) clones + CALIBRATED Q SELECTION: for each value of N, the full
//       calibration is redone as in analisi_sorgente (CpuCleanedQHistogramEngine
//       on events with that same N + RobustPeakEstimate -> mu_q, sigma_q),
//       then FiltraTraccePerQ(mu_q, sigma_q, 3) before evaluation.
//   "Accuracy" is shown on TWO panels to cover both readings of the term:
//   EFFICIENCY (MC particles in acceptance that were found) and PURITY
//   (fraction of reconstructed tracks matching a real particle).
//   A second, identically laid out canvas is also drawn where the efficiency
//   panel counts mc_trovati / mc_totali instead -- i.e. WITHOUT excluding MC
//   particles that missed a layer (never entered geometric acceptance) from
//   the denominator; the purity panel is repeated unchanged, since purity
//   does not depend on this choice.
//
// GRAPH B - Track-finding time vs number of tracks generated per event: CPU
//   panel and GPU panel (average time per event, in ms). As requested, NO
//   selection on the source q is applied here: the only active cut is the
//   seed acceptance range [q_min, q_max] from config_new.h, already internal
//   to both algorithms (find_tracks_cpu_quiet and the GPU finder) -- changing
//   q_min/q_max in the config directly changes the work measured here.
//
// Usage:
//   ./studio_prestazioni [N_eval_events] [N_calib_events] [N_timing_events]
//   (default: 200 events/point for accuracy, 3000 for calibration,
//    30 per timing point)
// ---------------------------------------------------------------------------

#include "detector.h"
#include "source.h"
#include "track_finder.h"
#include "truth.h"
#include "visualizer.h"
#include "config_new.h"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <cstdlib>
#include <vector>

// Generate a single event with n_tracce particles (runtime overload of
// Source::angles added for this study; the chain is identical to
// GenerateOneEvent in analisi_sorgente.cxx).
static EventHits GenerateOneEventN(Source& sorgente, Detector& detector,
                                    float prob_rumore, size_t n_tracce,
                                    std::vector<float>* angoli_out = nullptr) {
        std::vector<float> angoli = sorgente.angles(n_tracce);
        std::vector<std::vector<float>> y_teoriche = sorgente.y_th(angoli);
        std::vector<std::vector<float>> y_hit_true = sorgente.Check_hits(y_teoriche);

        EventHits pixel_positions(N_layers);
        for (size_t i = 0; i < N_layers; i++) {
                std::vector<uint32_t> pixels = detector.pos2pix(y_hit_true[i]);
                sorgente.MakeNoisePhysical(prob_rumore, pixels);
                pixel_positions[i] = detector.pix2pos(pixels);
        }
        if (angoli_out) *angoli_out = std::move(angoli);
        return pixel_positions;
}

int main(int argc, char* argv[]) {

        // ------------------ parameters ------------------
        int N_EVAL_EVENTS  = 200;   // events per point (accuracy)
        int N_CALIB_EVENTS = 3000;  // calibration events per point (q-selection variant)
        int N_TIME_EVENTS  = 30;    // events per point (CPU/GPU timing)

        if (argc > 1) N_EVAL_EVENTS  = std::atoi(argv[1]);
        if (argc > 2) N_CALIB_EVENTS = std::atoi(argv[2]);
        if (argc > 3) N_TIME_EVENTS  = std::atoi(argv[3]);

        const float mean = 0.0f;
        const float std_dev = 0.267f;
        const float prob_rumore = 0.05f;

        // "Tracks per event" values to scan for the ACCURACY graph. Note:
        // above ~50-100 tracks with l_pixel=0.0025 the lit pixels start to
        // saturate/merge (400 pixels total per layer), so high-N statistics
        // must be read accounting for this physical occupancy effect, not
        // just the algorithm.
        const std::vector<int> N_VALS = {5, 10, 20, 50, 100, 200, 400};

        // Points for the TIMING graph: one every 50 tracks, from 50 to 1000.
        // Kept as a separate list from the accuracy one because each accuracy
        // point costs a full calibration (N_CALIB_EVENTS events + peak
        // estimate), while a timing point only costs N_TIME_EVENTS
        // reconstructions: a much finer grid is affordable here without
        // blowing up total runtime.
        std::vector<int> N_VALS_TIMING;
        for (int n = 50; n <= 1000; n += 50) N_VALS_TIMING.push_back(n);

        std::cout << "=== STUDIO PRESTAZIONI vs NUMERO DI TRACCE ===\n";
        std::cout << "Distribuzione angoli: "
                  << (USE_GAUSSIAN_ANGLES ? "GAUSSIANA N(mean, std)" : "UNIFORME [mean +/- std*sqrt(3)]")
                  << "  (mean=" << mean << ", std=" << std_dev << ")\n";
        std::cout << "Punti accuratezza: ";
        for (int n : N_VALS) std::cout << n << " ";
        std::cout << "\nPunti timing: " << N_VALS_TIMING.front() << " -> " << N_VALS_TIMING.back()
                  << " a passi di 50 (" << N_VALS_TIMING.size() << " punti)";

        // Saturation warning: the number of DISTINCT hits per layer cannot
        // exceed the number of pixels (l_tracker/l_pixel). Above that N,
        // more tracks land on the same pixels and the track-finding workload
        // stops growing: the timing curve flattens by construction, not by
        // any property of the algorithm.
        const int n_pixel_per_layer = (int)std::round(l_tracker / l_pixel);
        if (N_VALS_TIMING.back() > n_pixel_per_layer) {
                std::cout << "\nATTENZIONE: i layer hanno " << n_pixel_per_layer << " pixel ciascuno: oltre N~"
                          << n_pixel_per_layer << " gli hit distinti saturano e la curva dei tempi si"
                          << "\nappiattisce (piu' tracce nello stesso pixel = stesso numero di hit da processare)."
                          << "\nPer vedere lo scaling vero fino a " << N_VALS_TIMING.back()
                          << " tracce, riduci l_pixel nel config.";
        }
        std::cout << "\nEventi/punto: " << N_EVAL_EVENTS << " (accuratezza), "
                  << N_CALIB_EVENTS << " (calibrazione), " << N_TIME_EVENTS << " (timing)\n\n";

        Visualizer viz(argc, argv);

        Source sorgente(mean, std_dev);
        Detector detector;

        // Fixed parameters for calibration binning/estimation (as in
        // analisi_sorgente.cxx: bins fine relative to the theoretical sigma)
        const DefinizioneTagli tagli = CalcolaTolleranzeTeoriche();
        const float Q_LO = q_min - 0.05f;
        const float Q_HI = q_max + 0.05f;
        int N_BINS = (tagli.sigma_q > 1e-9f)
            ? (int)((Q_HI - Q_LO) / (tagli.sigma_q / 50.0f)) : 100;
        N_BINS = std::max(100, std::min(N_BINS, 20000));

        // =====================================================================
        // GRAPH A - accuracy vs number of tracks
        // =====================================================================
        std::vector<double> x_vals, eff_cloni, eff_q, pur_cloni, pur_q, eff_cloni_all, eff_q_all;

        for (int n_tracce : N_VALS) {
                std::cout << "[Accuratezza] N=" << n_tracce << ": calibrazione (" << N_CALIB_EVENTS << " eventi)..." << std::flush;

                // --- calibration at this N (for the q-selection variant):
                // same chain as analisi_sorgente, but with n_tracce events.
                CpuCleanedQHistogramEngine calib(N_BINS, Q_LO, Q_HI);
                {
                        EventBatch batch_calib;
                        batch_calib.reserve(N_CALIB_EVENTS);
                        for (int e = 0; e < N_CALIB_EVENTS; e++) {
                                batch_calib.push_back(GenerateOneEventN(sorgente, detector, prob_rumore, (size_t)n_tracce));
                        }
                        calib.ProcessBatch(batch_calib);
                }
                QHistogram h_calib = calib.GetHistogram();
                PeakEstimate stima = RobustPeakEstimate(h_calib, tagli.sigma_q);

                if (!stima.valid) {
                        std::cout << " STIMA NON VALIDA (istogramma vuoto?), salto il punto.\n";
                        continue;
                }
                std::cout << " mu_q=" << stima.mu << " sigma_q=" << stima.sigma << " | valutazione..." << std::flush;

                // --- evaluation on fresh events: both variants on the SAME
                // events, so the comparison between the two curves is not
                // contaminated by different statistical fluctuations.
                long long trov_c = 0, gen_c = 0, buone_c = 0, ghost_c = 0;
                long long trov_q = 0, gen_q = 0, buone_q = 0, ghost_q = 0;
                long long gen_all = 0; // all generated MC particles, acceptance or not (same for both variants: same events)

                for (int e = 0; e < N_EVAL_EVENTS; e++) {
                        std::vector<float> angoli;
                        EventHits hits = GenerateOneEventN(sorgente, detector, prob_rumore, (size_t)n_tracce, &angoli);

                        std::vector<Track> pulite = find_tracks_cpu_quiet(hits);

                        // variant (1): clone removal only
                        AnalisiTracce r1 = ValutaTracce(pulite, angoli, sourceY);
                        trov_c  += r1.mc_trovati;
                        gen_c   += r1.mc_trovati + r1.mc_persi;
                        buone_c += r1.tracce_buone;
                        ghost_c += r1.tracce_ghost;
                        gen_all += r1.mc_totali;

                        // variant (2): clones + calibrated q selection
                        std::vector<Track> selezionate = FiltraTraccePerQ(pulite, stima.mu, stima.sigma, 3.0f);
                        AnalisiTracce r2 = ValutaTracce(selezionate, angoli, sourceY);
                        trov_q  += r2.mc_trovati;
                        gen_q   += r2.mc_trovati + r2.mc_persi;
                        buone_q += r2.tracce_buone;
                        ghost_q += r2.tracce_ghost;
                }

                // efficiency: particles in acceptance found / in acceptance
                // efficiency (all MC): particles found / ALL generated (mc_totali),
                // i.e. NOT excluding particles that missed a layer from the
                // denominator -- see truth.h/truth.cxx.
                // purity: good tracks / total reconstructed tracks
                const double e1 = (gen_c > 0) ? 100.0 * trov_c / gen_c : 0.0;
                const double e2 = (gen_q > 0) ? 100.0 * trov_q / gen_q : 0.0;
                const double e1_all = (gen_all > 0) ? 100.0 * trov_c / gen_all : 0.0;
                const double e2_all = (gen_all > 0) ? 100.0 * trov_q / gen_all : 0.0;
                const double p1 = (buone_c + ghost_c > 0) ? 100.0 * buone_c / (buone_c + ghost_c) : 0.0;
                const double p2 = (buone_q + ghost_q > 0) ? 100.0 * buone_q / (buone_q + ghost_q) : 0.0;

                x_vals.push_back((double)n_tracce);
                eff_cloni.push_back(e1);
                eff_q.push_back(e2);
                eff_cloni_all.push_back(e1_all);
                eff_q_all.push_back(e2_all);
                pur_cloni.push_back(p1);
                pur_q.push_back(p2);

                std::cout << " eff(cloni)=" << std::fixed << std::setprecision(1) << e1
                          << "% eff(cloni+q)=" << e2
                          << "% eff_tutte_MC(cloni)=" << e1_all
                          << "% eff_tutte_MC(cloni+q)=" << e2_all
                          << "% pur(cloni)=" << p1
                          << "% pur(cloni+q)=" << p2 << "%\n";
        }

        viz.DrawAccuracyVsTracks(x_vals, eff_cloni, eff_q, pur_cloni, pur_q, eff_cloni_all, eff_q_all);

        // =====================================================================
        // GRAPH B - CPU and GPU timing vs number of tracks.
        // NO selection on the source q: only the [q_min, q_max] acceptance
        // range from the config applies, internal to the algorithms.
        // =====================================================================
        std::cout << "\n[Timing] warm-up GPU (contesto CUDA + prima allocazione persistente)..." << std::endl;
        // Persistent-memory GPU engine (see GpuSingleEventFinder in
        // track_finder.h): device buffers are allocated ONCE here and reused
        // for every event. The old find_tracks() allocated and freed ~600 MB
        // of VRAM on every call: that fixed overhead, not the computation,
        // was why the GPU used to look ~0.5 ms slower than the CPU even at
        // high occupancy.
        GpuSingleEventFinder gpu_finder;
        {
                // The FIRST CUDA call of a process also pays for context
                // initialisation (hundreds of ms): without a warm-up it would
                // all land inside the first graph point.
                EventHits warmup = GenerateOneEventN(sorgente, detector, prob_rumore, 10);
                (void)gpu_finder.FindTracks(warmup);
        }

        std::vector<double> x_time, cpu_ms, gpu_ms;

        for (int n_tracce : N_VALS_TIMING) {
                // same events for CPU and GPU: apples-to-apples comparison
                EventBatch eventi(N_TIME_EVENTS);
                for (int e = 0; e < N_TIME_EVENTS; e++) {
                        eventi[e] = GenerateOneEventN(sorgente, detector, prob_rumore, (size_t)n_tracce);
                }

                // --- CPU: raw track search ONLY, the same work measured on
                //     the GPU side (clone removal, identical on both, stays
                //     out of the timer) ---
                auto t0 = std::chrono::high_resolution_clock::now();
                for (const auto& ev : eventi) {
                        volatile size_t sink = find_tracks_cpu_raw(ev).size();
                        (void)sink; // prevents the compiler from eliding the call
                }
                auto t1 = std::chrono::high_resolution_clock::now();
                const double ms_cpu = std::chrono::duration<double, std::milli>(t1 - t0).count() / N_TIME_EVENTS;

                // --- GPU: persistent buffers, raw search only (mirrors the
                //     CPU measurement above) ---
                auto t2 = std::chrono::high_resolution_clock::now();
                for (const auto& ev : eventi) {
                        volatile size_t sink = gpu_finder.FindTracks(ev).size();
                        (void)sink;
                }
                auto t3 = std::chrono::high_resolution_clock::now();
                const double ms_gpu = std::chrono::duration<double, std::milli>(t3 - t2).count() / N_TIME_EVENTS;

                x_time.push_back((double)n_tracce);
                cpu_ms.push_back(ms_cpu);
                gpu_ms.push_back(ms_gpu);

                std::cout << "[Timing] N=" << n_tracce
                          << "  CPU=" << std::fixed << std::setprecision(3) << ms_cpu << " ms/evento"
                          << "  GPU=" << ms_gpu << " ms/evento\n";
        }

        viz.DrawTimingVsTracks(x_time, cpu_ms, gpu_ms);

        std::cout << "\nNOTA sul confronto CPU/GPU: entrambe le curve misurano la SOLA ricerca\n"
                     "delle tracce grezze (il clone-removal, identico su entrambi i lati, e'\n"
                     "fuori dal cronometro). Il tempo GPU usa il motore a\n"
                     "memoria persistente (GpuSingleEventFinder): resta un costo fisso\n"
                     "irriducibile per evento (lancio kernel + trasferimenti PCIe, ~decine di\n"
                     "microsecondi), quindi a BASSA occupanza la CPU puo' ancora vincere; il\n"
                     "sorpasso della GPU emerge al crescere degli hit per layer, dove il\n"
                     "calcolo combinatorio domina sull'overhead. Per massimizzare l'effetto:\n"
                     "l_pixel piccolo (piu' pixel -> piu' hit distinti) e N alto.\n";

        viz.Run();
        return 0;
}

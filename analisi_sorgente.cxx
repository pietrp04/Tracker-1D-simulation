// Source calibration + noise scan.
//
// Calibration: for N_CALIB_EVENTS runs (default 100k), find the tracks of
// each event and histogram the q of two variants built from the SAME events:
//   - "clean":  clone removal applied (CpuCleanedQHistogramEngine, default
//     clean_clones=true) -- the real single-event chain.
//   - "raw":    no clone removal at all (same engine, clean_clones=false) --
//     the raw combinatorial tracks straight out of the finder.
// Both histograms get the same robust peak fit and are drawn side by side,
// so the effect of clone removal on the calibration peak/background is
// directly visible. Both are also drawn a second time WITHOUT the fit curve
// overlaid, to look at the raw histogram shape on its own.
//
// Noise scan: 0% -> 10%, mu_q/sigma_q from the CLEAN calibration are used to
// select tracks; each step generates new events, removes clones, applies the
// calibrated q selection, and checks against MC truth with the 3-sigma
// Mahalanobis ellipse (truth.cxx).
// Graphs: efficiency vs noise, false positives vs noise.

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

// Generate a single event: angles -> theoretical y -> acceptance -> pixel ->
// noise -> read-back positions.
static EventHits GenerateOneEvent(Source& sorgente, Detector& detector, float prob_rumore) {
        std::vector<float> angoli = sorgente.angles();
        std::vector<std::vector<float>> y_teoriche = sorgente.y_th(angoli);
        std::vector<std::vector<float>> y_hit_true = sorgente.Check_hits(y_teoriche);

        EventHits pixel_positions(N_layers);
        for (size_t i = 0; i < N_layers; i++) {
                std::vector<uint32_t> pixels = detector.pos2pix(y_hit_true[i]);
                sorgente.MakeNoisePhysical(prob_rumore, pixels);
                pixel_positions[i] = detector.pix2pos(pixels);
        }
        return pixel_positions;
}

// Generate n_events events in parallel. IMPORTANT: Source holds a random
// generator (m_gen) as an unlocked member -- sharing ONE Source across
// OpenMP threads would mean calling std::uniform_real_distribution /
// std::bernoulli_distribution on the same std::mt19937 from different
// threads concurrently: undefined behaviour (corrupted data, typically
// duplicated events or intermittent crashes, not a compile error). Here each
// thread builds its OWN Source (self-seeded from std::random_device in the
// constructor, so different threads get independent sequences) and only
// writes to its own batch[] indices -- no critical section needed since each
// index is written by exactly one thread.
static void GenerateBatchParallel(EventBatch& batch, size_t n_events,
                                   float mean, float std_dev, float prob_rumore) {
        batch.resize(n_events);

        #pragma omp parallel
        {
                Source sorgente_locale(mean, std_dev);
                Detector detector_locale;

                #pragma omp for schedule(static)
                for (long k = 0; k < (long)n_events; k++) {
                        batch[(size_t)k] = GenerateOneEvent(sorgente_locale, detector_locale, prob_rumore);
                }
        }
}

int main(int argc, char* argv[]) {

        // ------------------ parameters ------------------
        uint64_t N_CALIB_EVENTS = 100000;      // calibration run count
        int      N_SCAN_EVENTS  = 300;         // events per noise level
        const size_t BATCH_SIZE = 10000;       // events per batch (caps RAM use)
        const float mean = 0.0f;
        const float std_dev = 0.267f;
        const float prob_rumore_calib = 0.05f; // noise used during calibration

        const int   N_BINS = 200;
        const float Q_LO = q_min - 0.05f;
        const float Q_HI = q_max + 0.05f;

        if (argc > 1) N_CALIB_EVENTS = std::strtoull(argv[1], nullptr, 10);
        if (argc > 2) N_SCAN_EVENTS  = std::atoi(argv[2]);

        Visualizer viz(argc, argv);

        // =====================================================================
        // CALIBRATION: 100k runs, q of clone-cleaned tracks AND q of raw
        // (uncleaned) tracks, both accumulated from the same generated events.
        // =====================================================================
        std::cout << "=== CALIBRAZIONE (" << N_CALIB_EVENTS << " run) ===" << std::endl;
        std::cout << "Distribuzione angoli: "
                  << (USE_GAUSSIAN_ANGLES ? "GAUSSIANA N(mean, std)" : "UNIFORME [mean +/- std*sqrt(3)]")
                  << "  (mean=" << mean << ", std=" << std_dev << ")\n";

        CpuCleanedQHistogramEngine calib_engine(N_BINS, Q_LO, Q_HI);                          // clean_clones = true (default)
        CpuCleanedQHistogramEngine calib_engine_raw(N_BINS, Q_LO, Q_HI, /*clean_clones=*/false);

        EventBatch batch;

        auto t0 = std::chrono::high_resolution_clock::now();
        uint64_t generati = 0;
        double ms_generazione = 0.0, ms_tracking = 0.0, ms_tracking_raw = 0.0;

        while (generati < N_CALIB_EVENTS) {
                const size_t this_batch = (size_t)std::min<uint64_t>(BATCH_SIZE, N_CALIB_EVENTS - generati);

                auto tg0 = std::chrono::high_resolution_clock::now();
                GenerateBatchParallel(batch, this_batch, mean, std_dev, prob_rumore_calib);
                auto tg1 = std::chrono::high_resolution_clock::now();
                ms_generazione += std::chrono::duration<double, std::milli>(tg1 - tg0).count();

                // Same batch fed to both engines: no extra event generation needed
                // to also get the raw (non-clone-cleaned) histogram.
                calib_engine.ProcessBatch(batch);
                ms_tracking += calib_engine.last_batch_ms();

                calib_engine_raw.ProcessBatch(batch);
                ms_tracking_raw += calib_engine_raw.last_batch_ms();

                generati += this_batch;
                std::cout << "\r  " << generati << " / " << N_CALIB_EVENTS << " run processate" << std::flush;
        }
        std::cout << std::endl;

        auto t1 = std::chrono::high_resolution_clock::now();
        double s_tot = std::chrono::duration<double>(t1 - t0).count();
        std::cout << "  Tempo totale: " << std::fixed << std::setprecision(1) << s_tot << " s ("
                  << (double)N_CALIB_EVENTS / s_tot << " run/s; generazione "
                  << ms_generazione / 1000.0 << " s, tracking+pulizia cloni "
                  << ms_tracking / 1000.0 << " s, tracking senza pulizia cloni "
                  << ms_tracking_raw / 1000.0 << " s)" << std::endl;

        QHistogram h_calib     = calib_engine.GetHistogram();
        QHistogram h_calib_raw = calib_engine_raw.GetHistogram();

        // Gaussian fit + plot (single-histogram Visualizer overload), once for
        // the clean histogram and once for the raw one. mu_q/sigma_q used below
        // for the noise scan come from the CLEAN fit, as before.
        std::pair<float, float> fit_res = viz.DrawSourceSearchAndFit(
                h_calib, "", "Calibrazione: q delle tracce pulite dai cloni; q ricostruita (m); Conteggi");
        const float mu_q    = fit_res.first;
        const float sigma_q = fit_res.second;

        std::pair<float, float> fit_res_raw = viz.DrawSourceSearchAndFit(
                h_calib_raw, "_raw", "Calibrazione: q delle tracce SENZA pulizia cloni; q ricostruita (m); Conteggi");
        const float mu_q_raw    = fit_res_raw.first;
        const float sigma_q_raw = fit_res_raw.second;

        // Same two histograms again, WITHOUT the gaussian curve overlaid: just
        // the raw shape of the calibration peak, useful to look at before/next
        // to the fitted version (e.g. to show the comb-like quantisation
        // structure without a curve drawn over it).
        viz.DrawSourceSearchAndFit(
                h_calib, "_nofit", "Calibrazione: q delle tracce pulite dai cloni, senza fit; q ricostruita (m); Conteggi",
                /*draw_fit_curve=*/false);
        viz.DrawSourceSearchAndFit(
                h_calib_raw, "_raw_nofit", "Calibrazione: q delle tracce SENZA pulizia cloni, senza fit; q ricostruita (m); Conteggi",
                /*draw_fit_curve=*/false);

        std::cout << "\n  -> Sorgente stimata (cloni puliti) : y = " << mu_q     << " m, sigma_q = " << sigma_q
                  << "\n  -> Sorgente stimata (senza pulizia): y = " << mu_q_raw << " m, sigma_q = " << sigma_q_raw
                  << "\n     (vero: " << sourceY << " m)"
                  << "\n  -> Selezione tracce da qui in poi: q in ["
                  << mu_q - 3.0f * sigma_q << ", " << mu_q + 3.0f * sigma_q << "]\n" << std::endl;

        // =====================================================================
        // NOISE SCAN: efficiency and false positives vs noise (0% -> 10%),
        // after clone removal + calibrated q selection, MC matching with the
        // Mahalanobis ellipse (see truth.cxx).
        // =====================================================================
        std::cout << "=== SCANSIONE DEL RUMORE ===" << std::endl;
        viz.RunNoiseScan(mean, std_dev, mu_q, sigma_q,
                          N_SCAN_EVENTS,
                          /*noise_min=*/0.0f, /*noise_max=*/0.10f,
                          /*n_noise_steps=*/11);

        viz.Run();
        return 0;
}

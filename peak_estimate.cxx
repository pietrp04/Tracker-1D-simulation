#include "track_finder.h"
#include <cmath>
#include <algorithm>

PeakEstimate RobustPeakEstimate(const QHistogram& h, float initial_sigma_guess,
                                 int n_iter, float n_sigma_clip) {
        PeakEstimate result;

        if (h.n_bins <= 0 || h.total_tracks_found == 0) {
                return result; // valid stays false: empty histogram, nothing to estimate
        }

        // ---------------------------------------------------------------
        // STEP 0 - adaptive bin aggregation.
        //
        // With discretised inputs (hit positions quantised to multiples of
        // l_pixel) propagated through the linear fit, the possible q values
        // are not a continuum but a DISCRETE set: the histogram near the peak
        // is "comb-shaped" (bins with very high counts separated by bins at
        // EXACTLY zero -- a systematic structure, not noise). A FWHM computed
        // on the raw histogram stops at the first empty gap next to the
        // tallest tooth of the comb, giving a falsely narrow sigma (as wide
        // as ONE tooth, not the true spread across the teeth that make up the
        // physical peak).
        //
        // Two indirect criteria for deciding how much to aggregate before
        // trusting the FWHM were tried first -- "the tallest bin has enough
        // counts" and "a minimum fraction of nearby bins is non-empty" -- but
        // neither proved reliable: even with 80% of local bins non-empty the
        // FWHM stayed visibly too narrow (a single cluster of nearby teeth
        // can already satisfy those criteria without representing the whole
        // spread of the peak).
        //
        // The direct, robust criterion is: keep aggregating until the
        // measured FWHM reaches a physically credible size, compared against
        // initial_sigma_guess (ideally the theoretical sigma expected from
        // geometry, known a priori). If the FWHM stays much smaller than the
        // detector physics allows, it is almost certainly a quantisation
        // artefact, not a real measurement -- so aggregation continues.
        // ---------------------------------------------------------------
        const uint64_t MIN_PEAK_COUNTS = 100;
        const double MIN_CREDIBLE_SIGMA = 0.5 * (double)initial_sigma_guess;

        int group = 1;
        std::vector<uint64_t> agg;
        int n_agg = h.n_bins;

        int max_bin = 0;
        uint64_t max_count = 0;
        double fwhm = 0.0;

        auto agg_center = [&](int i) -> double {
                const int first = i * group;
                const int last  = std::min(h.n_bins, first + group) - 1;
                return (h.bin_center(first) + h.bin_center(last)) / 2.0;
        };

        while (true) {
                n_agg = (h.n_bins + group - 1) / group;
                agg.assign(n_agg, 0);
                for (int i = 0; i < h.n_bins; i++) {
                        agg[i / group] += h.counts[i];
                }

                max_bin = 0;
                max_count = 0;
                for (int i = 0; i < n_agg; i++) {
                        if (agg[i] > max_count) {
                                max_count = agg[i];
                                max_bin = i;
                        }
                }
                if (max_count == 0) {
                        return result; // all bins empty
                }

                std::vector<uint64_t> sorted_counts(agg.begin(), agg.end());
                std::sort(sorted_counts.begin(), sorted_counts.end());
                double bkg_level = (double)sorted_counts[sorted_counts.size() / 2];

                double half_max = bkg_level + ((double)max_count - bkg_level) / 2.0;
                if (half_max <= bkg_level) half_max = bkg_level + 1.0;

                int right_bin = max_bin;
                while (right_bin < n_agg - 1 && (double)agg[right_bin] > half_max) right_bin++;
                int left_bin = max_bin;
                while (left_bin > 0 && (double)agg[left_bin] > half_max) left_bin--;

                const double agg_bin_width = (double)h.bin_width() * group;
                fwhm = agg_center(right_bin) - agg_center(left_bin);
                if (fwhm < agg_bin_width * 2.0) fwhm = agg_bin_width * 2.0;

                const double sigma_candidate = fwhm / 2.35482;

                const bool statistica_ok = (max_count >= MIN_PEAK_COUNTS);
                const bool larghezza_credibile = (sigma_candidate >= MIN_CREDIBLE_SIGMA);

                if ((statistica_ok && larghezza_credibile) || group >= h.n_bins) break;
                group *= 2;
        }

        double mu = agg_center(max_bin);

        // sigma is NEVER recomputed from the data variance: it is fixed here,
        // once, to the value derived from the FWHM at the chosen aggregation
        // above. A refinement that recomputes it from the windowed weighted
        // variance is exactly the mechanism that diverged in an earlier
        // attempt: the variance is QUADRATICALLY sensitive to points far from
        // the center, so even a small background contamination inside the
        // window inflated it; a larger sigma widened the next iteration's
        // window, which absorbed even more background, and so on -- a
        // positive feedback loop (sigma~0.02 -> sigma~0.64 in 8 iterations,
        // observed). The FWHM is a LOCAL density measure: it is the right
        // quantity to keep fixed.
        const double sigma_raw = fwhm / 2.35482;
        const double sigma = (sigma_raw > 0.0 && std::isfinite(sigma_raw))
            ? sigma_raw : (double)initial_sigma_guess;

        // ---------------------------------------------------------------
        // STEP 2 - the remaining iterations only RECENTER mu: the window has
        // fixed width (n_sigma_clip * sigma, with sigma fixed above) and
        // slides to follow the updated mu estimate. It cannot diverge because
        // the window width never depends on its own result. This works again
        // on the ORIGINAL (non-aggregated) histogram for the final precision
        // of mu, since the window is now already well localised.
        // ---------------------------------------------------------------
        uint64_t entries_last_window = 0;

        for (int iter = 0; iter < n_iter; iter++) {
                const double lo = mu - (double)n_sigma_clip * sigma;
                const double hi = mu + (double)n_sigma_clip * sigma;

                double w = 0.0, wx = 0.0;
                uint64_t entries = 0;

                for (int i = 0; i < h.n_bins; i++) {
                        const double x = h.bin_center(i);
                        if (x < lo || x > hi) continue;
                        const double c = (double)h.counts[i];
                        w += c;
                        wx += c * x;
                        entries += h.counts[i];
                }

                if (w <= 0.0) break; // window emptied out: stop at the previous estimate

                const double new_mu = wx / w;
                entries_last_window = entries;

                const bool converged = std::abs(new_mu - mu) < (double)h.bin_width() * 0.01;
                mu = new_mu;
                if (converged) break;
        }

        result.mu = (float)mu;
        result.sigma = (float)sigma;
        result.entries_in_window = entries_last_window;
        result.valid = true;
        return result;
}

#include "truth.h"
#include "config_new.h"
#include <cmath>

DefinizioneTagli CalcolaTolleranzeTeoriche() {
    float Sx_teorico = 0.0f;
    float Sxx_teorico = 0.0f;

    for (size_t L = 0; L < N_layers; L++) {
        float x_L = d_s_l + L * dist_interlayers;
        Sx_teorico  += x_L;
        Sxx_teorico += x_L * x_L;
    }

    float N_f = static_cast<float>(N_layers);
    float Delta_teorico = (N_f * Sxx_teorico) - (Sx_teorico * Sx_teorico);

    DefinizioneTagli tagli;
    tagli.sigma_m = std::sqrt((N_f * sigma * sigma) / Delta_teorico);
    tagli.sigma_q = std::sqrt((Sxx_teorico * sigma * sigma) / Delta_teorico);
    
    tagli.delta_m_cut = 3.0f * tagli.sigma_m;
    tagli.delta_q_cut = 3.0f * tagli.sigma_q;

    return tagli;
}

// ---------------------------------------------------------------------------
// Track-to-MC matching with the physically correct discriminant.
//
// The least-squares fit over N points at positions x_L with error sigma does
// not produce independent m and q: the covariance matrix is
//
//     Var(m)   =  N   * sigma^2 / Delta
//     Var(q)   =  Sxx * sigma^2 / Delta
//     Cov(m,q) = -Sx  * sigma^2 / Delta     (ANTI-correlation: with all
//                                            x_L > 0, raising the slope
//                                            lowers the intercept and vice
//                                            versa)
//
// A "box" cut |dm|<3sigma_m AND |dq|<3sigma_q ignores this anti-correlation:
// it accepts box corners that lie much further than 3 "effective" sigma
// along the correlated direction, with a larger area than the true
// confidence ellipse. The correct discriminant is the Mahalanobis distance
// in (m,q) space:
//
//     D^2 = (dm, dq) * C^-1 * (dm, dq)^T
//         = [ Sxx*dm^2 + 2*Sx*dm*dq + N*dq^2 ] / sigma^2
//
// (C^-1 has a simple closed form: C^-1 = (1/sigma^2) * [[Sxx, Sx],[Sx, N]],
// verified by direct multiplication.) A match requires D^2 <= 9, i.e. the
// "3 sigma" contour of the ellipse.
//
// Second physical correction: each track is matched to the MC particle with
// the MINIMUM D^2 among those below threshold, not the FIRST one that passes
// the cut in vector order (with nearby tracks, first-match can attribute a
// track to the wrong particle and bias the efficiency counts).
// ---------------------------------------------------------------------------
AnalisiTracce ValutaTracce(const std::vector<Track>& tracce_ricostruite, 
                           const std::vector<float>& mc_angles, 
                           float source_y_true) 
{
    AnalisiTracce risultati;

    // Geometric covariance terms (fixed, depend only on the tracker layout)
    float Sx = 0.0f, Sxx = 0.0f;
    for (size_t L = 0; L < N_layers; L++) {
        float x_L = d_s_l + L * dist_interlayers;
        Sx  += x_L;
        Sxx += x_L * x_L;
    }
    const float N_f = static_cast<float>(N_layers);
    const float inv_sigma2 = 1.0f / (sigma * sigma);
    const float D2_cut = 9.0f; // 3-sigma contour in the (m,q) ellipse

    // Geometric acceptance: an MC particle is reconstructible only if its
    // line crosses ALL layers inside [0, l_tracker] (same y>0 && y<1
    // criterion as Source::Check_hits: outside that it produces no hit and
    // no algorithm can find it). Particles outside acceptance must be
    // EXCLUDED from the efficiency denominator, otherwise what is measured
    // is the detector's acceptance (~70% with these parameters:
    // |tan(theta)| < 0.5/1.5 over a uniform range of +/-0.46 rad) instead of
    // the algorithm's quality, and efficiency saturates well below 100% even
    // with zero noise and a perfect reconstruction.
    // mc_totali + geometric acceptance below: only mc_in_acceptance particles
    // feed mc_trovati/mc_persi, but mc_totali counts everyone generated.
    risultati.mc_totali = (int)mc_angles.size();

    std::vector<bool> mc_in_acceptance(mc_angles.size(), true);
    for (size_t i = 0; i < mc_angles.size(); i++) {
        const float m_true = std::tan(mc_angles[i]);
        for (size_t L = 0; L < N_layers; L++) {
            const float x_L = d_s_l + L * dist_interlayers;
            const float y_L = source_y_true + m_true * x_L;
            if (!(y_L > 0.0f && y_L < l_tracker)) { mc_in_acceptance[i] = false; break; }
        }
    }

    std::vector<bool> mc_matched(mc_angles.size(), false);

    for (const auto& traccia : tracce_ricostruite) {
        float best_D2 = 1e30f;
        int best_mc = -1;

        for (size_t i = 0; i < mc_angles.size(); i++) {
            const float m_true = std::tan(mc_angles[i]);
            const float q_true = source_y_true;

            const float dm = traccia.m - m_true;
            const float dq = traccia.q - q_true;

            // Squared Mahalanobis distance using the fit's full covariance
            const float D2 = (Sxx * dm * dm + 2.0f * Sx * dm * dq + N_f * dq * dq) * inv_sigma2;

            if (D2 < best_D2) {
                best_D2 = D2;
                best_mc = (int)i;
            }
        }

        if (best_mc >= 0 && best_D2 <= D2_cut) {
            risultati.tracce_buone++;
            mc_matched[best_mc] = true;
        } else {
            risultati.tracce_ghost++;
        }
    }

    // mc_trovati + mc_persi = particles IN ACCEPTANCE: this is the correct
    // denominator for the algorithm's efficiency (efficiency should be
    // computed as mc_trovati / (mc_trovati + mc_persi), NOT over the
    // generated N_tracks).
    for (size_t i = 0; i < mc_angles.size(); i++) {
        if (!mc_in_acceptance[i]) continue;
        if (mc_matched[i]) risultati.mc_trovati++;
        else risultati.mc_persi++;
    }

    return risultati;
}

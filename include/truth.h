#ifndef TRUTH_H
#define TRUTH_H

#include <vector>
#include "track_finder.h" // needed for the Track struct

struct DefinizioneTagli {
    	float sigma_m;
    	float sigma_q;
    	float delta_m_cut;
    	float delta_q_cut;
};


struct AnalisiTracce {
    	// mc_trovati/mc_persi only count MC particles in geometric acceptance
    	// (crossing all layers inside [0, l_tracker]): particles outside
    	// acceptance produce no hit and cannot be reconstructed by any
    	// algorithm. Algorithm efficiency = mc_trovati / (mc_trovati + mc_persi).
    	// mc_totali is the total number of MC particles generated for this
    	// event, WITHOUT the acceptance cut: it lets callers also compute the
    	// alternative efficiency mc_trovati / mc_totali, which counts every
    	// particle that missed even a single layer as a loss instead of
    	// excluding it from the denominator.
    	int mc_trovati = 0;     // MC particles in acceptance, reconstructed
    	int mc_persi = 0;       // MC particles in acceptance, NOT reconstructed
    	int tracce_buone = 0;   // reconstructed tracks matched to an MC particle (purity)
    	int tracce_ghost = 0;   // reconstructed tracks with no MC match (false positives)
    	int mc_totali = 0;      // all MC particles generated, acceptance or not
};

// Function declarations
DefinizioneTagli CalcolaTolleranzeTeoriche();

AnalisiTracce ValutaTracce(const std::vector<Track>& tracce_ricostruite, 
                           const std::vector<float>& mc_angles, 
                           float source_y_true);

#endif // TRUTH_H


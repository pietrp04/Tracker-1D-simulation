#include "source.h"

Source::Source()
	: m_mean(0.5), m_std(0.5 / std::sqrt(12)), m_gen(std::random_device{}())
{

}

Source::Source(float mean, float std)
        : m_mean(mean), m_std(std), m_gen(std::random_device{}())
{

}

// noise probability applies per-pixel over [0,1]
void Source::MakeNoisePhysical(float prob_rumore, std::vector<uint32_t>& pixels) {
        std::bernoulli_distribution probabilita_accensione(prob_rumore);

        for (uint32_t& p : pixels) {
                if (p == 0 && probabilita_accensione(m_gen)) {
                        p = 1;
                }
        }
}

// Dispatcher used by every event generator: picks the distribution set by
// USE_GAUSSIAN_ANGLES in config_new.h. Keeping the choice here means the
// generators in analisi_sorgente.cxx / studio_prestazioni.cxx /
// visualizza_apparato.cxx / Visualizer::RunNoiseScan never need to know
// which distribution is in use.
std::vector<float> Source::angles()
{
        return angles(static_cast<size_t>(N_tracks));
}

std::vector<float> Source::angles(size_t n)
{
        return USE_GAUSSIAN_ANGLES ? angle_gaussian(n) : angle_uniform(n);
}

std::vector<float> Source::angle_uniform()
{
        return angle_uniform(static_cast<size_t>(N_tracks));
}

std::vector<float> Source::angle_uniform(size_t n)
{
        float angle_max = m_mean + m_std * std::sqrt(3.0f); //calculates the range in which the angle has to be randomly generated
        float angle_min = m_mean - m_std * std::sqrt(3.0f);

        std::uniform_real_distribution<float> distro(angle_min, angle_max);
        std::vector<float> angles;
        angles.reserve(n);

        for (size_t i = 0; i < n; i++) {
                angles.push_back(distro(m_gen)); //m_gen is a  built-in variable of the class, so that the code does not waste time on reallocating every time the random hardwere device
        }

        return angles;
}

std::vector<float> Source::angle_gaussian()
{
        return angle_gaussian(static_cast<size_t>(N_tracks));
}

// Same mean and standard deviation as angle_uniform (whose range is built so
// that its std is exactly m_std), so the two differ only in SHAPE. Note the
// gaussian has unbounded tails, so it can emit angles beyond the uniform's
// hard limit of mean +/- m_std*sqrt(3): those simply miss the tracker and are
// dropped by Check_hits, which is the physically correct behaviour.
std::vector<float> Source::angle_gaussian(size_t n)
{
        std::normal_distribution<float> gaussian_dist(m_mean, m_std);
        std::vector<float> angles;
        angles.reserve(n);

        for (size_t i = 0; i < n; i++) {
                angles.push_back(gaussian_dist(m_gen));
        }

        return angles;
}

std::vector<std::vector<float>> Source::y_th(const std::vector<float>& angles)
{
	std::vector<std::vector<float>> y_th(N_layers);
	for(const float& theta : angles) {
		float tan_theta = std::tan(theta); //assume the experiment is in a 2D geomety, the source is at x = 0, (you can decide the y in the constructor), and at distance d_s_l there is the first layer
		for(size_t layer = 0; layer < N_layers; layer++) {
			float delta_x = d_s_l + (layer * dist_interlayers); // extra x distance for this layer
			float y_calc = sourceY + delta_x * tan_theta;
			y_th[layer].push_back(y_calc);
		}
	}
	return y_th;
}

std::vector<std::vector<float>> Source::Check_hits(const std::vector<std::vector<float>>& y_th)
{
        std::vector<std::vector<float>> layer_hits(N_layers);
        for(size_t i = 0; i < N_layers; i++) {
                for(float y : y_th[i]) {
                        if(y > 0 && y < 1) { //checks if the theoretical y is in the acceptance range
                                layer_hits[i].push_back(y);
                        }
                }
        }
        return layer_hits;
}



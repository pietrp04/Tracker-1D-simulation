#include "detector.h"

//Inizialization
Detector::Detector()
	: m_name("Tracker"), m_y(0.0), m_nPix(0)
{

}

void Detector::setName(const std::string& name)
{
	m_name = name;
}

std::vector<uint32_t> Detector::pos2pix(float y)
{
	m_y = y;
	m_nPix = l_tracker/l_pixel;
	std::vector<uint32_t> pixels(m_nPix, 0); //m_nPix is the total number of pixels
	m_pixels = pixels;
	float passo = l_pixel;

	for (size_t i = 0; i < m_nPix; i++) {
		if ( (i*passo <= m_y) && (m_y < (i+1)*passo) ) {
			m_pixels[i] = 1;
		} else {
			m_pixels[i] = 0;
		}
	}

	return m_pixels;
}

//overloading
std::vector<uint32_t> Detector::pos2pix(const std::vector<float>& y_hits)
{
        m_nPix = l_tracker / l_pixel;
        std::vector<uint32_t> pixels(m_nPix, 0);
        float passo = l_pixel;
        for (const float& y : y_hits) {
        	int hit_index = std::floor(y / passo);
        	//check if everything went well
                if (hit_index >= 0 && hit_index < static_cast<int>(m_nPix)) {
                        pixels[hit_index] = 1;
                }
        }

        m_pixels = pixels;
        return m_pixels;
}

std::vector<float> Detector::pix2pos(const std::vector<uint32_t>& pixels)
{
        std::vector<float> y_hits;
        float passo = l_pixel;

	//WARNING: pix2pos does not work with packed pixel, this function is just the inverse of the pos2pix functions, in facts it expects aa vector where every element is just a pixel
	//same as makeNoise() function
        for (size_t i = 0; i < pixels.size(); i++) {
                if (pixels[i] != 0) {
                        y_hits.push_back(passo * (static_cast<float>(i) + 0.5f));
                }
        }
        return y_hits;
}

#pragma once

#include <cmath>
#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include "config_new.h"

class Detector
{
        public:
        Detector();
        
        float y() const { return m_y; }
        uint32_t nPix() const { return m_nPix; }
        const std::string& Name() const { return m_name; }
        void setName(const std::string& name);
	
	std::vector<uint32_t> pos2pix(float y);
	std::vector<uint32_t> pos2pix(const std::vector<float>& y_hits);

	std::vector<float> pix2pos(const std::vector<uint32_t>& pixels);
        
        private:
        std::string m_name;
        float m_y;
        uint32_t m_nPix;
        std::vector<uint32_t> m_pixels;
};

#pragma once
#include <cmath>
#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include "config_new.h"
#include <stdlib.h>
#include <stdint.h>
#include <fstream>
#include <cstdint>

class Storage {
	public:
	Storage(const std::vector<std::vector<uint32_t>>& pixel);
	std::vector<std::vector<uint32_t>> Read(const char* filename);
	void Write(const char* filename , const std::vector<std::vector<uint32_t>>& pixel, std::vector<float> MC_angles);
	std::vector<float> ReadMC(const char* filename);
	//Unpacks 32 bit words that are given from the read() function in a vector where every element is a pixel, that is what makeNoise() and pix2pos() expect
	//Padding bits are discarded
	std::vector<std::vector<uint32_t>> UnpackAllLayers(const std::vector<std::vector<uint32_t>>& packed);
	private:
	std::vector<std::vector<uint32_t>> m_pixel;
};

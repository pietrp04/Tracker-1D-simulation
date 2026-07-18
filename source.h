#pragma once

#include <cmath>
#include <vector>
#include <string>
#include <iostream>
#include <cstdint>
#include "config_new.h"
#include <random>
#include <fstream>

class Source
{
	public:
	Source(); //constructor
	Source(float mean, float std);

	// --- angle generation ---------------------------------------------
	// angles()/angles(n) are the ones event generators should call: they
	// dispatch to the uniform or gaussian version according to the
	// USE_GAUSSIAN_ANGLES flag in config_new.h, so switching distribution
	// is a one-line config change instead of an edit in every generator.
	// The explicit angle_uniform/angle_gaussian below stay available for
	// code that must pin one distribution regardless of the flag.
	std::vector<float> angles();          // uses N_tracks
	std::vector<float> angles(size_t n);  // explicit track count

	std::vector<float> angle_uniform(); //angles generated with a uniform distribution (you have to define mean and standard deviation in the constructor)
	// Overload with an explicit track count: used by studies where the
	// number of particles per event varies at runtime (accuracy/timing vs
	// track count), where the N_tracks constant from config_new.h is not
	// enough. The no-argument version is unchanged (uses N_tracks).
	std::vector<float> angle_uniform(size_t n);
	std::vector<float> angle_gaussian(); //same but with gaussian distribution
	std::vector<float> angle_gaussian(size_t n); //same, with explicit track count

	void MakeNoisePhysical(float prob_rumore, std::vector<uint32_t>& pixels);

	std::vector<std::vector<float>> y_th(const std::vector<float>& angles); //it gives back the theoretical y on each layer that has to be converted in pixel hits
	std::vector<std::vector<float>> Check_hits(const std::vector<std::vector<float>>& y_th); //checks whether the theoretical y are in the acceptance range [0, 1]

	private:
	float m_mean;
	float m_std;
	std::mt19937 m_gen;
};

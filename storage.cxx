#include "storage.h"
#include <cstdint>
//aggiungi rumore
//aggiungi MC truth

Storage::Storage(const std::vector<std::vector<uint32_t>>& pixel)
{
	m_pixel = pixel;
}

void Storage::Write(const char* filename, const std::vector<std::vector<uint32_t>>& pixel, std::vector<float> MC_angles)
{
        std::cout << "Sto scrivendo in binario..." << std::endl;
        std::ofstream o(filename, std::ios::binary);
        
        o.write(reinterpret_cast<const char*>(&full_event_cw), sizeof(full_event_cw)); //header of the simulation

        //metadata
        o.write(reinterpret_cast<const char*>(&N_layers), sizeof(N_layers));
        o.write(reinterpret_cast<const char*>(&l_tracker), sizeof(l_tracker));
        o.write(reinterpret_cast<const char*>(&l_pixel), sizeof(l_pixel));
        o.write(reinterpret_cast<const char*>(&dist_interlayers), sizeof(dist_interlayers)); //da modificare
        o.write(reinterpret_cast<const char*>(&d_s_l), sizeof(d_s_l));
        o.write(reinterpret_cast<const char*>(&source_x), sizeof(source_x));
        o.write(reinterpret_cast<const char*>(&chi2_threshold), sizeof(chi2_threshold));
        o.write(reinterpret_cast<const char*>(&N_tracks), sizeof(N_tracks));
        o.write(reinterpret_cast<const char*>(&versione), sizeof(versione));

	//data
        std::vector<uint32_t> binary_pixel; //vector of 32 bit element that will be written on the file
        uint32_t current_word = 0; //compress pixels in 32 bit words
        int bit_index = 0;
        
        for (size_t i = 0; i < N_layers; i++) {
        	if (bit_index > 0) {
                	binary_pixel.push_back(current_word);
			current_word = 0;
                	bit_index = 0;
        	}

        	uint32_t checkword_layer = 0xAAA00000;

        	checkword_layer = checkword_layer + static_cast<uint32_t>(i); //layer's checkword
        	binary_pixel.push_back(checkword_layer);
                for (size_t n = 0; n < pixel[i].size(); n++) {
                        if (pixel[i][n] == 1) {
                                current_word |= (1U << bit_index);
                        }

                        bit_index++;

                        if (bit_index == 32) { //stop word
                                binary_pixel.push_back(current_word);
                                current_word = 0;
                                bit_index = 0;
                        }
                }
	}

	if (bit_index > 0) {
                binary_pixel.push_back(current_word);
        }

        o.write(reinterpret_cast<const char*>(binary_pixel.data()), binary_pixel.size() * sizeof(uint32_t)); //end of real data
        //MC checkword
	o.write(reinterpret_cast<const char*>(&MC_checkword), sizeof(uint32_t));
        o.write(reinterpret_cast<const char*>(MC_angles.data()), MC_angles.size() * sizeof(float)); //end MC data
        std::cout << "Ho scritto il file binario." << std::endl;
}


std::vector<std::vector<uint32_t>> Storage::Read(const char* filename) //WARNING, THIS FUNCTION DOES NOT RETURN MC_DATA
{
        std::vector<uint32_t> data_read;
        std::ifstream in(filename, std::ios::binary | std::ios::ate); //at the end of the file to be read
        if (!in) { //check if all good
                std::cerr << "Errore apertura file!" << std::endl;
                return std::vector<std::vector<uint32_t>> ();
        }

        std::streamsize size = in.tellg(); //tells the position in the file
        std::streamsize meta_size = sizeof(full_event_cw) + sizeof(N_layers) + sizeof(l_tracker) + sizeof(l_pixel) + sizeof(dist_interlayers) + sizeof(versione) + sizeof(N_tracks) + sizeof(chi2_threshold) +sizeof(source_x) + sizeof(d_s_l);
        std::streamsize data_size = size - meta_size;
        in.seekg(meta_size, std::ios::beg); //pointer after metadata
        data_read.resize(data_size / sizeof(uint32_t)); //data_size is in byte format
        in.read(reinterpret_cast<char*>(data_read.data()), data_size);
	std::vector<std::vector<uint32_t>> output(N_layers);
	int current_layer = -1;

        for(size_t i = 0; i < data_read.size(); i++) {
        	uint32_t word = data_read[i];
		if (word == MC_checkword) {
			break;
		}
        	if((word & 0xFFF00000) == 0xAAA00000) {
			current_layer = word & 0x000FFFFF; //extract the number of layers
        	}

        	if(current_layer >= 0 && current_layer < int(N_layers)) {
			output[current_layer].push_back(word);
        	}
	}

	//verify if the checkwords are correct

	for (size_t i = 0; i < N_layers; i++) {
		uint32_t expected_cw = 0xAAA00000 + static_cast<uint32_t>(i);
		if (!output[i].empty()) {
			if (output[i].front() == expected_cw) {
				output[i].erase(output[i].begin());
			} else {
				std::cerr << "ERRORE: Checkword mancante o errata nel layer " << i << std::endl;
			}
		}
	}

        return output;
}

std::vector<float> Storage::ReadMC(const char* filename)
{
        std::ifstream in(filename, std::ios::binary | std::ios::ate);
        if (!in) {
                std::cerr << "Errore apertura file per lettura MC!" << std::endl;
                return std::vector<float>();
        }

        std::streamsize total_size = in.tellg();
        std::streamsize meta_size = sizeof(full_event_cw) + sizeof(N_layers) + sizeof(l_tracker) + sizeof(l_pixel) + sizeof(dist_interlayers) + sizeof(versione) + sizeof(N_tracks) + sizeof(chi2_threshold) +sizeof(source_x) + sizeof(d_s_l);
        in.seekg(meta_size, std::ios::beg);

        uint32_t word = 0;
        bool mc_found = false;

        // Scan data until we find the MC_checkword
        while (in.read(reinterpret_cast<char*>(&word), sizeof(word))) {
                if (word == MC_checkword) {
                        mc_found = true;
                        break;
                }
        }

        std::vector<float> mc_angles;

        if (mc_found) {
                // tellg() now points exactly at the first byte after MC_checkword
                std::streamsize current_pos = in.tellg();
                std::streamsize mc_data_bytes = total_size - current_pos;

                if (mc_data_bytes > 0) {
                        mc_angles.resize(mc_data_bytes / sizeof(float));
                        in.read(reinterpret_cast<char*>(mc_angles.data()), mc_data_bytes);
                        std::cout << "Estratti " << mc_angles.size() << " angoli MC." << std::endl;
                } else {
                        std::cerr << "ATTENZIONE: Checkword trovata ma nessun dato MC successivo." << std::endl;
                }
        } else {
                std::cerr << "ERRORE: MC_checkword non trovata nel file!" << std::endl;
        }

        return mc_angles;
}

std::vector<std::vector<uint32_t>> Storage::UnpackAllLayers(const std::vector<std::vector<uint32_t>>& packed)
{
        const size_t n_pix = static_cast<size_t>(std::round(l_tracker / l_pixel));
        std::vector<std::vector<uint32_t>> unpacked(N_layers);

        for (size_t L = 0; L < N_layers && L < packed.size(); L++) {
                unpacked[L].assign(n_pix, 0);
                for (size_t p = 0; p < n_pix; p++) {
                        const size_t word_idx = p / 32;
                        const size_t bit_idx  = p % 32;
                        if (word_idx >= packed[L].size()) break; //if it is in the end of file, it breaks
                        if (packed[L][word_idx] & (1U << bit_idx)) {
                                unpacked[L][p] = 1;
                        }
                }
        }
        return unpacked;
}

#pragma once

#include<stdio.h>
#include<iostream>
#include<stdint.h>
#include<cstring>
#include<cmath>
#include<random>

const float sourceY = 0.5;
const uint32_t N_layers = 4; //Numero di layers, che assumo fissato in quanto dovrei cambiare il codice nella classe source.h
const float l_tracker = 1; //metri, lunghezza del tracker
const float l_pixel = 0.00025; //in metri
const float dist_interlayers = 0.3; //assumiamo che i layers siano sempre equidistanti tra di loro
const uint32_t full_event_cw = 0xcaccaaaa; //header dell'evento
const uint32_t versione = 1; //versione della simulazione
const float d_s_l = 1; //sidtanza tra sorgente e layer
const float source_x = 0; //per come è stata disegnata la simulazione la sorgente è sempre ad x = 0
const int max_tracce = 25000000; //massimo numero di tracce pensate per non far andare in overflow la GPU
const float rad_12 = 3.4641016f; //per il calcolo della sigma del chi quadro
const float chi2_threshold = 5;
const int N_tracks = 100; //number of real tracks
const float q_max = sourceY + 0.1f;
const float q_min = sourceY - 0.1f;
const float sigma = l_pixel / rad_12;
const uint32_t MC_checkword = 0xDEFAC0CA;


// ---------------------------------------------------------------------------
// Angular distribution of the particles emitted by the source. THIS IS THE
// ONLY LINE TO EDIT to switch every program (analisi_sorgente,
// studio_prestazioni, visualizza_apparato) between the two distributions --
// every event generator goes through Source::angles(), which reads this flag.
//
//   false -> uniform over [mean - std*sqrt(3), mean + std*sqrt(3)]
//   true  -> gaussian N(mean, std)
//
// The comparison is fair at equal sigma: the uniform range is built so that
// its standard deviation is exactly `std` (a uniform of half-width a has
// std = a/sqrt(3)), the same as the gaussian's. Switching therefore changes
// only the SHAPE of the distribution, not its width.
// ---------------------------------------------------------------------------
const bool USE_GAUSSIAN_ANGLES = true;

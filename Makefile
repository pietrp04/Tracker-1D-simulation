# Makefile - adjust CUDA_PATH if needed (e.g. /usr/local/cuda)
NVCC := nvcc
CXX  := g++

# RTX 4050 = Ada Lovelace -> compute capability 8.9
ARCH := sm_89

CUDA_PATH   ?= /usr/local/cuda
ROOT_CFLAGS := $(shell root-config --cflags)
ROOT_LIBS   := $(shell root-config --libs)

CXXFLAGS  := -O3 -march=native -std=c++17 -fopenmp $(ROOT_CFLAGS) -I$(CUDA_PATH)/include
NVCCFLAGS := -O3 -std=c++17 -arch=$(ARCH) --use_fast_math -Xcompiler -fopenmp
LIBS := $(ROOT_LIBS) -L$(CUDA_PATH)/lib64 -lcudart

COMMON_OBJS := track_finder_cpu.o peak_estimate.o truth.o visualizer.o source.o detector.o

.PHONY: all clean

all: analisi_sorgente studio_prestazioni visualizza_apparato

%.o: %.cxx track_finder.h truth.h visualizer.h config_new.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

track_finder_gpu.o: track_finder_gpu.cu track_finder.h config_new.h
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Calibration (100k runs, robust mu_q/sigma_q estimate without a Minuit fit,
# see peak_estimate.cxx) + 0-10% noise scan. Does not link CUDA: calibration
# with real clone removal runs on CPU/OpenMP only.
analisi_sorgente: analisi_sorgente.cxx $(COMMON_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(ROOT_LIBS)

# Final graphs: accuracy vs number of tracks (clones only / clones+q
# selection) and CPU vs GPU track-finding time (needs the GPU: links CUDA).
studio_prestazioni: studio_prestazioni.cxx $(COMMON_OBJS) track_finder_gpu.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIBS)

# Draws the apparatus (source + layers, geometry from config_new.h) with the
# tracks of one event in 4 progressively filtered variants (noise fixed at
# 5%). Does not use the Visualizer class (it owns its own TApplication), so
# visualizer.o is not needed here; does not link CUDA either.
visualizza_apparato: visualizza_apparato.cxx track_finder_cpu.o peak_estimate.o truth.o source.o detector.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(ROOT_LIBS)

clean:
	rm -f *.o analisi_sorgente studio_prestazioni visualizza_apparato

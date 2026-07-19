# Makefile for the include/src/apps layout.
# Object files go to build/, executables stay in the repo root.
# Adjust ARCH for your GPU and CUDA_PATH if CUDA is not in /usr/local/cuda.

NVCC := nvcc
CXX  := g++

# RTX 4050 = Ada Lovelace -> compute capability 8.9
ARCH := sm_89

CUDA_PATH   ?= /usr/local/cuda
ROOT_CFLAGS := $(shell root-config --cflags)
ROOT_LIBS   := $(shell root-config --libs)

# -Iinclude makes every  #include "config_new.h"  etc. resolve unchanged.
CXXFLAGS  := -O3 -march=native -std=c++17 -fopenmp -Iinclude $(ROOT_CFLAGS) -I$(CUDA_PATH)/include
NVCCFLAGS := -O3 -std=c++17 -arch=$(ARCH) --use_fast_math -Iinclude -Xcompiler -fopenmp
LIBS := $(ROOT_LIBS) -L$(CUDA_PATH)/lib64 -lcudart

BUILD := build

# All headers: used as a blanket prerequisite so editing any header rebuilds.
HEADERS := $(wildcard include/*.h)

# Library object files shared by the executables.
COMMON_OBJS := $(BUILD)/track_finder_cpu.o $(BUILD)/peak_estimate.o \
               $(BUILD)/truth.o $(BUILD)/visualizer.o \
               $(BUILD)/source.o $(BUILD)/detector.o

.PHONY: all clean
.PRECIOUS: $(BUILD)/%.o

all: analisi_sorgente studio_prestazioni visualizza_apparato

# --- compile library sources (src/) into build/ ---
$(BUILD)/%.o: src/%.cxx $(HEADERS) | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/%.o: src/%.cu $(HEADERS) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD):
	mkdir -p $(BUILD)

# --- link the executables (apps/) ---

# Calibration (robust mu_q/sigma_q estimate without a Minuit fit, see
# peak_estimate.cxx) + 0-10% noise scan. Does not link CUDA: calibration with
# real clone removal runs on CPU/OpenMP only.
analisi_sorgente: apps/analisi_sorgente.cxx $(COMMON_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(ROOT_LIBS)

# Accuracy vs number of tracks (clones only / clones+q selection) and CPU vs
# GPU track-finding time (needs the GPU: links CUDA).
studio_prestazioni: apps/studio_prestazioni.cxx $(COMMON_OBJS) $(BUILD)/track_finder_gpu.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIBS)

# Draws the apparatus (source + layers, geometry from config_new.h) with the
# tracks of one event in 4 progressively filtered variants (noise fixed at
# 5%). Does not use the Visualizer class (it owns its own TApplication), so
# visualizer.o is not needed here; does not link CUDA either.
visualizza_apparato: apps/visualizza_apparato.cxx \
                     $(BUILD)/track_finder_cpu.o $(BUILD)/peak_estimate.o \
                     $(BUILD)/truth.o $(BUILD)/source.o $(BUILD)/detector.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(ROOT_LIBS)

clean:
	rm -rf $(BUILD) analisi_sorgente studio_prestazioni visualizza_apparato

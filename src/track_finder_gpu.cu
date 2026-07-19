#include <stdio.h>
#include "config_new.h"
#include "track_finder.h"
#include "truth.h"
#include <math.h>
#include <cuda_runtime.h>
#include <chrono>
#include <algorithm>

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// =============================================================================
// Single-event GPU track-finding kernel: one thread per (layer0, last-layer)
// hit pair. Each thread snaps the nearest middle-layer hit to the seed line,
// least-squares fits the 3 points, and writes the track if chi2 < threshold.
// Used by GpuSingleEventFinder below. Track ordering within an event is not
// deterministic (atomic append), which is irrelevant after clone removal.
// =============================================================================

__global__ void TrackFinderKernel(
        const float* d_y_hits,
        const int* d_layer_offsets,
        const int* d_layer_sizes,
        Track* d_tracks_out,
        int* d_track_count,
        int track_capacity)
{
        __shared__ int shared_track_count;
        __shared__ int global_offset;

        if (threadIdx.x == 0 && threadIdx.y == 0) {
                shared_track_count = 0;
        }
        __syncthreads();

        int idx_L0 = blockIdx.x * blockDim.x + threadIdx.x;
        int idx_Llast = blockIdx.y * blockDim.y + threadIdx.y;

        int num_hits_L0 = d_layer_sizes[0];
        int num_hits_Llast = d_layer_sizes[N_layers - 1];

        bool is_valid_track = false;
        float m_fit = 0.0f, q_fit = 0.0f, chi2_globale = 0.0f;

        int best_indices[N_layers];

        if (idx_L0 < num_hits_L0 && idx_Llast < num_hits_Llast) {

                float best_y[N_layers];

                best_y[0] = d_y_hits[d_layer_offsets[0] + idx_L0];
                best_indices[0] = idx_L0;

                best_y[N_layers - 1] = d_y_hits[d_layer_offsets[N_layers - 1] + idx_Llast];
                best_indices[N_layers - 1] = idx_Llast;

                float x_start = d_s_l;
                float x_end   = d_s_l + (N_layers - 1) * dist_interlayers;
                float delta_x = x_end - x_start;

                float m_seed = (best_y[N_layers - 1] - best_y[0]) / delta_x;
                float q_seed = best_y[N_layers - 1] - (m_seed * x_end);

                if (q_seed >= q_min && q_seed <= q_max) {
                        bool missing_hit = false;

                        for (int L = 1; L < N_layers - 1; L++) {
                                float x_L = d_s_l + L * dist_interlayers;
                                float y_theory = (m_seed * x_L) + q_seed;

                                int offset = d_layer_offsets[L];
                                int size = d_layer_sizes[L];

                                if (size == 0) {
                                        missing_hit = true;
                                        break;
                                }

                                float min_dist_sq = 1e9f;
                                float closest_y = -1.0f;
                                int closest_idx = -1;

                                for (int i = 0; i < size; i++) {
                                        float y_reale = d_y_hits[offset + i];
                                        float diff = y_reale - y_theory;
                                        float dist_sq = diff * diff;

                                       if (dist_sq < min_dist_sq) {
                                                min_dist_sq = dist_sq;
                                                closest_y = y_reale;
                                                closest_idx = i;
                                        }
                                }
                                best_y[L] = closest_y;
                                best_indices[L] = closest_idx;
                        }

                        if (!missing_hit) {
                                float Sx = 0.0f, Sxx = 0.0f, Sy = 0.0f, Sxy = 0.0f;

                                for (int L = 0; L < N_layers; L++) {
                                        float x_L = d_s_l + L * dist_interlayers;
                                        Sx  += x_L;
                                        Sxx += x_L * x_L;
                                        Sy  += best_y[L];
                                        Sxy += x_L * best_y[L];
                                }

                                float N_float = static_cast<float>(N_layers);
                                float Delta = (N_float * Sxx) - (Sx * Sx);

                                if (fabsf(Delta) > 1e-6f) {
                                        m_fit = ((N_float * Sxy) - (Sx * Sy)) / Delta;
                                        q_fit = ((Sxx * Sy) - (Sx * Sxy)) / Delta;

                                        float errore_sigma_pixel = sigma;

                                        for (int L = 0; L < N_layers; L++) {
                                                float x_L = d_s_l + L * dist_interlayers;
                                                float residuo = best_y[L] - ((m_fit * x_L) + q_fit);
                                                chi2_globale += (residuo * residuo);
                                        }

                                        chi2_globale /= (errore_sigma_pixel * errore_sigma_pixel);

                                        if (chi2_globale < chi2_threshold) {
                                                is_valid_track = true;
                                        }
                                }
                        }
                }
        }

        int local_index = -1;

        if (is_valid_track) {
                local_index = atomicAdd(&shared_track_count, 1);
        }

        __syncthreads();

        if (threadIdx.x == 0 && threadIdx.y == 0 && shared_track_count > 0) {
                global_offset = atomicAdd(d_track_count, shared_track_count);
        }

        __syncthreads();

        if (local_index != -1) {
                int write_index = global_offset + local_index;

                if (write_index < track_capacity) {
                        d_tracks_out[write_index].m = m_fit;
                        d_tracks_out[write_index].q = q_fit;
                        d_tracks_out[write_index].chi2 = chi2_globale;

                        for (int k = 0; k < N_layers; k++) {
                                d_tracks_out[write_index].hit_indices[k] = best_indices[k];
                        }
                }
        }
}

// =============================================================================
// GpuSingleEventFinder: single-event GPU finder with PERSISTENT device memory.
// Device buffers are allocated once in the constructor and reused across
// calls (they only grow if an event needs more), and the output buffer is
// sized on the real case (at most hits_L0 * hits_Llast pairs). This removes
// the fixed per-call cudaMalloc/cudaFree + PCIe overhead that otherwise
// dominates single-event timing. Returns RAW tracks (no clone removal), so
// callers apply RimuoviCloni on the host, exactly like find_tracks_cpu_quiet.
// =============================================================================

GpuSingleEventFinder::GpuSingleEventFinder(size_t initial_track_capacity) {
        cap_tracks_ = initial_track_capacity;
        CUDA_CHECK(cudaMalloc((void**)&d_tracks_out_, cap_tracks_ * sizeof(Track)));
        CUDA_CHECK(cudaMalloc((void**)&d_track_count_, sizeof(int)));
        CUDA_CHECK(cudaMalloc((void**)&d_offsets_, N_layers * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void**)&d_sizes_,   N_layers * sizeof(int)));
        cap_hits_ = 1u << 14;
        CUDA_CHECK(cudaMalloc((void**)&d_y_hits_, cap_hits_ * sizeof(float)));
}

GpuSingleEventFinder::~GpuSingleEventFinder() {
        if (d_y_hits_)      cudaFree(d_y_hits_);
        if (d_offsets_)     cudaFree(d_offsets_);
        if (d_sizes_)       cudaFree(d_sizes_);
        if (d_tracks_out_)  cudaFree(d_tracks_out_);
        if (d_track_count_) cudaFree(d_track_count_);
}

void GpuSingleEventFinder::EnsureCapacity(size_t n_hits, size_t n_tracks) {
        if (n_hits > cap_hits_) {
                cudaFree(d_y_hits_);
                cap_hits_ = std::max(n_hits, (cap_hits_ * 3) / 2);
                CUDA_CHECK(cudaMalloc((void**)&d_y_hits_, cap_hits_ * sizeof(float)));
        }
        if (n_tracks > cap_tracks_) {
                cudaFree(d_tracks_out_);
                cap_tracks_ = std::max(n_tracks, (cap_tracks_ * 3) / 2);
                CUDA_CHECK(cudaMalloc((void**)&d_tracks_out_, cap_tracks_ * sizeof(Track)));
        }
}

std::vector<Track> GpuSingleEventFinder::FindTracks(const EventHits& hits_lette) {
        auto t0 = std::chrono::high_resolution_clock::now();

        std::vector<float> h_y_hits_flat;
        std::vector<int> h_layer_offsets(N_layers, 0);
        std::vector<int> h_layer_sizes(N_layers, 0);

        size_t total_hits = 0;
        for (size_t i = 0; i < N_layers; i++) {
                h_layer_sizes[i] = (int)hits_lette[i].size();
                h_layer_offsets[i] = (int)total_hits;
                total_hits += hits_lette[i].size();
        }
        if (total_hits == 0 || h_layer_sizes[0] == 0 || h_layer_sizes[N_layers - 1] == 0) {
                last_ms_ = 0.0;
                return {};
        }

        h_y_hits_flat.reserve(total_hits);
        for (size_t i = 0; i < N_layers; i++) {
                h_y_hits_flat.insert(h_y_hits_flat.end(), hits_lette[i].begin(), hits_lette[i].end());
        }

        // True upper bound on tracks writable for this event: one per
        // (hit_L0, hit_Llast) pair. Buffer grows only if needed.
        const size_t max_pairs = (size_t)h_layer_sizes[0] * (size_t)h_layer_sizes[N_layers - 1];
        EnsureCapacity(total_hits, max_pairs);

        cudaMemcpy(d_y_hits_, h_y_hits_flat.data(), total_hits * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_offsets_, h_layer_offsets.data(), N_layers * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_sizes_, h_layer_sizes.data(), N_layers * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemset(d_track_count_, 0, sizeof(int));

        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks(
                (h_layer_sizes[0] + threadsPerBlock.x - 1) / threadsPerBlock.x,
                (h_layer_sizes[N_layers - 1] + threadsPerBlock.y - 1) / threadsPerBlock.y
        );

        TrackFinderKernel<<<numBlocks, threadsPerBlock>>>(
                d_y_hits_, d_offsets_, d_sizes_, d_tracks_out_, d_track_count_, (int)cap_tracks_);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
                std::cerr << "CUDA kernel error (single event finder): " << cudaGetErrorString(err) << std::endl;
        }

        int h_track_count = 0;
        cudaMemcpy(&h_track_count, d_track_count_, sizeof(int), cudaMemcpyDeviceToHost); // implicit sync

        if ((size_t)h_track_count > cap_tracks_) {
                std::cerr << "WARNING: " << h_track_count << " tracks, clamping to " << cap_tracks_ << "\n";
                h_track_count = (int)cap_tracks_;
        }

        std::vector<Track> found_tracks((size_t)h_track_count);
        if (h_track_count > 0) {
                cudaMemcpy(found_tracks.data(), d_tracks_out_, h_track_count * sizeof(Track), cudaMemcpyDeviceToHost);
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        last_ms_ = std::chrono::duration<double, std::milli>(t1 - t0).count();

        return found_tracks;
}

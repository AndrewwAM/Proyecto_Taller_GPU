#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cuda_runtime.h>
#include "VideoLoader.hpp"
#include "CpuFilters.hpp"

// Incluimos los headers para tener acceso a los patrones en GPU
#include "ascii_patterns_5x5.cuh"
#include "ascii_patterns_10x10.cuh"

#define CUDA_CHECK(call) \
    do { cudaError_t err = call; if (err!=cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; exit(1); } } while(0)

// ==========================================
// KERNELS GPU (Replicados)
// ==========================================

// 1. Grayscale
__global__ void k_grayscale(uint8_t* buf, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < w && y < h) {
        int idx = (y * w + x) * 4;
        uint8_t g = (buf[idx]*77 + buf[idx+1]*150 + buf[idx+2]*29) >> 8;
        buf[idx] = g; buf[idx+1] = g; buf[idx+2] = g;
    }
}

// 2. Posterization
__global__ void k_posterization(uint8_t* buffer, int width, int height, float step) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int idx = (y * width + x) * 4;
    int lum = (buffer[idx] * 77 + buffer[idx + 1] * 150 + buffer[idx + 2] * 29) >> 8;
    float quantized = rintf(static_cast<float>(lum) / step) * step;
    quantized = fminf(fmaxf(quantized, 0.0f), 255.0f);
    buffer[idx] = (uint8_t)quantized;
    buffer[idx + 1] = (uint8_t)(quantized * 0.85f);
    buffer[idx + 2] = (uint8_t)(quantized * 0.60f);
}

// 3. Chromatic
__global__ void k_chromatic(const uint8_t* src, uint8_t* dst, int width, int height, float intensity) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    float u = (float)x/width - 0.5f; float v = (float)y/height - 0.5f;
    float mag = sqrtf(u*u + v*v);
    float off = intensity * mag;
    float dx = (mag > 0) ? u/mag : 0; float dy = (mag > 0) ? v/mag : 0;
    int rx = x + (int)(dx*off); int ry = y + (int)(dy*off);
    int bx = x - (int)(dx*off); int by = y - (int)(dy*off);
    
    auto get = [&](int cx, int cy, int ch) {
        cx = max(0, min(cx, width-1)); cy = max(0, min(cy, height-1));
        return src[(cy*width+cx)*4+ch];
    };
    int idx = (y*width+x)*4;
    dst[idx] = get(rx, ry, 0); dst[idx+1] = src[idx+1]; dst[idx+2] = get(bx, by, 2); dst[idx+3] = src[idx+3];
}

// 4. Dither
__constant__ float c_dither_bench[4][4];
const float h_dither_bench[4][4] = {
    {-0.5f, 0.0f, -0.375f, 0.125f}, { 0.25f, -0.25f, 0.375f, -0.125f},
    {-0.3125f, 0.1875f, -0.4375f, 0.0625f}, { 0.4375f, -0.0625f, 0.3125f, -0.1875f}
};
__global__ void k_dither(uint8_t* buffer, int width, int height, float step) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int idx = (y * width + x) * 4;
    int lum = (buffer[idx]*77 + buffer[idx+1]*150 + buffer[idx+2]*29) >> 8;
    float noise = c_dither_bench[y&3][x&3] * step;
    float val = rintf((float)lum + noise / step) * step;
    val = fminf(fmaxf(val, 0.0f), 255.0f);
    buffer[idx] = (uint8_t)val; buffer[idx+1] = (uint8_t)(val*0.85); buffer[idx+2] = (uint8_t)(val*0.6);
}

// 5. ASCII 10x10
__global__ void k_ascii10(uint8_t* buffer, int width, int height) {
    int block_x = blockIdx.x * blockDim.x + threadIdx.x;
    int block_y = blockIdx.y * blockDim.y + threadIdx.y;
    int bw = width / 10; int bh = height / 10; // BLOCK_SIZE 10
    if (block_x >= bw || block_y >= bh) return;

    int sx = block_x * 10; int sy = block_y * 10;
    int sum = 0; int count = 0;
    for (int dy=0; dy<10; dy++) {
        for (int dx=0; dx<10; dx++) {
            int px = sx + dx; int py = sy + dy;
            if (px < width && py < height) {
                int idx = (py * width + px) * 4;
                sum += (buffer[idx]*77 + buffer[idx+1]*150 + buffer[idx+2]*29) >> 8;
                count++;
            }
        }
    }
    if (count == 0) return;
    int avg = sum / count;
    int char_idx = (avg * 9) / 255; 
    
    for (int dy=0; dy<10; dy++) {
        for (int dx=0; dx<10; dx++) {
            int px = sx + dx; int py = sy + dy;
            if (px < width && py < height) {
                int idx = (py * width + px) * 4;
                if (d_patterns_10x10[char_idx][dy * 10 + dx]) {
                    buffer[idx]=255; buffer[idx+1]=255; buffer[idx+2]=255;
                } else {
                    buffer[idx]=0; buffer[idx+1]=0; buffer[idx+2]=0;
                }
            }
        }
    }
}

// 6. ASCII 5x5
__global__ void k_ascii5(uint8_t* buffer, int width, int height) {
    int block_x = blockIdx.x * blockDim.x + threadIdx.x;
    int block_y = blockIdx.y * blockDim.y + threadIdx.y;
    int bw = width / 5; int bh = height / 5; // BLOCK_SIZE 5
    if (block_x >= bw || block_y >= bh) return;

    int sx = block_x * 5; int sy = block_y * 5;
    int sum = 0; int count = 0;
    for (int dy=0; dy<5; dy++) {
        for (int dx=0; dx<5; dx++) {
            int px = sx + dx; int py = sy + dy;
            if (px < width && py < height) {
                int idx = (py * width + px) * 4;
                sum += (buffer[idx]*77 + buffer[idx+1]*150 + buffer[idx+2]*29) >> 8;
                count++;
            }
        }
    }
    if (count == 0) return;
    int avg = sum / count;
    int char_idx = (avg * 9) / 255; 
    
    for (int dy=0; dy<5; dy++) {
        for (int dx=0; dx<5; dx++) {
            int px = sx + dx; int py = sy + dy;
            if (px < width && py < height) {
                int idx = (py * width + px) * 4;
                if (d_patterns_5x5[char_idx][dy * 5 + dx]) {
                    buffer[idx]=255; buffer[idx+1]=255; buffer[idx+2]=255;
                } else {
                    buffer[idx]=0; buffer[idx+1]=0; buffer[idx+2]=0;
                }
            }
        }
    }
}

// ==========================================
// BENCHMARK ENGINE
// ==========================================

struct FilterStats {
    std::string name;
    double avg_cpu_ms = 0;
    double avg_gpu_ms = 0;
};

void run_test(const std::string& name, 
              void(*cpu_func)(const uint8_t*, uint8_t*, int, int),
              void(*gpu_wrapper)(uint8_t*, uint8_t*, int, int),
              const std::vector<uint8_t>& src_host,
              int w, int h, FilterStats& stats) {
    
    int iterations = 20; // Más iteraciones para promediar mejor
    std::vector<uint8_t> dst_host(src_host.size());
    
    // --- CPU ---
    auto t1 = std::chrono::high_resolution_clock::now();
    for(int i=0; i<iterations; ++i) cpu_func(src_host.data(), dst_host.data(), w, h);
    auto t2 = std::chrono::high_resolution_clock::now();
    stats.avg_cpu_ms = std::chrono::duration<double, std::milli>(t2 - t1).count() / iterations;

    // --- GPU ---
    uint8_t *d_src, *d_dst;
    size_t sz = src_host.size();
    CUDA_CHECK(cudaMalloc(&d_src, sz));
    CUDA_CHECK(cudaMalloc(&d_dst, sz));
    
    // Warmup
    CUDA_CHECK(cudaMemcpy(d_src, src_host.data(), sz, cudaMemcpyHostToDevice));
    gpu_wrapper(d_src, d_dst, w, h);
    CUDA_CHECK(cudaDeviceSynchronize());

    t1 = std::chrono::high_resolution_clock::now();
    for(int i=0; i<iterations; ++i) {
        // Incluimos memcpy H->D para simular frame a frame realista
        CUDA_CHECK(cudaMemcpy(d_src, src_host.data(), sz, cudaMemcpyHostToDevice));
        gpu_wrapper(d_src, d_dst, w, h);
        CUDA_CHECK(cudaDeviceSynchronize());
        // No contamos el memcpy de vuelta para no penalizar tanto, 
        // pero en stream real si contaría.
    }
    t2 = std::chrono::high_resolution_clock::now();
    stats.avg_gpu_ms = std::chrono::duration<double, std::milli>(t2 - t1).count() / iterations;

    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
}

// WRAPPERS
void wrap_grayscale(uint8_t* d_in, uint8_t* d_out, int w, int h) {
    cudaMemcpy(d_out, d_in, w*h*4, cudaMemcpyDeviceToDevice);
    dim3 block(16,16);
    dim3 grid((w+15)/16, (h+15)/16);
    k_grayscale<<<grid, block>>>(d_out, w, h);
}
void wrap_posterization(uint8_t* d_in, uint8_t* d_out, int w, int h) {
    cudaMemcpy(d_out, d_in, w*h*4, cudaMemcpyDeviceToDevice);
    dim3 block(16,16);
    dim3 grid((w+15)/16, (h+15)/16);
    k_posterization<<<grid, block>>>(d_out, w, h, 255.0f/3.0f);
}
void wrap_chromatic(uint8_t* d_in, uint8_t* d_out, int w, int h) {
    dim3 block(16,16);
    dim3 grid((w+15)/16, (h+15)/16);
    k_chromatic<<<grid, block>>>(d_in, d_out, w, h, 10.0f);
}
void wrap_dither(uint8_t* d_in, uint8_t* d_out, int w, int h) {
    cudaMemcpy(d_out, d_in, w*h*4, cudaMemcpyDeviceToDevice);
    dim3 block(16,16);
    dim3 grid((w+15)/16, (h+15)/16);
    k_dither<<<grid, block>>>(d_out, w, h, 255.0f/3.0f);
}
void wrap_ascii10(uint8_t* d_in, uint8_t* d_out, int w, int h) {
    cudaMemcpy(d_out, d_in, w*h*4, cudaMemcpyDeviceToDevice);
    dim3 block(16,16);
    // Grid basada en bloques de 10x10, no píxeles
    int blocks_w = (w / 10 + 15) / 16;
    int blocks_h = (h / 10 + 15) / 16;
    dim3 grid(blocks_w, blocks_h);
    k_ascii10<<<grid, block>>>(d_out, w, h);
}
void wrap_ascii5(uint8_t* d_in, uint8_t* d_out, int w, int h) {
    cudaMemcpy(d_out, d_in, w*h*4, cudaMemcpyDeviceToDevice);
    dim3 block(16,16);
    // Grid basada en bloques de 5x5
    int blocks_w = (w / 5 + 15) / 16;
    int blocks_h = (h / 5 + 15) / 16;
    dim3 grid(blocks_w, blocks_h);
    k_ascii5<<<grid, block>>>(d_out, w, h);
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: ./benchmark.out <video.mp4>" << std::endl;
        return 1;
    }

    VideoLoader loader(argv[1]);
    uint8_t* frame = nullptr;
    int w = 0, h = 0;
    loader.load_next_frame(&frame, &w, &h);
    std::vector<uint8_t> host_frame(frame, frame + (w*h*4));
    
    CUDA_CHECK(cudaMemcpyToSymbol(c_dither_bench, h_dither_bench, sizeof(h_dither_bench)));

    std::vector<FilterStats> results;
    
    std::cout << "Benchmarking " << w << "x" << h << " video..." << std::endl;
    std::cout << "--------------------------------------------" << std::endl;

    FilterStats s;

    s.name = "Grayscale";
    run_test(s.name, cpu_grayscale, wrap_grayscale, host_frame, w, h, s);
    results.push_back(s);

    s.name = "Posterization";
    run_test(s.name, cpu_posterization, wrap_posterization, host_frame, w, h, s);
    results.push_back(s);

    s.name = "Chromatic";
    run_test(s.name, cpu_chromatic, wrap_chromatic, host_frame, w, h, s);
    results.push_back(s);

    s.name = "Dither";
    run_test(s.name, cpu_dither, wrap_dither, host_frame, w, h, s);
    results.push_back(s);

    s.name = "ASCII 10x10";
    run_test(s.name, cpu_ascii10, wrap_ascii10, host_frame, w, h, s);
    results.push_back(s);

    s.name = "ASCII 5x5";
    run_test(s.name, cpu_ascii5, wrap_ascii5, host_frame, w, h, s);
    results.push_back(s);

    std::cout << "\nRESULTADOS FINALES:" << std::endl;
    std::cout << "Filtro,CPU(ms),GPU(ms),Speedup" << std::endl;
    for(const auto& r : results) {
        double speedup = (r.avg_gpu_ms > 0) ? r.avg_cpu_ms / r.avg_gpu_ms : 0;
        std::cout << r.name << "," 
                  << r.avg_cpu_ms << "," 
                  << r.avg_gpu_ms << "," 
                  << speedup << "x" << std::endl;
    }

    return 0;
}
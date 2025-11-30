#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include "VideoLoader.hpp"

// Utility for error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// --- KERNEL ---
// Pure arithmetic quantization.
__global__ void fixed_quantization_kernel(uint8_t* buffer, int width, int height, float step) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check
    if (x >= width || y >= height) return;

    int idx = (y * width + x) * 4;

    // 1. Load Pixel
    uint8_t r = buffer[idx];
    uint8_t g = buffer[idx + 1];
    uint8_t b = buffer[idx + 2];

    // 2. Compute Luminance (Standard weighting)
    int lum = (r * 77 + g * 150 + b * 29) >> 8;

    // 3. Fixed Quantization Logic
    // Formula: Round(Value / Step) * Step
    // rintf() is an intrinsic GPU function for fast rounding to nearest integer.
    float val = static_cast<float>(lum);
    float quantized = rintf(val / step) * step;

    // Saturate to 0-255 range
    quantized = fminf(fmaxf(quantized, 0.0f), 255.0f);

    // 4. Apply Aesthetic Tint (Golden/Sepia)
    buffer[idx]     = static_cast<uint8_t>(quantized * 1.00f);
    buffer[idx + 1] = static_cast<uint8_t>(quantized * 0.85f);
    buffer[idx + 2] = static_cast<uint8_t>(quantized * 0.60f);
    // Alpha channel untouched
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_video> <output_video>" << std::endl;
        return 1;
    }

    try {
        VideoLoader loader(argv[1]);

        uint8_t* frame_data = nullptr;
        int width = 0, height = 0;

        // Load metadata
        if (!loader.load_next_frame(&frame_data, &width, &height)) {
            std::cerr << "Failed to load video." << std::endl;
            return 1;
        }

        loader.init_writer(argv[2], width, height, loader.get_fps());

        // --- CONFIGURATION ---
        const float palette_count = 4.0f; 
        const float step = 255.0f / (palette_count - 1.0f);

        // --- MEMORY ALLOCATION (Once) ---
        size_t buffer_size = width * height * 4 * sizeof(uint8_t);
        uint8_t* d_buffer;
        CUDA_CHECK(cudaMalloc(&d_buffer, buffer_size));

        // Kernel dims
        dim3 block_size(16, 16);
        dim3 grid_size((width + block_size.x - 1) / block_size.x,
                       (height + block_size.y - 1) / block_size.y);

        std::cout << "Processing Fixed Quantization (GPU)..." << std::endl;

        do {
            // Host -> Device
            CUDA_CHECK(cudaMemcpy(d_buffer, frame_data, buffer_size, cudaMemcpyHostToDevice));

            // Launch Kernel
            fixed_quantization_kernel<<<grid_size, block_size>>>(d_buffer, width, height, step);
            CUDA_CHECK(cudaGetLastError());

            // Device -> Host
            CUDA_CHECK(cudaMemcpy(frame_data, d_buffer, buffer_size, cudaMemcpyDeviceToHost));

            // Write
            loader.write_frame(frame_data);

        } while (loader.load_next_frame(&frame_data, &width, &height));

        // Cleanup
        CUDA_CHECK(cudaFree(d_buffer));
        std::cout << "Done." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}

#include <algorithm>
#include <iostream>
#include <cuda_runtime.h>
#include "VideoLoader.hpp"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// CPU IMPLEMENTATION
constexpr double dither[4][4] = {
    {-0.5,      0.0,    -0.375,   0.125 },
    { 0.25,    -0.25,    0.375,  -0.125 },
    {-0.3125,   0.1875, -0.4375,  0.0625},
    { 0.4375,  -0.0625,  0.3125, -0.1875}
};

void cpu_filter(uint8_t* buffer, int width, int height) {
	const int stride = width * 4;
	const double palette_size = 16.0;
	const double step = 255.0 / palette_size;

	for (int y = 0; y < height; ++y) {
		uint8_t* row_ptr = buffer + (y * stride);
		const double* dither_row = dither[y & 3];

		for (int x = 0; x < width; ++x) {
			uint8_t* px = row_ptr + (x * 4);

			int lum = (px[0] * 77 + px[1] * 150 + px[2] * 29) >> 8;

			double noise = dither_row[x & 3] * step;
			double raw = static_cast<double>(lum) + noise;
			double color_val = std::round(raw / step) * step;

			uint8_t color = static_cast<uint8_t>(std::clamp(color_val, 0.0, 255.0));
			px[0] = static_cast<uint8_t>(color * 1.00f);
			px[1] = static_cast<uint8_t>(color * 0.85f);
			px[2] = static_cast<uint8_t>(color * 0.60f);
		}
	}
}

// --- CONSTANT MEMORY ---
// The dither matrix is stored in special GPU constant memory for high-speed broadcast access.
__constant__ float c_dither[4][4];

// Host-side definition to copy to GPU later
const float h_dither_data[4][4] = {
    {-0.5f,      0.0f,    -0.375f,   0.125f },
    { 0.25f,    -0.25f,    0.375f,  -0.125f },
    {-0.3125f,   0.1875f, -0.4375f,  0.0625f},
    { 0.4375f,  -0.0625f,  0.3125f, -0.1875f}
};

// --- KERNEL ---
__global__ void dither_kernel(uint8_t* buffer, int width, int height, float step) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check
    if (x >= width || y >= height) return;

    // Calculate pixel index (4 bytes per pixel)
    int idx = (y * width + x) * 4;

    // 1. Load from Global Memory
    uint8_t r = buffer[idx];
    uint8_t g = buffer[idx + 1];
    uint8_t b = buffer[idx + 2];

    // 2. Calculate Luminance
    int lum = (r * 77 + g * 150 + b * 29) >> 8;

    // 3. Apply Dithering
    float noise = c_dither[y & 3][x & 3] * step;

    // Use float for precision blending
    float raw = static_cast<float>(lum) + noise;

    // Quantize: rintf rounds to nearest integer efficiently
    float color_val = rintf(raw / step) * step;

    // Clamp result (saturate logic)
    color_val = fminf(fmaxf(color_val, 0.0f), 255.0f);

    // 4. Apply Tint (Golden/Sepia)
    buffer[idx]     = static_cast<uint8_t>(color_val * 1.00f);
    buffer[idx + 1] = static_cast<uint8_t>(color_val * 0.85f);
    buffer[idx + 2] = static_cast<uint8_t>(color_val * 0.60f);
    // Alpha channel (idx+3) is left untouched
}

// --- HOST HELPER ---
void setup_constants() {
    // Copy the matrix to GPU Constant Memory once
    CUDA_CHECK(cudaMemcpyToSymbol(c_dither, h_dither_data, sizeof(h_dither_data)));
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

        // Load first frame to get dimensions
        if (!loader.load_next_frame(&frame_data, &width, &height)) {
            std::cerr << "Could not load video." << std::endl;
            return 1;
        }

        // Initialize Writer
        loader.init_writer(argv[2], width, height, loader.get_fps());
        
        // --- GPU INITIALIZATION (Once per video) ---
        setup_constants(); // Load dither matrix

        size_t buffer_size = width * height * 4 * sizeof(uint8_t);
        uint8_t* d_buffer;
        
        // Alloc only ONCE
        CUDA_CHECK(cudaMalloc(&d_buffer, buffer_size));

        // Kernel Configuration
        dim3 block_size(16, 16);
        dim3 grid_size((width + block_size.x - 1) / block_size.x,
                       (height + block_size.y - 1) / block_size.y);
        
        // Dithering Configuration
        // Correct step logic for N colors: 255 / (N - 1). 
        const float palette_count = 2.0f; 
        const float step = 255.0f / (palette_count - 1.0f); 

        std::cout << "Processing with GPU..." << std::endl;

        // Process first frame (already loaded)
        do {
            // Copy CPU -> GPU
            CUDA_CHECK(cudaMemcpy(d_buffer, frame_data, buffer_size, cudaMemcpyHostToDevice));

            // Launch Kernel
            dither_kernel<<<grid_size, block_size>>>(d_buffer, width, height, step);
            CUDA_CHECK(cudaGetLastError()); // Check for launch errors
            
            // Copy GPU -> CPU
            CUDA_CHECK(cudaMemcpy(frame_data, d_buffer, buffer_size, cudaMemcpyDeviceToHost));
            
            // Wait for GPU (optional if writing is slow, but good for safety)
            CUDA_CHECK(cudaDeviceSynchronize());

            // Write to disk
            loader.write_frame(frame_data);

        } while (loader.load_next_frame(&frame_data, &width, &height));

        // --- CLEANUP ---
        CUDA_CHECK(cudaFree(d_buffer));
        std::cout << "Done." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}

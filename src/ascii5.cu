#include <iostream>
#include <cuda_runtime.h>
#include "VideoLoader.hpp"
#include "ascii_patterns_5x5.cuh"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define BLOCK_SIZE PATTERN_SIZE_5X5
#define ASCII_PALETTE_SIZE ASCII_PALETTE_SIZE_5X5

__global__ void ascii_filter_kernel(uint8_t* buffer, int width, int height) {
    int block_x = blockIdx.x * blockDim.x + threadIdx.x;
    int block_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    int blocks_width = width / BLOCK_SIZE;
    int blocks_height = height / BLOCK_SIZE;
    
    if (block_x >= blocks_width || block_y >= blocks_height) {
        return;
    }
    
    int pixel_start_x = block_x * BLOCK_SIZE;
    int pixel_start_y = block_y * BLOCK_SIZE;
    
    int sum_luminance = 0;
    int pixel_count = 0;
    
    for (int dy = 0; dy < BLOCK_SIZE; dy++) {
        for (int dx = 0; dx < BLOCK_SIZE; dx++) {
            int px = pixel_start_x + dx;
            int py = pixel_start_y + dy;
            
            if (px < width && py < height) {
                int idx = (py * width + px) * 4;
                uint8_t r = buffer[idx];
                uint8_t g = buffer[idx + 1];
                uint8_t b = buffer[idx + 2];
                
                int lum = (r * 77 + g * 150 + b * 29) >> 8;
                sum_luminance += lum;
                pixel_count++;
            }
        }
    }
    
    if (pixel_count == 0) return;
    
    int avg_luminance = sum_luminance / pixel_count;
    
    int char_index = (avg_luminance * (ASCII_PALETTE_SIZE - 1)) / 255;
    char_index = min(char_index, ASCII_PALETTE_SIZE - 1);
    
    for (int dy = 0; dy < BLOCK_SIZE; dy++) {
        for (int dx = 0; dx < BLOCK_SIZE; dx++) {
            int px = pixel_start_x + dx;
            int py = pixel_start_y + dy;
            
            if (px < width && py < height) {
                int idx = (py * width + px) * 4;
                int pattern_idx = dy * BLOCK_SIZE + dx;
                
                uint8_t pattern_value = d_patterns_5x5[char_index][pattern_idx];
                
                if (pattern_value) {
                    buffer[idx] = 255;
                    buffer[idx + 1] = 255;
                    buffer[idx + 2] = 255;
                } else {
                    buffer[idx] = 0;
                    buffer[idx + 1] = 0;
                    buffer[idx + 2] = 0;
                }
            }
        }
    }
}

void gpu_process_ascii(uint8_t* cpu_buffer, int width, int height) {
    size_t buffer_size = static_cast<size_t>(width * height * 4);
    
    uint8_t* gpu_buffer;
    CUDA_CHECK(cudaMalloc(&gpu_buffer, buffer_size));
    
    CUDA_CHECK(cudaMemcpy(gpu_buffer, cpu_buffer, buffer_size, cudaMemcpyHostToDevice));
    
    int blocks_width = (width / BLOCK_SIZE + 15) / 16;
    int blocks_height = (height / BLOCK_SIZE + 15) / 16;
    
    dim3 block_size(16, 16);
    dim3 grid_size(blocks_width, blocks_height);
    
    ascii_filter_kernel<<<grid_size, block_size>>>(gpu_buffer, width, height);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(cpu_buffer, gpu_buffer, buffer_size, cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(gpu_buffer));
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
        bool first_frame = true;

        std::cout << "Applying ASCII filter (5x5) with GPU..." << std::endl;

        while (loader.load_next_frame(&frame_data, &width, &height)) {
            
            if (first_frame) {
                loader.init_writer(argv[2], width, height, loader.get_fps());
                first_frame = false;
                std::cout << "Video resolution: " << width << "x" << height << std::endl;
                std::cout << "ASCII blocks: " << (width/BLOCK_SIZE) << "x" << (height/BLOCK_SIZE) << std::endl;
            }

            gpu_process_ascii(frame_data, width, height);

            loader.write_frame(frame_data);
        }

        std::cout << "Done." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
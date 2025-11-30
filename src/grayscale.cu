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

__global__ void grayscale_kernel(uint8_t* buffer, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = (y * width + x) * 4;
        
        uint8_t r = buffer[idx];
        uint8_t g = buffer[idx + 1];
        uint8_t b = buffer[idx + 2];
        
        uint8_t gray = static_cast<uint8_t>((r * 77 + g * 150 + b * 29) >> 8);
        
        buffer[idx] = gray;
        buffer[idx + 1] = gray;
        buffer[idx + 2] = gray;
    }
}

void gpu_process_grayscale(uint8_t* cpu_buffer, int width, int height) {
    size_t buffer_size = static_cast<size_t>(width * height * 4);
    
    uint8_t* gpu_buffer;
    CUDA_CHECK(cudaMalloc(&gpu_buffer, buffer_size));
    
    CUDA_CHECK(cudaMemcpy(gpu_buffer, cpu_buffer, buffer_size, cudaMemcpyHostToDevice));
    
    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x,
                   (height + block_size.y - 1) / block_size.y);
    
    grayscale_kernel<<<grid_size, block_size>>>(gpu_buffer, width, height);
    
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

        std::cout << "Processing with GPU..." << std::endl;

        while (loader.load_next_frame(&frame_data, &width, &height)) {
            
            if (first_frame) {
                loader.init_writer(argv[2], width, height, loader.get_fps());
                first_frame = false;
            }

            gpu_process_grayscale(frame_data, width, height);

            loader.write_frame(frame_data);
        }

        std::cout << "Done." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
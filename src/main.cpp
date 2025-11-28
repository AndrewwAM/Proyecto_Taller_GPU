// main.cpp usage example
#include <iostream>
#include "VideoLoader.hpp"

void cpu_process_grayscale(uint8_t* buffer, int width, int height) {
    size_t total_pixels = static_cast<size_t>(width * height);
    uint8_t* ptr = buffer;
    for (size_t i = 0; i < total_pixels; ++i) {
        // Optimized Luminance: Y = (77*R + 150*G + 29*B) >> 8
        uint8_t y = static_cast<uint8_t>((ptr[0] * 77 + ptr[1] * 150 + ptr[2] * 29) >> 8);
        ptr[0] = ptr[1] = ptr[2] = y; // Keep RGBA format but gray color
        ptr += 4;
    }
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

        std::cout << "Processing..." << std::endl;

        while (loader.load_next_frame(&frame_data, &width, &height)) {
            
            // On first frame, init the writer (we need dimensions first)
            if (first_frame) {
                loader.init_writer(argv[2], width, height, loader.get_fps());
                first_frame = false;
            }

            // 1. Process
            cpu_process_grayscale(frame_data, width, height);

            // 2. Save
            loader.write_frame(frame_data);
        }

        // Writer closes automatically in destructor, or call loader.close_writer();
        std::cout << "Done." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}

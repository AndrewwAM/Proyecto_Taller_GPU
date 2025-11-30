// main.cpp usage example
#include <algorithm>
#include <iostream>
#include <cmath>
#include "VideoLoader.hpp"

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
            cpu_filter(frame_data, width, height);

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

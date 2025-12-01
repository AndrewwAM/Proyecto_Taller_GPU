#include <algorithm>
#include <iostream>
#include <cmath>
#include <vector>
#include "VideoLoader.hpp"

void cpu_filter(const uint8_t* src, uint8_t* dst, int width, int height) {
    const int stride = width * 4;
	const double intensity = 10.0;

    for (int y = 0; y < height; ++y) {
        // Source is read-only
        // Destination is write-only
        uint8_t* row_dst = dst + (y * stride);

		double v = (static_cast<double>(y) / height) - 0.5;

        for (int x = 0; x < width; ++x) {
            uint8_t* px_out = row_dst + (x * 4);

			double u = (static_cast<double>(x) / width) - 0.5;
			double magnitude = std::sqrt(u*u + v*v);

			// Evitar división por cero
			double dir_x = (magnitude > 0.0) ? (u / magnitude) : 0.0;
			double dir_y = (magnitude > 0.0) ? (v / magnitude) : 0.0;

			// Fuerza de desplazamiento, aumenta hacia los bordes
			double offset_mag = intensity * magnitude;

			// --- SEPARACIÓN DE CANALES ---
			auto get_channel = [&](int sx, int sy, int ch_idx) -> uint8_t {
				int safe_x = std::clamp(sx, 0, width - 1);
				int safe_y = std::clamp(sy, 0, height - 1);
				return src[(safe_y * stride) + (safe_x * 4) + ch_idx];
			};

			// ROJO
			int r_x = x + static_cast<int>(dir_x * offset_mag);
			int r_y = y + static_cast<int>(dir_y * offset_mag);

			// VERDE
			int g_x = x;
			int g_y = y;

			// AZUL
			int b_x = x - static_cast<int>(dir_x * offset_mag);
			int b_y = y - static_cast<int>(dir_y * offset_mag);

            px_out[0] = get_channel(r_x, r_y, 0); // R
            px_out[1] = get_channel(g_x, g_y, 1); // G
            px_out[2] = get_channel(b_x, b_y, 2); // B
            px_out[3] = src[(y * stride) + (x * 4) + 3];
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

        uint8_t* input_frame = nullptr;
        
        // Output buffer managed by std::vector (RAII)
        // Prevents memory leaks and handles allocation size automatically.
        std::vector<uint8_t> output_buffer;
        
        int width = 0, height = 0;
        bool first_frame = true;

        std::cout << "Processing..." << std::endl;

        while (loader.load_next_frame(&input_frame, &width, &height)) {
            
            if (first_frame) {
                loader.init_writer(argv[2], width, height, loader.get_fps());
                
                // Allocate output memory once.
                // 4 channels (RGBA) assumption based on logic.
                size_t frame_size = static_cast<size_t>(width * height * 4);
                output_buffer.resize(frame_size);
                
                first_frame = false;
            }

            // 1. Process: Read from input_frame, write to output_buffer
            cpu_filter(input_frame, output_buffer.data(), width, height);

            // 2. Save: Write the NEW buffer to disk
            loader.write_frame(output_buffer.data());
        }

        std::cout << "Done." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}

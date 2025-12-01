#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm> // Para std::min/max si fuera necesario en host
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

// --- DEVICE HELPER ---
// Realiza la lectura segura con Clamping (Boundary Check)
__device__ uint8_t get_pixel_clamped(const uint8_t* src, int x, int y, int channel, int width, int height) {
    // Clamp manual para CUDA (equivalente a std::clamp)
    // Aseguramos que x esté entre 0 y width-1
    int sx = max(0, min(x, width - 1));
    int sy = max(0, min(y, height - 1));
    
    // Calculamos el índice lineal plano: (Fila * Ancho + Columna) * 4 canales
    int idx = (sy * width + sx) * 4 + channel;
    return src[idx];
}

// --- KERNEL ---
__global__ void chromatic_aberration_kernel(const uint8_t* src, uint8_t* dst, int width, int height, float intensity) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check
    if (x >= width || y >= height) return;

    // Coordenadas normalizadas (-0.5 a 0.5)
    float u = (static_cast<float>(x) / width) - 0.5f;
    float v = (static_cast<float>(y) / height) - 0.5f;

    // Distancia al centro
    float dist_sq = u*u + v*v;
    float magnitude = sqrtf(dist_sq);

    // Dirección normalizada (evitando división por cero)
    float dir_x = (magnitude > 0.0f) ? (u / magnitude) : 0.0f;
    float dir_y = (magnitude > 0.0f) ? (v / magnitude) : 0.0f;

    // Fuerza de desplazamiento
    float offset_mag = intensity * magnitude;

    // --- CÁLCULO DE COORDENADAS ---
    // Rojo: Desplazamiento hacia afuera
    int r_x = x + static_cast<int>(dir_x * offset_mag);
    int r_y = y + static_cast<int>(dir_y * offset_mag);

    // Verde: Ancla (Sin desplazamiento)
    int g_x = x;
    int g_y = y;

    // Azul: Desplazamiento hacia adentro (opuesto al rojo)
    int b_x = x - static_cast<int>(dir_x * offset_mag);
    int b_y = y - static_cast<int>(dir_y * offset_mag);

    // --- ESCRITURA ---
    int out_idx = (y * width + x) * 4;

    // Lectura corregida: Canal 0 para R, 1 para G, 2 para B
    dst[out_idx + 0] = get_pixel_clamped(src, r_x, r_y, 0, width, height); // R
    dst[out_idx + 1] = get_pixel_clamped(src, g_x, g_y, 1, width, height); // G
    dst[out_idx + 2] = get_pixel_clamped(src, b_x, b_y, 2, width, height); // B
    
    // Alpha: Copia directa del pixel original
    dst[out_idx + 3] = src[out_idx + 3];
}

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

        uint8_t* h_input = nullptr;
        std::vector<uint8_t> h_output;
        
        int width = 0, height = 0;
        
        // Cargar primer frame para obtener dimensiones
        if (!loader.load_next_frame(&h_input, &width, &height)) {
            std::cerr << "Failed to load video." << std::endl;
            return 1;
        }

        loader.init_writer(argv[2], width, height, loader.get_fps());
        
        // Resize del buffer de salida en CPU
        size_t frame_bytes = width * height * 4 * sizeof(uint8_t);
        h_output.resize(frame_bytes);

        // --- GPU ALLOCATION (Solo una vez) ---
        uint8_t *d_input, *d_output;
        CUDA_CHECK(cudaMalloc(&d_input, frame_bytes));
        CUDA_CHECK(cudaMalloc(&d_output, frame_bytes));

        // Configuración de ejecución
        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

        std::cout << "Processing on GPU..." << std::endl;

        // Bucle de procesamiento
        do {
            // 1. Host -> Device
            CUDA_CHECK(cudaMemcpy(d_input, h_input, frame_bytes, cudaMemcpyHostToDevice));

            // 2. Kernel Launch
            chromatic_aberration_kernel<<<grid, block>>>(d_input, d_output, width, height, 10.0f);
            CUDA_CHECK(cudaGetLastError());
            
            // Sincronización implícita en el Memcpy, pero buena práctica para debug
            // CUDA_CHECK(cudaDeviceSynchronize());

            // 3. Device -> Host
            CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, frame_bytes, cudaMemcpyDeviceToHost));

            // 4. Write
            loader.write_frame(h_output.data());

        } while (loader.load_next_frame(&h_input, &width, &height));

        // Cleanup
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_output));
        std::cout << "Done." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}

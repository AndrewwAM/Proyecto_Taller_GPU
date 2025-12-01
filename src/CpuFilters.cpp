#include "CpuFilters.hpp"
#include <cmath>
#include <algorithm>
#include <vector>
#include <iostream>

// --- UTILIDADES ---
static uint8_t clamp_pixel(int v) {
    return static_cast<uint8_t>(std::max(0, std::min(255, v)));
}

// ==========================================
// PATRONES ASCII (Copiados para la CPU)
// ==========================================

// Patrones 5x5
static const uint8_t pattern_5x5[10][25] = {
    {0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0}, // ' '
    {0,0,0,0,0, 0,0,0,0,0, 0,0,1,0,0, 0,0,0,0,0, 0,0,0,0,0}, // '.'
    {0,0,0,0,0, 0,0,1,0,0, 0,0,0,0,0, 0,0,1,0,0, 0,0,0,0,0}, // ':'
    {0,0,0,0,0, 0,0,0,0,0, 0,1,1,1,0, 0,0,0,0,0, 0,0,0,0,0}, // '-'
    {0,0,0,0,0, 0,1,1,1,0, 0,0,0,0,0, 0,1,1,1,0, 0,0,0,0,0}, // '='
    {0,0,0,0,0, 0,0,1,0,0, 0,1,1,1,0, 0,0,1,0,0, 0,0,0,0,0}, // '+'
    {0,0,0,0,0, 0,1,0,1,0, 0,0,1,0,0, 0,1,0,1,0, 0,0,0,0,0}, // '*'
    {0,1,0,1,0, 1,1,1,1,1, 0,1,0,1,0, 1,1,1,1,1, 0,1,0,1,0}, // '#'
    {1,1,0,0,1, 0,0,0,1,0, 0,0,1,0,0, 0,1,0,0,0, 1,0,0,1,1}, // '%'
    {0,1,1,1,0, 1,0,1,1,1, 1,0,1,0,1, 1,0,0,0,0, 0,1,1,1,0}  // '@'
};

// Patrones 10x10
static const uint8_t pattern_10x10[10][100] = {
// ' ' - espacio (vacío)
    {
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0
    },
    // '.' - punto
    {
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,1,1,0,0,0,0,
        0,0,0,0,1,1,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0
    },
    // ':' - dos puntos
    {
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,1,1,0,0,0,0,
        0,0,0,0,1,1,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,1,1,0,0,0,0,
        0,0,0,0,1,1,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0
    },
    // '-' - guión
    {
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,1,1,1,1,1,1,1,1,0,
        0,1,1,1,1,1,1,1,1,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0
    },
    // '=' - igual
    {
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,1,1,1,1,1,1,1,1,0,
        0,1,1,1,1,1,1,1,1,0,
        0,1,1,1,1,1,1,1,1,0,
        0,1,1,1,1,1,1,1,1,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0
    },
    // '+' - más
    {
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,1,1,0,0,0,0,
        0,0,0,0,1,1,0,0,0,0,
        0,0,0,0,1,1,0,0,0,0,
        0,1,1,1,1,1,1,1,1,0,
        0,1,1,1,1,1,1,1,1,0,
        0,0,0,0,1,1,0,0,0,0,
        0,0,0,0,1,1,0,0,0,0,
        0,0,0,0,1,1,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0
    },
    // '*' - asterisco
    {
        0,0,0,0,0,0,0,0,0,0,
        0,0,1,0,1,1,0,1,0,0,
        0,0,0,1,1,1,1,0,0,0,
        0,0,0,0,1,1,0,0,0,0,
        0,1,1,1,1,1,1,1,1,0,
        0,1,1,1,1,1,1,1,1,0,
        0,0,0,0,1,1,0,0,0,0,
        0,0,0,1,1,1,1,0,0,0,
        0,0,1,0,1,1,0,1,0,0,
        0,0,0,0,0,0,0,0,0,0
    },
    // '#' - numeral
    {
        0,0,1,0,0,0,1,0,0,0,
        0,0,1,0,0,0,1,0,0,0,
        1,1,1,1,1,1,1,1,1,0,
        1,1,1,1,1,1,1,1,1,0,
        0,0,1,0,0,0,1,0,0,0,
        0,0,1,0,0,0,1,0,0,0,
        1,1,1,1,1,1,1,1,1,0,
        1,1,1,1,1,1,1,1,1,0,
        0,0,1,0,0,0,1,0,0,0,
        0,0,1,0,0,0,1,0,0,0
    },
    // '%' - porcentaje
    {
        0,1,1,0,0,0,0,1,0,0,
        0,1,1,0,0,0,1,0,0,0,
        0,0,0,0,0,1,0,0,0,0,
        0,0,0,0,1,0,0,0,0,0,
        0,0,0,1,0,0,0,0,0,0,
        0,0,1,0,0,0,0,0,0,0,
        0,1,0,0,0,0,1,1,0,0,
        1,0,0,0,0,0,1,1,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0
    },
    // '@' - arroba (más denso)
    {
        0,1,1,1,1,1,1,1,0,0,
        1,1,0,0,0,0,1,1,1,0,
        1,0,0,1,1,1,0,0,1,0,
        1,0,1,0,0,1,1,0,1,0,
        1,0,1,0,0,1,0,1,1,0,
        1,0,1,1,1,1,0,1,0,0,
        1,0,0,0,0,0,0,0,0,0,
        1,1,0,0,0,0,0,1,0,0,
        0,1,1,1,1,1,1,1,0,0,
        0,0,0,0,0,0,0,0,0,0
    }
}; 


// ==========================================
// IMPLEMENTACIONES
// ==========================================

// 1. GRAYSCALE
void cpu_grayscale(const uint8_t* src, uint8_t* dst, int width, int height) {
    int total_pixels = width * height;
    for (int i = 0; i < total_pixels; ++i) {
        int idx = i * 4;
        uint8_t gray = (src[idx] * 77 + src[idx+1] * 150 + src[idx+2] * 29) >> 8;
        dst[idx] = gray; dst[idx+1] = gray; dst[idx+2] = gray; dst[idx+3] = src[idx+3];
    }
}

// 2. POSTERIZATION
void cpu_posterization(const uint8_t* src, uint8_t* dst, int width, int height) {
    const float step = 255.0f / 3.0f; // 4 colores
    int total_pixels = width * height;
    for (int i = 0; i < total_pixels; ++i) {
        int idx = i * 4;
        int lum = (src[idx] * 77 + src[idx+1] * 150 + src[idx+2] * 29) >> 8;
        float val = std::round(lum / step) * step;
        uint8_t q = static_cast<uint8_t>(std::max(0.0f, std::min(val, 255.0f)));
        
        dst[idx] = q; 
        dst[idx+1] = static_cast<uint8_t>(q * 0.85f); 
        dst[idx+2] = static_cast<uint8_t>(q * 0.60f); 
        dst[idx+3] = src[idx+3];
    }
}

// 3. DITHER
void cpu_dither(const uint8_t* src, uint8_t* dst, int width, int height) {
    const double matrix[4][4] = {
        {-0.5, 0.0, -0.375, 0.125}, {0.25, -0.25, 0.375, -0.125},
        {-0.3125, 0.1875, -0.4375, 0.0625}, {0.4375, -0.0625, 0.3125, -0.1875}
    };
    const double step = 255.0 / 3.0; // 4 colores -> step ~85
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = (y * width + x) * 4;
            int lum = (src[idx] * 77 + src[idx+1] * 150 + src[idx+2] * 29) >> 8;
            double noise = matrix[y & 3][x & 3] * step;
            double val = std::round((lum + noise) / step) * step;
            uint8_t c = clamp_pixel((int)val);
            
            dst[idx] = c; 
            dst[idx+1] = (uint8_t)(c * 0.85); 
            dst[idx+2] = (uint8_t)(c * 0.60); 
            dst[idx+3] = src[idx+3];
        }
    }
}

// 4. CHROMATIC
void cpu_chromatic(const uint8_t* src, uint8_t* dst, int width, int height) {
    const double intensity = 10.0;
    for (int y = 0; y < height; ++y) {
        double v = (double)y / height - 0.5;
        for (int x = 0; x < width; ++x) {
            double u = (double)x / width - 0.5;
            double dist = std::sqrt(u*u + v*v);
            double off = intensity * dist;
            double dx = (dist > 0) ? u/dist : 0;
            double dy = (dist > 0) ? v/dist : 0;
            
            int rx = x + (int)(dx * off); int ry = y + (int)(dy * off);
            int bx = x - (int)(dx * off); int by = y - (int)(dy * off);
            
            auto get = [&](int cx, int cy, int ch) {
                cx = std::max(0, std::min(cx, width-1));
                cy = std::max(0, std::min(cy, height-1));
                return src[(cy*width+cx)*4 + ch];
            };
            
            int idx = (y*width+x)*4;
            dst[idx] = get(rx, ry, 0);
            dst[idx+1] = src[idx+1];
            dst[idx+2] = get(bx, by, 2);
            dst[idx+3] = src[idx+3];
        }
    }
}

// 5. ASCII GENERICO
template<int BLOCK, const uint8_t PATTERNS[][BLOCK*BLOCK]>
void cpu_ascii_template(const uint8_t* src, uint8_t* dst, int width, int height) {
    std::fill(dst, dst + width*height*4, 0); // Fondo negro
    
    int bw = width / BLOCK;
    int bh = height / BLOCK;
    
    for (int by = 0; by < bh; ++by) {
        for (int bx = 0; bx < bw; ++bx) {
            int sx = bx * BLOCK;
            int sy = by * BLOCK;
            
            // Promedio brillo
            int sum = 0;
            for(int dy=0; dy<BLOCK; ++dy)
                for(int dx=0; dx<BLOCK; ++dx) {
                    int idx = ((sy+dy)*width + (sx+dx))*4;
                    sum += (src[idx]*77 + src[idx+1]*150 + src[idx+2]*29) >> 8;
                }
            
            int avg = sum / (BLOCK*BLOCK);
            int char_idx = (avg * 9) / 255; // 10 caracteres (0-9)
            
            // Dibujar
            for(int dy=0; dy<BLOCK; ++dy) {
                for(int dx=0; dx<BLOCK; ++dx) {
                    if (PATTERNS[char_idx][dy*BLOCK + dx]) {
                        int idx = ((sy+dy)*width + (sx+dx))*4;
                        dst[idx] = 255; dst[idx+1] = 255; dst[idx+2] = 255;
                    }
                    // Alpha siempre 255 para que se vea
                    dst[((sy+dy)*width + (sx+dx))*4 + 3] = 255;
                }
            }
        }
    }
}

void cpu_ascii5(const uint8_t* src, uint8_t* dst, int width, int height) {
    cpu_ascii_template<5, pattern_5x5>(src, dst, width, height);
}

// Nota: Usa el pattern_10x10 que definimos arriba (aunque esté con ceros para compilar rápido)
void cpu_ascii10(const uint8_t* src, uint8_t* dst, int width, int height) {
    // Si realmente copiaste los datos al array pattern_10x10, esto funcionará perfecto.
    // Si no, usa pattern_5x5 o similar solo para probar el rendimiento (el cálculo es el mismo).
    cpu_ascii_template<10, pattern_10x10>(src, dst, width, height);
}
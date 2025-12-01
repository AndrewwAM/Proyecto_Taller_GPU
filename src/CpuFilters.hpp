#pragma once
#include <cstdint>

// Declaración de funciones de filtro para CPU
void cpu_grayscale(const uint8_t* src, uint8_t* dst, int width, int height);
void cpu_posterization(const uint8_t* src, uint8_t* dst, int width, int height);
void cpu_dither(const uint8_t* src, uint8_t* dst, int width, int height);
void cpu_chromatic(const uint8_t* src, uint8_t* dst, int width, int height);
void cpu_ascii5(const uint8_t* src, uint8_t* dst, int width, int height);
void cpu_ascii10(const uint8_t* src, uint8_t* dst, int width, int height);
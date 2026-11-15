#ifndef NEON_MOCK_H
#define NEON_MOCK_H

#include <string.h>

// Scalar emulation of the NEON intrinsics used by main.c (for testing on non-ARM machines)
typedef struct {
    float v[4];
} float32x4_t;

static inline float32x4_t vdupq_n_f32(const float x) {
    float32x4_t r = {{x, x, x, x}};
    return r;
}

static inline float32x4_t vld1q_f32(const float *p) {
    float32x4_t r;
    memcpy(r.v, p, sizeof(r.v));
    return r;
}

static inline void vst1q_f32(float *p, const float32x4_t a) {
    memcpy(p, a.v, sizeof(a.v));
}

// a + b * s
static inline float32x4_t vmlaq_n_f32(float32x4_t a, const float32x4_t b, const float s) {
    for (int i = 0; i < 4; ++i) a.v[i] += b.v[i] * s;
    return a;
}

static inline float32x4_t vfmaq_n_f32(const float32x4_t a, const float32x4_t b, const float s) {
    return vmlaq_n_f32(a, b, s);
}

static inline float32x4_t vmaxq_f32(float32x4_t a, const float32x4_t b) {
    for (int i = 0; i < 4; ++i) a.v[i] = a.v[i] > b.v[i] ? a.v[i] : b.v[i];
    return a;
}

#endif

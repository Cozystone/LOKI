/* Portable read-only visual query. Inputs and output are contiguous, row-major float32.
   Use the first 48 rows of ReverseMemory.omega and the entire ReverseMemory.visual.
   An RGB frame contains 16 x 16 x 3 normalized camera samples. No heap or file I/O. */
#ifndef LOKI_V6_RECALL_H
#define LOKI_V6_RECALL_H
#include <math.h>

static inline void loki_v6_recall(const float *omega, const float *visual,
                                  const float *cue, float *past) {
    float coarse[48] = {0};
    float key[24];
    for (int y = 0; y < 16; ++y) {
        for (int x = 0; x < 16; ++x) {
            int tile = 3 * ((y / 4) * 4 + x / 4);
            for (int c = 0; c < 3; ++c)
                coarse[tile+c] += cue[(y*16+x)*3+c] / 16.0f;
        }
    }
    for (int k = 0; k < 12; ++k) {
        float phase = 0.0f;
        for (int i = 0; i < 48; ++i)
            phase += (coarse[i] - 0.5f) * omega[12*i+k];
        key[k] = cosf(phase) / sqrtf(12.0f);
        key[12+k] = sinf(phase) / sqrtf(12.0f);
    }
    for (int i = 0; i < 768; ++i) past[i] = 0.5f;
    for (int k = 0; k < 24; ++k) {
        for (int i = 0; i < 768; ++i)
            past[i] += key[k] * visual[768*k+i];
    }
    for (int i = 0; i < 768; ++i)
        past[i] = fminf(1.0f, fmaxf(0.0f, past[i]));
}
#endif

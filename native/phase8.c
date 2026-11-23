/* Fused pooling, phase projection and validation. Build without fast-math. */
#include <math.h>
#ifdef _WIN32
__declspec(dllexport)
#endif
int phase8(const float *x, const float *h, const float *omega,
           const float *binding, float *key) {
    float tiles[48] = {0}, mean[3] = {0}, phase[32] = {0}, variance = 0;
    for (int i=0; i<768; ++i) {
        if (!(x[i]>=0 && x[i]<=1)) return 0;
        int tile = (i/192)*4 + (i/12)%4;
        tiles[3*tile+i%3] += x[i]/16;
    }
    for (int i=0; i<24; ++i)
        if (!(h[i]>=0 && h[i]<=1)) return 0;
    for (int i=0; i<48; ++i) mean[i%3] += tiles[i]/16;
    for (int i=0; i<48; ++i) {
        tiles[i] -= mean[i%3];
        variance += tiles[i]*tiles[i]/48;
    }
    float scale = .14f/(.1f+sqrtf(variance));
    for (int i=0; i<51; ++i) {
        float v = i<48 ? scale*tiles[i] : .7f*(mean[i-48]-.5f);
        for (int j=0; j<32; ++j) phase[j] += v*omega[i*32+j];
    }
    for (int i=0; i<64; ++i) {
        float v = 2.5f*(2*h[8+i/8]-1)*(h[16+i%8]-h[i%8]);
        for (int j=0; j<32; ++j) phase[j] += v*binding[i*32+j];
    }
    for (int j=0; j<32; ++j) {
        key[j] = cosf(phase[j])/sqrtf(32);
        key[32+j] = sinf(phase[j])/sqrtf(32);
    }
    return 1;
}

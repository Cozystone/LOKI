/* C99, float32 on target. Layout matches PortableField.data.
 * Initialize from its little-endian weights on a little-endian IEEE-754 host.
 * The caller validates normalized input, serializes writes, and owns handoff/IO. */
#ifndef LOKI_V7_FIELD_H
#define LOKI_V7_FIELD_H
#include <math.h>
enum { V7_OUTPUT=776, V7_KEY=64, V7_CELLS=64*(776+64)+51*32 };
static inline void loki7_key(const float *field,const float *rgb,float *key) {
    const float *omega=field+64*(776+64);
    float d[51]={0}, mean[3]={0}, variance=0;
    for(int y=0;y<16;y++) for(int x=0;x<16;x++) for(int c=0;c<3;c++)
        d[3*((y/4)*4+x/4)+c]+=rgb[3*(16*y+x)+c]/16;
    for(int i=0;i<48;i++) mean[i%3]+=d[i]/16;
    for(int i=0;i<48;i++){ d[i]-=mean[i%3]; variance+=d[i]*d[i]/48; }
    float scale=.1f+sqrtf(variance);
    for(int i=0;i<48;i++) d[i]/=scale;
    for(int c=0;c<3;c++) d[48+c]=(mean[c]-.5f)*5;
    for(int j=0;j<32;j++){
        float angle=0;
        for(int i=0;i<51;i++) angle+=d[i]*omega[i*32+j];
        key[j]=cosf(angle*.55f)/sqrtf(32);
        key[32+j]=sinf(angle*.55f)/sqrtf(32);
    }
}
static inline float loki7_precision(const float *field,const float *key,float *v){
    const float *p=field+64*776;
    float norm=0;
    for(int i=0;i<64;i++){
        float sum=0;
        for(int j=0;j<64;j++) sum+=p[i*64+j]*key[j];
        v[i]=sum; norm+=sum*key[i];
    }
    return norm;
}
static inline float loki7_recall(const float *field,const float *rgb,float *out){
    float k[64],v[64];
    loki7_key(field,rgb,k);
    for(int j=0;j<776;j++){
        float sum=.5f;
        for(int i=0;i<64;i++) sum+=k[i]*field[i*776+j];
        out[j]=fminf(1,fmaxf(0,sum));
    }
    return fminf(1,fmaxf(0,1-.1f*loki7_precision(field,k,v)));
}
static inline void loki7_write(float *field,const float *rgb,const float *past){
    float k[64],v[64],gain[64];
    float *p=field+64*776;
    loki7_key(field,rgb,k);
    float denom=1+loki7_precision(field,k,v);
    for(int i=0;i<64;i++) gain[i]=v[i]/denom;
    for(int j=0;j<776;j++){
        float error=past[j]-.5f;
        for(int i=0;i<64;i++) error-=k[i]*field[i*776+j];
        for(int i=0;i<64;i++) field[i*776+j]+=gain[i]*error;
    }
    for(int i=0;i<64;i++) for(int j=0;j<64;j++) p[i*64+j]-=gain[i]*v[j];
}
#endif

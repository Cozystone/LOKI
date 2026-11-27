typedef long long I;
typedef unsigned short H;
__device__ float get(H h){ float f; asm("cvt.f32.f16 %0,%1;":"=f"(f):"h"(h));return f;}
__device__ H put(float f){H h;asm("cvt.rn.f16.f32 %0,%1;":"=h"(h):"f"(f));return h;}
extern "C" __global__ void scores(const H* f,const int* visits,const I* chart,const I* frame,
 const bool* loc,const float* off,const I* body,const float* key,const float* pos,const float* ctx,
 float* cost,int N,int B,int landmark){
 int w=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x%32;
 if(w>=N*B)return;
 int i=w%N,b=w/N,r=body[b];
 if(!visits[i]){if(!lane)cost[w]=__int_as_float(0x7f800000);return;}
 float x=key[32*b+lane]-get(f[52*i+lane]),d=x*x/.32f;
 if(lane<16){x=ctx[16*b+lane]-get(f[52*i+32+lane]);d+=x*x/.1296f;}
 if(lane<2 && !landmark && loc[r] && chart[i]==frame[r]){
 x=pos[2*b+lane]+off[2*r+lane]-get(f[52*i+48+lane]);d+=x*x/.4225f;}
 for(int s=16;s;s/=2)d+=__shfl_down_sync(0xffffffff,d,s);
 if(!lane)cost[w]=d;
}
// Pairwise tests are independent; a serial representative election handles nontransitive neighbors.
extern "C" __global__ void pairs(const I* frame,const float* off,const I* body,const float* key,
 const float* pos,const float* ctx,unsigned char* pair,int B){
 int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=B*B)return;
 int b=t/B,j=t%B,r=body[b],a=body[j];pair[t]=0;if(j>=b||frame[r]!=frame[a])return;
 float d=0;for(int h=0;h<2;h++){float x=pos[2*b+h]+off[2*r+h]-pos[2*j+h]-off[2*a+h];d+=x*x/.4225f;}
 for(int h=0;h<32 && d<=1;h++){float x=key[32*b+h]-key[32*j+h];d+=x*x/.32f;}
 for(int h=0;h<16 && d<=1;h++){float x=ctx[16*b+h]-ctx[16*j+h];d+=x*x/.1296f;}
 pair[t]=d<=1;
}
// Election precedes all updates. Every member of a batch sees the same pre-write state.
extern "C" __global__ void assign(const float* cost,const I* best,const int* visits,const I* gen,
 const I* chart,const I* frame,const bool* located,const float* off,const I* body,
 const float* key,const float* pos,const float* ctx,const bool* live,const unsigned char* pair,I* out,I* meta,int N,int B){
 if(threadIdx.x||blockIdx.x)return;
 int free_slot=0;
 for(int b=0;b<B;b++){int r=body[b],idx=-1;out[b]=-1;if(!live[b])continue;
 if(cost[b]<=1){int i=best[b];if(located[r]&&chart[i]==frame[r])idx=i;}
 else{
 for(int j=0;j<b && idx<0;j++)if(out[j]>=0 && cost[j]>1 && pair[b*B+j]){
 int k=0;while(k<j&&out[k]!=out[j])k++;
 if(k==j)idx=out[j]%N;}
 if(idx<0){while(free_slot<N&&visits[free_slot])free_slot++;
 if(free_slot<N)idx=free_slot++;else meta[1]++;}}
 if(idx>=0)out[b]=gen[idx]*N+idx;
 }}
extern "C" __global__ void update(H* f,H* relief,int* visits,I* last,I* chart,const I* frame,
 bool* located,const float* off,const I* body,const float* key,const float* pos,const float* ctx,
 const float* rise,const float* drop,const I* out,const I* meta,int N,int B,int C){
 int b=blockIdx.x,k=threadIdx.x;if(out[b]<0)return;
 for(int j=0;j<b;j++)if(out[j]==out[b])return;
 int i=out[b]%N,r=body[b];__shared__ int prior,old;
 if(!k){old=visits[i];prior=min(31,old);}__syncthreads();
 int count=0;
 for(int h=k;h<52+C;h+=blockDim.x){
 float sum=0;count=0;
 for(int j=b;j<B;j++)if(out[j]==out[b]){
 int a=body[j];sum+=h<32?key[32*j+h]:h<48?ctx[16*j+h-32]:h<50?pos[2*j+h-48]+off[2*a+h-48]:h==50?drop[j]:h==51?0:rise[C*j+h-52];count++;if(!k)located[a]=true;}
 H* dest=h<52?f+52*i+h:relief+C*i+h-52;
 *dest=put((prior*get(*dest)+sum)/(prior+count));}
 if(!k){visits[i]=(int)min(2147483647LL,(I)old+count);last[i]=meta[0];chart[i]=frame[r];}
}
extern "C" __global__ void edges(const I* prev,const I* now,const bool* live,const bool* boundary,const I* rows,const int* visits,
 const I* gen,const I* chart,I* nbr,int* cnt,int N,int B){
 int b=blockIdx.x;if(threadIdx.x||prev[b]<0)return;
 for(int j=0;j<b;j++)if(prev[j]==prev[b])return;
 int i=prev[b]%N;if(!visits[i]||gen[i]!=prev[b]/N)return;
 I ids[72];int v[72],used=0;
 for(int k=0;k<8;k++)if(cnt[i*8+k]){ids[used]=nbr[i*8+k];v[used++]=cnt[i*8+k];}
 for(int j=b;j<B;j++)if(prev[j]==prev[b]&&live[j]&&!boundary[rows[j]]&&now[j]>=0&&now[j]!=prev[b]){
 int t=now[j]%N;if(!visits[t]||gen[t]!=now[j]/N||chart[t]!=chart[i])continue;
 int k=0;while(k<used&&ids[k]!=now[j])k++;if(k==used){ids[k]=now[j];v[k]=0;used++;}if(v[k]<2147483647)v[k]++;}
 for(int k=0;k<8;k++){
 if(k>=used){cnt[i*8+k]=0;nbr[i*8+k]=-1;continue;}
 int m=k;for(int j=k+1;j<used;j++)if(v[j]>v[m]||(v[j]==v[m]&&ids[j]<ids[m]))m=j;
 nbr[i*8+k]=ids[m];cnt[i*8+k]=v[m];ids[m]=ids[k];v[m]=v[k];}
}
extern "C" __global__ void finish(const float* cost,const I* ix,const I* gen,I* ids,float* sim,
float* conf,float* novel,int N,int K,int W,int B){
 int t=threadIdx.x+blockIdx.x*blockDim.x;if(t>=B*K)return;int b=t/K,j=t%K;
 float d=cost[b*W+j],s=expf(-.5f*d);I i=ix[b*W+j];ids[t]=isfinite(d)?gen[i]*N+i:-1;sim[t]=s;
 if(!j){float second=W>1?expf(-.5f*cost[b*W+1]):0;conf[b]=s*fmaxf(0,1-second/(s+1.e-12f));novel[b]=1-s;}
}
extern "C" __global__ void finite_inputs(const float* key,const float* pos,const float* ctx,
const float* rise,const float* drop,int* ok,int B,int C){
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if((i<B*32&&!isfinite(key[i]))||(i<B*2&&!isfinite(pos[i]))||(i<B*16&&!isfinite(ctx[i]))||
 (C&&((i<B*C&&!isfinite(rise[i]))||(i<B&&!isfinite(drop[i])))))atomicExch(ok,0);
}

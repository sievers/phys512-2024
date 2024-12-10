//nvcc -o libaddVecs.so addVecs.cu -shared -Xcompiler -fPIC -lgomp

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>


__global__
void add_vecs(float *out, float *in1, float *in2, long n)
{
  //slightly more complicated version that should work with arbitrarily large arrays*/
  for (long i=threadIdx.x+blockDim.x*blockIdx.x;i<n;i+=gridDim.x*blockDim.x)
    out[i]=in1[i]+in2[i];
  
}

/*--------------------------------------------------------------------------------*/

__global__
void add_vecs_simple(float *out, float *in1, float *in2, long n)
{
  /* Simple way to add vectors where each element gets one thread.  works great
     for small arrays.*/
  long idx=threadIdx.x+blockDim.x*blockIdx.x;
  if (idx<n)
    out[idx]=in1[idx]+in2[idx];
}

/*--------------------------------------------------------------------------------*/
extern "C" {
void add(float *out, float *in1,float *in2, long n)
{  
  long bs=256;  //Set a block size for threads per block
  long nblock=n/bs;
  if ((nblock*bs)<n)
    nblock++;
  add_vecs_simple<<<nblock,bs>>>(out,in1,in2,n);
  printf("err is currently %s\n",cudaGetErrorString(cudaGetLastError()));

  if (1==0) {
    float *tmp=(float *)malloc(sizeof(float)*n);
    if (cudaMemcpy(tmp,out,n*sizeof(float),cudaMemcpyDeviceToHost)!=cudaSuccess)
      printf("we had an issue with the memcpy.\n");
    else
      printf("first element of in1 is %f\n",tmp[0]);      
    free(tmp);
  }
  
}
}
/*--------------------------------------------------------------------------------*/
extern "C" {
void add2(float *out, float *in1,float *in2, long n)
{  
  long bs=256;
  long nblock=n/bs;
  if ((nblock*bs)<n)
    nblock++;
  long nblock_max=128;
  if (nblock>nblock_max)
    nblock=nblock_max;
  add_vecs<<<nblock,bs>>>(out,in1,in2,n);
  //printf("err is currently %s\n",cudaGetErrorString(cudaGetLastError()));

  if (1==0) {
    float *tmp=(float *)malloc(sizeof(float)*n);
    if (cudaMemcpy(tmp,out,n*sizeof(float),cudaMemcpyDeviceToHost)!=cudaSuccess)
      printf("we had an issue with the memcpy.\n");
    else
      printf("first element of in1 is %f\n",tmp[0]);      
    free(tmp);
  }
  
}
}
/*--------------------------------------------------------------------------------*/
extern "C" {
void add3(float *out, float *in1,float *in2, long n,long nblock,long bs)
{
  //printf("block size/nblock are %ld %ld with n %ld\n",nblock,bs,n);
  add_vecs<<<nblock,bs>>>(out,in1,in2,n);
  //printf("err is currently %s\n",cudaGetErrorString(cudaGetLastError()));
  
}
}

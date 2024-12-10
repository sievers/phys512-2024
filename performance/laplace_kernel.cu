//nvcc -o liblaplace.so laplace_kernel.cu -shared -Xcompiler -fPIC -lgomp                             

#include <stdio.h>
#include <cuda.h>


__global__
void apply_stencil_cuda(float *out, float *in, long n, long m)
{
  for (long i=threadIdx.x+blockDim.x*blockIdx.x;i<n-1;i+=blockDim.x*gridDim.x)
    if (i>0) {
      for (long j=threadIdx.y+blockDim.y*blockIdx.y;j<m-1;j+=blockDim.y*gridDim.y)
	if (j>0) {
	  long ind=i*m+j;
	  float left=in[ind-1];
	  float right=in[ind+1];
	  float bot=in[ind-m];
	  float top=in[ind+m];
	  out[ind]=in[ind]-0.25*(left+right+top+bot);
	}
    }
}

/*--------------------------------------------------------------------------------*/
extern "C"
{
void apply_stencil(float *out, float *in, long n, long m)
{
  dim3 bs(16,16);
  dim3 nb(16,16);
  apply_stencil_cuda<<<nb,bs>>>(out,in,n,m);
  //printf("err is currently %s\n",cudaGetErrorString(cudaGetLastError()));
    
}
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//gcc-10 -O3 -fPIC -fopenmp -shared -o liblaplace_kernel.so laplace_kernel.c -lgomp



void apply_kernel(double *x, double *y, int n, int m)
{
#pragma omp parallel for
  for (int i=1;i<n-1;i++) {
    for (int j=1;j<m-1;j++)
      y[i*m+j]=x[i*m+j-1]+x[i*m+j+1]+x[(i-1)*m+j]+x[(i+1)*m+j]-4*x[i*m+j];
  }
}

/*--------------------------------------------------------------------------------*/
void apply_kernel_chunked(double *x, double *y, int n, int m)
{
  #pragma omp parallel
  {
    double *top=(double *)malloc(m*sizeof(double));
    double *bot=(double *)malloc(m*sizeof(double));
    double *mid=(double *)malloc(m*sizeof(double));
    double *out=(double *)malloc(m*sizeof(double));
#pragma omp for
    for (int i=1;i<n-1;i++) {
      memcpy(top,x+m*(i-1),m*sizeof(double));
      memcpy(mid,x+m*(i),m*sizeof(double));
      memcpy(bot,x+m*(i+1),m*sizeof(double));
      for (int j=1;j<m-1;j++)
	out[j]=top[j]+bot[j]+mid[j-1]-4*mid[j]+mid[j+1];
      memcpy(y+m*i,out,m*sizeof(double));
    }
    free(top);
    free(bot);
    free(mid);
    free(out);
  }
}

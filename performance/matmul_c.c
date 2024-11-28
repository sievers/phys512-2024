#include <stdio.h>
#include <omp.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
  int n=1000;
  if (argc>1)
    n=atoi(argv[1]);
  printf("matrix size: %d\n",n);
  double *a=(double *)(malloc(n*n*sizeof(double)));
  double *b=(double *)(malloc(n*n*sizeof(double)));
  double *c=(double *)(malloc(n*n*sizeof(double)));
  double t1=omp_get_wtime();
#pragma omp parallel for
  for (int i=0;i<n;i++)
    for (int j=0;j<n;j++) {
      a[i*n+j]=1.0;
      b[i*n+j]=2.0;
      c[i*n+j]=0.0;      
    }
  double t2=omp_get_wtime();
  printf("time to fill matrices: %10.5f\n",t2-t1);
  int nbyte=n*n*3*sizeof(double);
  printf("memory bandwidth: %12.4f GB/s\n",nbyte/(t2-t1)/(1024*1024*1024));

  t1=omp_get_wtime();
#pragma omp parallel for
  for (int i=0;i<n;i++)
    for (int j=0;j<n;j++)
      for (int k=0;k<n;k++) {
	//c[i*n+j]+=a[i*n+k]*b[k*n+j];
	c[i*n+j]+=a[i*n+k]*b[j*n+k];
      }
  t2=omp_get_wtime();
  printf("matrix multiply time: %10.4f\n",t2-t1);
  double nop=2.0*n*n*n;
  printf("effective gflops: %10.4f\n",nop/(t2-t1)/1e9);
}
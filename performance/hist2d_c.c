#include <stdio.h>
//you can compile into a shared library from the command line like so.
//gcc-9 -o libhist2d_c.so hist2d_c.c -O3 --shared
//gcc-13 -O3 -o libhist2d_c.so -fPIC hist2d_c.c --shared
void hist2d(long *inds, double *grid, long n, long nx, long ny)
{
  printf("starting hist2d with sizes %ld %ld %ld\n",n,nx,ny);
  for (long i=0;i<n;i++) {
    long myind=inds[2*i]*nx+inds[2*i+1];
    grid[myind]++;
  }
  
}

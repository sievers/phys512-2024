#we'll do an integration using MPI to split up the
#integration region.

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD  #this sets up communicator between processes
rank = comm.Get_rank() #this tells us which process number we are
nproc=comm.Get_size()  #this tells us how many total processes there are


xmin=-20
xmax=20
dx_targ=0.001

x=np.linspace(xmin,xmax,nproc+1)
myxmin=x[rank]
myxmax=x[rank+1]
print('process ',rank,' is responsible for ',myxmin,myxmax)

nx=int((myxmax-myxmin)//dx_targ)
if nx%2==0:  #we need an odd number of points for Simpson's rule
    nx=nx+1
    
x=np.linspace(myxmin,myxmax,nx)
dx=x[1]-x[0]

y=np.exp(-0.5*x**2)
mytot=(y[0]+y[-1]+4*np.sum(y[1:-1:2])+2*np.sum(y[2:-1:2]))*dx/3
#to avoid making a mess, we can just send the pieces of the integral to one process

tot=comm.reduce(mytot)
print('on rank ',rank,' tot is ',tot)
comm.Barrier()

if rank==0:
    print('Final integration result: ',tot,' expected ',np.sqrt(2*np.pi))

comm.Barrier()
MPI.Finalize()

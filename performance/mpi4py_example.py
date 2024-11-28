import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD  #this sets up communicator between processes
rank = comm.Get_rank() #this tells us which process number we are
nproc=comm.Get_size()  #this tells us how many total processes there are
print('my rank is ',rank,' out of ',nproc)


#let's get a 1-element array and put our rank in it
#we have to do this because messages are memory regions,
#not single variables, so we need to make something
#like an array before we can pass it around
rr=np.empty(1,dtype='int')
rr_out=np.zeros(1,dtype='int')
rr[0]=rank


#we can sum all the ranks and put it in rr_out
#Allreduce takes a buffer from all the workers,
#applies an operator across workers and
#then spreads that out to all the workers.  By default,
#the operator is MPI.SUM, the sum.
comm.Allreduce(rr,rr_out)
#we can also do a more python-esque version without the capitalization,
#where mpi4py handles conversions to/from buffers for us
rsum=comm.allreduce(rank)
print('rsum is ',rsum,' with type ',type(rsum))
print('sum of all ranks is ',rr_out[0])
print('should have been ',nproc*(nproc-1)//2)
comm.barrier()

#we can take the product of all ranks as well by passing
#MPI.PROD to Allreduce
comm.Allreduce(rr+1,rr_out,MPI.PROD)
print('product of all ranks+1 is ',rr_out[0])
print('should have been ',np.prod(np.arange(1,nproc+1)))
comm.barrier()

#now let's send a message to a buddy.  We'll define
#the buddy to be one rank up from us, so if I'm process 3, then
#I'll send a message to process4.  If we're the highest process,
#wrap around to zero.
#if we are sending to our right-hand neighbor, get the index
my_send_buddy=(rank+1)%nproc
#now figure out who our left-hand buddy is, because they are
#sending a message to us.
my_receive_buddy=(rank-1)%nproc

rr[0]=2*rank #let's make the message twice our rank

comm.send(rr,dest=my_send_buddy)
mesg=comm.recv(source=my_receive_buddy)
print('process ',rank,' got message ',mesg,' from ',my_receive_buddy)

comm.barrier()
MPI.Finalize()

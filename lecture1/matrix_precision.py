import numpy as np
import time
n=5000
a_double=np.random.randn(n,n) #generates a matrix in double
a_float=a_double.astype('float32') #convert to single

t1=time.time()
b_double=a_double@a_double #multiply double precison
t2=time.time()
b_float=a_float@a_float #multiply single precison
t3=time.time()
print('Time to multiply single/double precision: ',t3-t2,t2-t1)

#You will have to decide if the increased error from single is worth it
print('fractional difference: ',np.std(b_float-b_double)/np.std(b_double))

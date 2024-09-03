import numpy as np
from scipy import integrate
import time

def fun(x,y,lifetime=[1,1e-5]):
    #let's do a 2-state radioactive decay
    dydx=np.zeros(len(lifetime)+1)
    dydx[0]=-y[0]/lifetime[0]
    dydx[1]=y[0]/lifetime[0]-y[1]/lifetime[1]
    dydx[2]=y[1]/lifetime[1]
    return dydx


y0=np.asarray([1,0,0]) 
x0=0
x1=1
t1=time.time();
ans_rk4=integrate.solve_ivp(fun,[x0,x1],y0);
t2=time.time();
print('took ',ans_rk4.nfev,' evaluations and ',t2-t1,' seconds to solve with RK4.')

t1=time.time()
ans_stiff=integrate.solve_ivp(fun,[x0,x1],y0,method='Radau')
t2=time.time()

print('took ',ans_stiff.nfev,' evaluations and ',t2-t1,' seconds to solve implicitly')
print('final values were ',ans_rk4.y[0,-1],' and ',ans_stiff.y[0,-1],' with truth ',np.exp(-1*(x1-x0)))


    

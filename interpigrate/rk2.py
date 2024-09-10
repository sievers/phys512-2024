import numpy as np
def fun(x,y):
    return -y
def fun2(x,y):
    return -y**2
def fun3(x,y):
    return x/y
def fun4(x,y):
    return np.cos(x)

#y' = -y, then y=exp(-x)+c, if y(0)+1, then c=0
#y' = -y^2, dy/y^2 = -dx, 1/y=x+c, y=1/(x+c), if y(0)=1, then c=1
#y'=x/y, ydy=xdx, y^2=x^2+c, y=(x^2+c)^0.5.  y(0)=1 -> c=1
#y'=cos(x), y=sin(x)+c, if y(0)=1, then c=1
def rk2(f,x,y,h):
    k0=h*f(x,y)
    k1=h*f(x+h,y+k0)
    return (k0+k1)/2

def rk2b(f,x,y,h):
    k0=h*f(x,y)
    k1=h*f(x+h/2,y+k0/2)
    return k1

def rk22(f,x,y,h):
    #take two half-size steps with second order
    #and combine to cancel leading error term.
    
    y1=rk2(f,x,y,h)
    y2a=rk2(f,x,y,h/2)
    y2b=rk2(f,x+h/2,y+y2a,h/2)
    return (4*(y2a+y2b)-y1)/3


def rk4(f,x,y,h):
    k1=h*f(x,y)
    k2=h*f(x+h/2,y+k1/2)
    k3=h*f(x+h/2,y+k2/2)
    k4=h*f(x+h,y+k3)
    return (k1+2*k2+2*k3+k4)/6


x=np.linspace(0,1,100)
y=0*x
y2=0*x
y3=0*x
y[0]=1
y2[0]=y[0]
y3[0]=y[0]
fuse=fun2
if fuse==fun:
    pred=np.exp(-x)
if fuse==fun2:
    c=1
    pred=1/(x+c)
if fuse==fun3:
    c=1
    pred=(x**2+c)**0.5
if fuse==fun4:
    c=1
    pred=1+np.sin(x)

for i in range(len(x)-1):
    y[i+1]=y[i]+rk2(fuse,x[i],y[i],x[i+1]-x[i])
    y2[i+1]=y2[i]+rk2b(fuse,x[i],y2[i],x[i+1]-x[i])
    y3[i+1]=y3[i]+rk22(fuse,x[i],y3[i],x[i+1]-x[i])
print('errs are ',np.std(y-pred),np.std(y2-pred),np.std(y3-pred),np.std(y-y2))

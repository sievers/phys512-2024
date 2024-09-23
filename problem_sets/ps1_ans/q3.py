import numpy as np

#let xm,x0,xp=[-1,0,1], and ym,y0,yp be the function values
#if y=ax^2+bx+c, then c=y0, b=(yp-ym)/2
#and a=(yp+ym-2y0)/2
#we can evaluate to double check:
#y(-1)=(yp+ym-2y0)/2-(yp-ym)/2+y0 = ym
#y(0)=0+0+y0=y0
#y(1)=(yp+ym-2y0)/2 + (yp-ym)/2+y0=yp
#integral from 0 to 1 of a x^2 +bx +c is ax^3/3 + bx^2/2 +cx
#= a/3+b/2+c.  Plug in: (yp+ym-2y0)/6+(yp-ym)/4+y0
#expand and group:  (1/6+1/4)=5/12, 1/6-1/4=-1/12, and -1/3+1=2/3=8/12
#so area is (-ym + 8 y0 + 5 yp)/12

def flexsimp(y,dx):
    if len(y)%2==0:
        #for fun, let's be a little clever.  since the cubic term doesn't
        #cancel on the odd extra interval, we can look at both of the areas
        #on the left and right edge, and pick the smaller one
        extra=(-y[-3]+8*y[-2]+5*y[-1])/12*dx
        extra2=(5*y[0]+8*y[1]-y[2])/12*dx
        if np.abs(extra)<np.abs(extra2):
            return flexsimp(y[:-1],dx)+extra
        else:
            return flexsimp(y[1:],dx)+extra2
    else:
        ans=(y[0]+y[-1]+4*np.sum(y[1::2])+2*np.sum(y[2:-1:2]))*dx/3
        return ans

npt=40
x=np.linspace(-1,1,npt+1)
dx=x[1]-x[0]
y=np.exp(x)
ans=y[-1]-y[0]
a_even=flexsimp(y,dx)
print('odd point simpsons: ',a_even,' with err ',a_even-ans)

x=np.linspace(-1,1,npt)
dx=x[1]-x[0]
y=np.exp(x)
a_odd=flexsimp(y,dx)
print('even point simpson: ',a_odd,' with err ',a_odd-ans)

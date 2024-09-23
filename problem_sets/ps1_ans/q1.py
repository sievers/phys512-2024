import numpy as np

#we know shrinking dx by 2 shrinks error by 4
#so d(2dx)=4e+d, d(1dx)=1e+d, (4d(1dx)-d(2dx))/3 cancels e, and leaves d
#so that should be our deriv. Ignoring constants,
#f(x+dx)-f(x-dx) = f0+f' dx + f'' dx^2 + f''' dx^3 +... - (f0-f'+f''-f'''...)
#so all odd terms in Taylor series are cancelled.  That means the next term 
#that survives looks like f'''' dx^4, so setting eps=f'''' dx^4, we have
#dx~(eps/f'''')^(1/4)

xx=np.linspace(-2,2,5)
dx=0.02

x0=1
a=0.01  #y=exp(ax), so change a here to see how your estimate depends
x=x0+dx*xx
y=np.exp(a*x)
d1=(y[3]-y[1])/(2*dx)
d2=(y[4]-y[0])/(4*dx)
d=(4*d1-d2)/3
ans=a*y[2]
print('fractional deriv errors are ',(d1-ans)/ans,(d2-ans)/ans,(d-ans)/ans)
#for "typical" x take average of x range, in case we asked for deriv at 0
eps=1e-16*np.mean(np.abs(x))

#as an estimate, assume ratio of derivatives is roughly
#constant, so f'''' ~ f''*(f''/f')^2.  This works if the
#taylor series is roughly constant for f(x), but you have a rescaled
#f(ax).  That means our fourth deriv is roughly the second deriv
#times the square of (f''/f')
curve=np.abs(y[1]+y[3]-2*y[2])/dx**2
df4=np.abs(curve/d)**2*np.abs(curve)

dx_targ=(eps/df4)**(1/4)
print('my target dx is ',dx_targ)

x_new=x0+dx_targ*xx
y_new=np.exp(a*x_new)
d1_new=(y_new[3]-y_new[1])/(2*dx_targ)
d2_new=(y_new[4]-y_new[0])/(4*dx_targ)
d_new=(4*d1_new-d2_new)/3
print('best-guess deriv ',d_new,d_new-ans)

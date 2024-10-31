import numpy as np
from matplotlib import pyplot as plt

#P(y)dy=P(x)dx, P(y)dy/dx dy = P(x)dx, P(y) =P(x)/(dy/dx) = P(x)dx/dy
#let y=atan(x), x=tan(y), dx/dy=sec^2(y) or dy/dx=1/(1+x^2)
#so, P(y)=P(tan(y))sec^2(y)

y=np.linspace(-np.pi/2,np.pi/2,1001)
y=0.5*(y[1:]+y[:-1])
tany=np.tan(y)
cosy=np.cos(y)
p=np.exp(-0.5*tany**2)/cosy**2



plt.ion()
plt.figure(1)
plt.clf()
plt.plot(y,p)
plt.show()
plt.savefig('arctan_gauss.png')


n=10000000
yy=np.pi*(np.random.rand(n)-0.5)

myp=np.exp(-0.5*np.tan(yy)**2)/np.cos(yy)**2
fac=1.01*p.max()
accept=(np.random.rand(n)*fac)<myp
print('accept fraction is ',np.mean(accept))
y_use=yy[accept]
x_use=np.tan(y_use)
aa,bb=np.histogram(x_use,np.linspace(-4,4,41))
b_cent=0.5*(bb[1:]+bb[:-1])
pred=np.exp(-0.5*b_cent**2)
pred=pred/pred.sum()
aa=aa/aa.sum()
plt.figure(2)
plt.clf()
plt.plot(b_cent,pred)
plt.bar(b_cent,aa,0.15)
plt.show()
plt.savefig('arctan_gauss_pdf.png')

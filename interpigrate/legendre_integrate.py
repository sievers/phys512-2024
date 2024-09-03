import numpy as np

def legendre_coeffs(n):
    if (n<3):
        print('we are not checking edge cases.')
        return
    x=np.linspace(-1,1,n)
    A=np.zeros([n,n])
    A[:,0]=1
    A[:,1]=x
    for i in range(1,n-1):
        A[:,i+1]=( (2*i+1)*x*A[:,i]-i*A[:,i-1])/(i+1)
    Ainv=np.linalg.inv(A)
    coeffs=Ainv[0,:]
    return coeffs

coeffs3=legendre_coeffs(3)
coeffs5=legendre_coeffs(5)
coeffs7=legendre_coeffs(7)
coeffs51=legendre_coeffs(51)

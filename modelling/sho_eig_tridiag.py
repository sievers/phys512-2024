#example code to solve Schrodinger equation for a simple harmonic oscillator
import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import eigh_tridiagonal
import time
plt.ion()

#let's set up our x-range first
x=np.linspace(-15,15,1001)
dx=x[1]-x[0]

#set up our potential, assuming hbar=m=1
#In the units normally used in the text, then
#the potential is 0.25 x^2
V=0.25*x**2

#and make the diagonal/off-diagonal vectors of the Hamiltonian
#diagonal vector of the Hamiltonian
H0=2/dx**2*np.ones(len(x))+V
#off-diagonal vector of the Hamiltonian
H1=-1/dx**2*np.ones(len(x)-1)
                     
#solve the Shrodinger equation (in one line!)
#using the specialized symmetric tridiagonal
#eigenvalue routine in scipy.  This is much faster than making a dense matrix
#mostly full of zeros
e,v=eigh_tridiagonal(H0,H1)
plt.figure(1)
plt.clf()
plt.plot(x,v[:,0]/np.sqrt(dx),linewidth=2)
plt.plot(x,np.exp(-0.25*x**2)*2/np.pi)
plt.legend(['Numerical','Analytic'])
plt.title('SHO Ground State')
plt.savefig('sho_ground.png')
plt.figure(2)
plt.clf()
plt.plot(0.5+np.arange(20),'o')
plt.plot(e[:20],'.')
plt.legend(['Analytic','Numerical'])
plt.xlabel('Rank')
plt.ylabel('Eigenvalue')
plt.title('Eigenvalues of Simple Harmonic Oscillator')
plt.savefig('sho_energy.png')

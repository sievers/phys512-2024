import numpy as np
import camb
from matplotlib import pyplot as plt
import time


def get_spectrum(pars,lmax=3000):
    #print('pars are ',pars)
    H0=pars[0]
    ombh2=pars[1]
    omch2=pars[2]
    tau=pars[3]
    As=pars[4]
    ns=pars[5]
    pars=camb.CAMBparams()
    pars.set_cosmology(H0=H0,ombh2=ombh2,omch2=omch2,mnu=0.06,omk=0,tau=tau)
    pars.InitPower.set_params(As=As,ns=ns,r=0)
    pars.set_for_lmax(lmax,lens_potential_accuracy=0)
    results=camb.get_results(pars)
    powers=results.get_cmb_power_spectra(pars,CMB_unit='muK')
    cmb=powers['total']
    tt=cmb[:,0]    #you could return the full power spectrum here if you wanted to do say EE
    return tt[2:]


plt.ion()

pars=np.asarray([60,0.02,0.1,0.05,2.00e-9,1.0])
planck=np.loadtxt('COM_PowerSpect_CMB-TT-full_R3.01.txt',skiprows=1)
ell=planck[:,0]
spec=planck[:,1]
errs=0.5*(planck[:,2]+planck[:,3]);
model=get_spectrum(pars)
model=model[:len(spec)]
resid=spec-model
chisq=np.sum( (resid/errs)**2)
print("chisq is ",chisq," for ",len(resid)-len(pars)," degrees of freedom.")
#read in a binned version of the Planck PS for plotting purposes
planck_binned=np.loadtxt('COM_PowerSpect_CMB-TT-binned_R3.01.txt',skiprows=1)
errs_binned=0.5*(planck_binned[:,2]+planck_binned[:,3]);
plt.clf()
plt.plot(ell,model)
plt.errorbar(planck_binned[:,0],planck_binned[:,1],errs_binned,fmt='.')
plt.show()

omcvec=np.linspace(0.05,0.15,21)
pp=np.asarray(pars)
models=[None]*len(omcvec)
for i,omc in enumerate(omcvec):
    pp[2]=omc
    model=get_spectrum(pars)
    model=model[:len(spec)]
    plt.plot(ell,model,'k')
    plt.pause(0.01)
    models[i]=model

while True:
    for i in range(len(omcvec)):
        plt.clf()
        plt.errorbar(planck_binned[:,0],planck_binned[:,1],errs_binned,fmt='.')
        plt.plot(ell,models[i],'k')
        plt.ylim([0,6000])
        plt.text(1500,3000,r'$\Omega_Ch^2=$'+f'{omcvec[i]:.3f}',fontsize=20)
        #plt.text(1500,3000,r'$\Omega_C=$'+repr(omcvec[i]))
        plt.pause(0.2)
    

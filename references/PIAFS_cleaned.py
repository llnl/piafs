import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pylab
import scipy as scp
from scipy.fft import fft, ifft, fft2, ifft2, fftshift, ifftshift
from scipy.signal import find_peaks
from scipy.interpolate import *
import multiprocessing
import time
from scipy.signal import argrelextrema
from scipy.fft import fft, fftfreq

from scipy.constants import c
from scipy.constants import k as kb
from scipy.constants import epsilon_0
from scipy.constants import mu_0
from scipy.constants import m_e
from scipy.constants import m_p
from scipy.constants import e
from scipy.constants import h
from scipy.constants import N_A
from scipy.constants import R as Rm



# # Lasers properties


theta = 0.17*np.pi/180 #half angle between beams

print(2*np.pi*np.sin(theta))

n=1 #change the resolution

tpulse = 10*10**-9 # 10ns, corresponding to the pulse duration

# lUV = 266*10**-9 
lUV = 248*10**-9 # value article Michine & Yoneda writing beams wavelength (must be between 200 and 300)
kUV = 2*np.pi/lUV # writing beams wave vector 
wUV=kUV*c

kg = 2*kUV*np.sin(theta) # grating wave vector m**-1
lg = 2*np.pi/kg # grating wavelength m

ldiff = 532*10**-9 #value article M&Y, diffracted beam
kdiff = 2*np.pi/ldiff

print(lg,kg)

# # Constants

CvO2=5./2*kb
kaw = 1 # normalized acoustic wavevector
law = 2*np.pi/kaw # normalized acoustic wavelength 

nu = c/lUV
F0 = 2000 # Fluence of the writing beams in J/m2
F0v = np.linspace(0,F0,26)
print(F0v)
I0 = F0/tpulse

mmolO2 = 0.032 #O2molar mass in kg m-3 
gamma = 7./5 # for O2


# # Initial conditions

rhoinit = 1 # initial density normalized to O2 density
uinit = 0
Ti = 288 # Initial temperature in K
Ptot = 101325 # total gas pressure in Pa for 288 K
p00 = 1 # initial pressure normalized to Ptot
sigma = 0.1 #CFL

# # Spatial steps (in the x direction, normalized)

Nperx = 5 #nb of spatial periods of the acoustic wave in the box, must be odd to be centered in x=0
xl = -Nperx/2*law
xr = Nperx/2*law
nxppp = 30*n #nb of points per period
dx = law/nxppp
nx = int((xr-xl)/dx)
nxplot = int(nx/2) # index corresponding to x=0 of the plots, where the density modulation is max
nxmin = int(nx/2+law/2/dx)
xv = np.linspace(xl,xr,nx)

#z, not normalized (because no hydro in the z direction)

Lz = 30*10**-3 # gas length, 
nz = 22 #at least 20 otherwise bug
zv = np.linspace(0,Lz,nz)
dz=Lz/nz
print(dx)


# # Temporal steps, normalized

Npert = 5 #nb of temporal periods of the acoustic wave in the simulation box
t0 = 0
tf = Npert*2*np.pi
ntppp = nxppp/sigma #nb of points per period
dt = 2*np.pi/ntppp
nt=int(tf/dt)
tplot=2*np.pi/2 # max of first period (if linear)
ntplot = int(tplot/dt) # corresponding index
tv = np.linspace(t0,tf,nt)


# # Chemical properties

fCO2 = 0. #CO2 fraction
fO2 = 1-fCO2
fO3v = np.linspace(0.005,0.15,30) # Ozone fraction varying between 0.05% and 0.15%
print('fO3v=',fO3v)

sO3 = 1.1e-17*1e-4 #ozone absorption cross-section in the center of the Hartley band in m2
Fs = h*nu/sO3 # saturation fluence

NO2 = 1+2.71*10**-4 #O2 refractive index
NCO2 = 1+4.5*10**-4 #CO2 refractive index
Ngas = fO2*NO2+fCO2*NCO2 # gas refractive index


#Initial concentration in m-3
n_O2 = fO2*Ptot/(kb*Ti)
n_O3v = fO3v*Ptot/(kb*Ti)
n_CO2 = fCO2*Ptot/(kb*Ti)
n_hnumax = I0 / c / h / nu
# print('n_O2=',n_O2,'n_hnumax=',n_hnumax,'n_O3v=',n_O3v)


Ei = Ptot/(gamma-1)/n_O2 #initial internal energy in J

# Heat energy in eV converted in J and normalized to initial internal energy

q0a = 0.73 * e / Ei
q0b = 2.8 * e / Ei

q1a = 0.29 * e / Ei # 3 in Pierre's matlab code
q1b = 0.84 * e / Ei

q2a = 4.3 * e / Ei
q2b = 0.81 * e / Ei

q3a = 0.42 * e / Ei #4 in Pierre's matlab code
q3b = 1.16 * e / Ei

q4 = 1.81 * e / Ei #5 in Pierre's matlab code

qqv3 = 0.42 * e / Ei #1 in Pierre's matlab code

qCO2 = 1.23 * e / Ei


#physical units to link with coefficient rate


cs = np.sqrt(gamma*Rm*Ti/mmolO2) # initial acoustic velocity
print('cs=',cs)
wg = kg*cs
print('wg=',wg)
Tawnm = lg/cs # accoustic time period in physical unit
print('T=',Tawnm*10**9,'ns')
ntpulse = np.argmin(np.abs(np.array(tv) - tpulse*cs*kg)) # index of the ind of the pulse in the vector tv
print('ntpulse=',ntpulse,'tf=',tf/cs/kg,'nt=',nt)

# Reaction rates, m3 s-1 before factnorm

factnorm=wg/n_O2 # normalisation for k reaction rates (ion acoustic period and O2 concentration)

k0a = 0.9*3.3*10**-13/factnorm 
k0b = 0.1*3.3*10**-13/factnorm


k1a = 0.8*3.95*10**-17/factnorm #3 in Pierre's matlab code
k1b = 0.2*3.95*10**-17/factnorm

k2a = 1.2*10**-16/factnorm
k2b = 1.2*10**-16/factnorm

k3a = 1.2*10**-17/factnorm #4 matlab
k3b = 10**-17/factnorm

k4 = 2*10**-16/factnorm # 5 matlab, negligible

kqv3 = 0.2*3*10**-17/factnorm #1 matlab, negligible

kCO2 = 1.1*10**-16/factnorm


print('tnorm=',1/2/np.pi*Tawnm*10**9)



def upF(Utab,nx):
    Ftab = np.zeros((3,nx))
    Ftab[0,:] = Utab[1,:]
    Ftab[1,:] = (Utab[1,:]*Utab[1,:]/Utab[0,:]) + (gamma - 1)*(Utab[2,:] - Utab[1,:]*Utab[1,:]/(2*Utab[0,:])) 
    Ftab[2,:] = (Utab[1,:]/Utab[0,:])*(Utab[2,:] + (gamma - 1)*(Utab[2,:] - Utab[1,:]*Utab[1,:]/(2*Utab[0,:]))) 
    #print(F[1,:])
    return Ftab

def Chem_Hydro_Eulersolver(zi,nO3,F0): 
    
    I0 = F0/tpulse 
    rhoinit = 1
    uinit=0
    
    Pvinit = np.zeros(nx)
    Utab = np.zeros((3,nx))
    Ftab = Utab.copy()
    Ftabp = Utab.copy()
    Ftabm = Utab.copy()
    Ftab12 = Utab.copy()
    Utabt = Utab.copy()
    Utabc = Utab.copy()
    Utabp = Utab.copy()
    Utabm = Utab.copy()
    nv_O3 = np.zeros((nx,nz))
    nv_O2 = np.zeros((nx,nz))
    nv_CO2 = np.zeros((nx,nz))
    nv_1D = np.zeros((nx,nz))
    nv_1Dtg = np.zeros((nx,nz))
    nv_1Sg = np.zeros((nx,nz))
    nv_3S = np.zeros((nx,nz))
    nv_hnu = np.zeros((nx,nz))
    nv_hnu0 = np.zeros((nt,nx))
    nv_O3c = np.zeros((nx,nz))
    nv_1Dc =np.zeros((nx,nz))
    nv_1Dtgc =np.zeros((nx,nz))
    nv_1Sgc = np.zeros((nx,nz))
    nv_3Sc = np.zeros((nx,nz))
    nv_hnuc = np.zeros((nx,nz))
    ntab_hnu = np.zeros(((nt,nx,nz)))
    ntab_O3 = np.zeros(((nt,nx,nz)))
    
    Uxt = np.zeros(((3,nt,nx)))
    Fxt = Uxt.copy()
    Qxt = np.zeros((nt,nx))
    Hxt = Qxt.copy()
    nhnuxt = np.zeros((nt,nx))
    nO3xt = np.zeros((nt,nx))
    O3zt = np.zeros((nt,nz))
    Izt = np.zeros((nt,nz))
    nO3zt = np.zeros((nt,nz))
    nhnuzt = np.zeros((nt,nz))
    
    
    Qv = np.zeros((nx,nz))
    Hv = np.zeros(nx)
    Qvc = np.zeros(nx)
    I0v = np.zeros(nx)

    
    
    # Initialize arrays
    ix = np.arange(nx)
    nv_O2 = np.ones((nx,nz))
    nv_O3 = nO3 / n_O2 * np.ones((nx,nz))
    I0v = I0 * (1 + np.cos(np.linspace(0, 2 * np.pi, nx) * Nperx))
    
    ntab_hnu[0:ntpulse, :, 0] = I0v / (c * h * nu * n_O2)
    
    ntab_O3[0] = nO3 / n_O2
    
    Pvinit = np.full(nx, p00)
    
    Utab[0,:] = rhoinit 
    Utab[1,:] = rhoinit*uinit
    Utab[2, :] = Pvinit / (gamma - 1) + 0.5 * rhoinit * uinit**2
    
    Ftab = np.zeros((3, nx))
    Ftab[0, :] = Utab[1, :]
    Ftab[1, :] = Ftab[0, :] * (Utab[1, :] / Utab[0, :]) + Pvinit
    Ftab[2, :] = Ftab[0, :] * (Utab[2, :] / Utab[0, :] + (Utab[1, :] / Utab[0, :])**2 * 0.5)
    
    
        
    for it in range(0,nt):
        for iz in range(1,zi+1):
            ntab_hnu[it,:,iz] = ntab_hnu[it,:,iz-1] - dz*sO3*n_O2*ntab_O3[it-1,:,iz-1]*ntab_hnu[it,:,iz-1]
            
        nv_O3c = nv_O3 + dt*(
        - kqv3*nv_O3*nv_1Dtg
        - (k0a+k0b)*nv_O3*ntab_hnu[it]
        - (k2a+k2b)*nv_O3*nv_1D
        - (k3a+k3b)*nv_O3*nv_1Sg
        - k4*nv_3S*nv_O3 )
        nv_O3c[0] = nv_O3c[1] # absorbing BC
        nv_O3c[-1] = nv_O3c[-2]
        # nv_O3c[0] = nv_O3c[-2] # Periodic BC
        # nv_O3c[-1] = nv_O3c[1]            
            
        nv_1Dc = nv_1D + dt*(
        + k0a*ntab_hnu[it]*nv_O3
        - (k2a+k2b)*nv_O3*nv_1D 
        - (k1a+k1b)*nv_1D*nv_O2
        - kCO2*nv_1D*nv_CO2 )
        nv_1Dc[0] = nv_1Dc[1]
        nv_1Dc[-1] = nv_1Dc[-2]
        # nv_1Dc[0] = nv_1Dc[-2]
        # nv_1Dc[-1] = nv_1Dc[1]
        
        nv_1Dtgc[:] = nv_1Dtg + dt*(
        - kqv3*nv_1Dtg*nv_O3
        + k0a*ntab_hnu[it]*nv_O3
        - k1b*nv_1Dtg*nv_O2 
        + k4*nv_3S*nv_O3)
        nv_1Dtgc[0] = nv_1Dtgc[1]
        nv_1Dtgc[-1] = nv_1Dtgc[-2]
        # nv_1Dtgc[0] = nv_1Dtgc[-2]
        # nv_1Dtgc[-1] = nv_1Dtgc[1]
        
        nv_1Sgc = nv_1Sg + dt*(
        + k1a*nv_1D*nv_O2
        - (k3a+k3b)*nv_1Sg*nv_O3 )
        nv_1Sgc[0] = nv_1Sgc[1]
        nv_1Sgc[-1] = nv_1Sgc[-2]
        # nv_1Sgc[0] = nv_1Sgc[-2]
        # nv_1Sgc[-1] = nv_1Sgc[1]   
        
        
        Qvc = ((q0a*k0a+q0b*k0b)*ntab_hnu[it]*nv_O3
        +(q1a*k1a+q1b*k1b)*nv_O2*nv_1D
        +(q2a*k2a+q2b*k2b)*nv_1D*nv_O3
        +(q3a*k3a+q3b*k3b)*nv_1Sg*nv_O3
        +qCO2*kCO2*nv_1D*nv_CO2 ) 
        Qvc[0] = Qvc[1]
        Qvc[-1] = Qvc[-2]
        # Qvc[0] = Qvc[-2]
        # Qvc[-1] = Qvc[1]
        
        Qv = Qvc.copy()
        
        Hv[:] += Qv[:,zi]*dt
        
        
        ntab_O3[it] = nv_O3c.copy()
        nv_1Sg = nv_1Sgc.copy()
        nv_3Sc = nv_3Sc.copy()
        nv_1Dtg = nv_1Dtgc.copy()
        nv_1D = nv_1Dc.copy()
        nv_O3 = nv_O3c.copy()
        Utabp[0,1:-1] = (Utab[0,2:] + Utab[0,1:-1])/2. - dt/dx/2.*(Ftab[0,2:]-Ftab[0,1:-1]) 
        Utabp[1,1:-1] = (Utab[1,2:] + Utab[1,1:-1])/2. - dt/dx/2.*(Ftab[1,2:]-Ftab[1,1:-1]) 
        Utabp[2,1:-1] = (Utab[2,2:] + Utab[2,1:-1])/2. - dt/dx/2.*(Ftab[2,2:]-Ftab[2,1:-1]) + Qv[1:-1,zi]*dt / (gamma-1) 


        Utabp[:,0] = Utabp[:,1] 
        Utabp[:,-1] = Utabp[:,-2] 
        # Utabp[:,0] = Utabp[:,-2] 
        # Utabp[:,-1] = Utabp[:,1] 
        
        Utabm[0,1:-1] = (Utab[0,:-2] + Utab[0,1:-1] )/2. - dt/dx/2.*(Ftab[0,1:-1]-Ftab[0,:-2]) 
        Utabm[1,1:-1] = (Utab[1,:-2] + Utab[1,1:-1] )/2. - dt/dx/2.*(Ftab[1,1:-1]-Ftab[1,:-2]) 
        Utabm[2,1:-1] = (Utab[2,:-2] + Utab[2,1:-1] )/2. - dt/dx/2.*(Ftab[2,1:-1]-Ftab[2,:-2]) + Qv[1:-1,zi]*dt / (gamma-1) 

        Utabm[:,0] = Utabm[:,1]
        Utabm[:,-1] = Utabm[:,-2] 
        # Utabm[:,0] = Utabm[:,-2] 
        # Utabm[:,-1] = Utabm[:,1]   
        
        
        Ftabp = upF(Utabp,nx) 
        Ftabm = upF(Utabm,nx) 
        

        Utabc[0,1:-1] =  Utab[0,1:-1] - dt/dx*(Ftabp[0,1:-1]-Ftabm[0,1:-1]) 
        Utabc[1,1:-1] = Utab[1,1:-1] - dt/dx*(Ftabp[1,1:-1]-Ftabm[1,1:-1]) 
        Utabc[2,1:-1] =  Utab[2,1:-1] - dt/dx*(Ftabp[2,1:-1]-Ftabm[2,1:-1]) + Qv[1:-1,zi]*dt / (gamma-1) 


        
        Utabc[:,0] = Utabc[:,1]
        Utabc[:,-1] = Utabc[:,-2]
        # Utabc[:,0] = Utabc[:,-2]
        # Utabc[:,-1] = Utabc[:,1]
    
        
        Ftab = upF(Utabc,nx)
        Utab = Utabc.copy()
        Uxt[:,it,:] = Utabc.copy()
        Fxt[:,it,:] = Ftab.copy()
        Qxt[it,:] = Qv[:,zi].copy()
        Hxt[it,:] = Hv.copy()        
        nhnuxt[it,:] = ntab_hnu[it,:,zi].copy()
        nO3xt[it,:] = ntab_O3[it,:,zi].copy()
        
    return (Uxt, Fxt, Qxt, nO3xt, nhnuxt,Hxt)

zmm = 0
zi = int(zmm*10**-3*nz/Lz)
print('z= ',zi,', [O3]=',n_O3v[0]/n_O2,'%')
R = Chem_Hydro_Eulersolver(zi,n_O3v[2],F0v[10])
print(n_O3v[3]/n_O2,F0v[16])
Uxt=R[0]
Fxt=R[1]
Qxt=R[2]
nO3xt=R[3]
nhnuxt=R[4]
Hxt=R[5]


def plotmap(Uxt,Fxt,Hxt=None,Qxt=None,figure=2):
    tt,xx=np.meshgrid(tv,xv,indexing='ij')
    plt.rcParams.update({'font.size': 70})
    fig = plt.figure(figure,figsize=[30,55])
    fig.clf()
    
    plt.subplots_adjust(hspace=0.5)  

    
#     ax0 = fig.add_subplot(511)  
#     ax0.ticklabel_format(axis='x', style='scientific', scilimits=(-3, 4))
#     ax0.ticklabel_format(axis='y', style='scientific', scilimits=(-3, 4))
#    divider = make_axes_locatable(ax0)
#     cf0 = ax0.pcolormesh(tt/2/np.pi,xx/law,Uxt[1,:,:]/Uxt[0,:,:], cmap='rainbow')#,vmin=0,vmax=0.001) 
#     cax = divider.append_axes("right",size= "3%",pad=0.1)
#     cf0 = plt.colorbar(cf0,ax=ax0,orientation="vertical",cax=cax)
#     cf0.formatter.set_powerlimits((-2, 2))
#     cf0.update_ticks()
#     cf0.set_label(r"$\Delta u / u$")
#     ax0.set_xlabel(r"$t \, / \, \tau_{ac}$")
# #     ax0.set_ylim(-2.5,2.5)
#     ax0.set_ylabel(r"$x \, / \, \Lambda_{ac}$ ")

    ax2 = fig.add_subplot(512)  
    ax2.ticklabel_format(axis='x', style='scientific', scilimits=(-3, 4))
    ax2.ticklabel_format(axis='y', style='scientific', scilimits=(-3, 4))
    divider = make_axes_locatable(ax2)
    cf2 = ax2.pcolormesh(tt/2/np.pi,xx/law,Fxt[1,:,:] - Uxt[1,:,:]*Uxt[1,:,:]/Uxt[0,:,:]-1, cmap='rainbow')#,vmin=0,vmax=0.05) 
    cax = divider.append_axes("right",size= "3%",pad=0.1)
    cf2 = plt.colorbar(cf2,ax=ax2,orientation="vertical",cax=cax)
    cf2.formatter.set_powerlimits((-2, 2))
    cf2.update_ticks()
    cf2.set_label(r"$\Delta P / P$")
    ax2.set_xlabel(r"$t \, / \, \tau_{ac}$")
#     ax2.set_ylim(-2.5,2.5)
    ax2.set_ylabel(r"$x \, / \, \Lambda_{ac}$ ")
    
    ax1 = fig.add_subplot(513)  
    ax1.ticklabel_format(axis='x', style='scientific', scilimits=(-3, 4))
    ax1.ticklabel_format(axis='y', style='scientific', scilimits=(-3, 4))
    divider = make_axes_locatable(ax1)
    cf1 = ax1.pcolormesh(tt/2/np.pi,xx/law,Uxt[0,:,:]-1, cmap='rainbow')#,vmax=1) 
    cax = divider.append_axes("right",size= "3%",pad=0.1)
    cf1 = plt.colorbar(cf1,ax=ax1,orientation="vertical",cax=cax)
    cf1.formatter.set_powerlimits((-2, 2))
    cf1.update_ticks()
    cf1.set_label(r"$\Delta \rho / \rho $")
    ax1.set_xlabel(r"$t \, / \, \tau_{ac}$")
#     ax1.set_ylim(-2.5,2.5)
    ax1.set_ylabel(r"$x \, / \, \Lambda_{ac}$ ")
    
    if Qxt is not None:
        ax3 = fig.add_subplot(514)  
        ax3.ticklabel_format(axis='x', style='scientific', scilimits=(-3, 4))
        ax3.ticklabel_format(axis='y', style='scientific', scilimits=(-3, 4))
        divider = make_axes_locatable(ax3)
        cf3 = ax3.pcolormesh(tt/2/np.pi,xx/law,Qxt, cmap='rainbow')#,vmin=-0.3,vmax=0.5) 
        cax = divider.append_axes("right",size= "3%",pad=0.1)
        cf3 = plt.colorbar(cf3,ax=ax3,orientation="vertical",cax=cax)
        cf3.formatter.set_powerlimits((-2, 2))
        cf3.update_ticks()
        cf3.set_label(r"$\Delta Q $")
        ax3.set_xlabel(r"$t \, / \, \tau_{ac}$")
        ax3.set_ylabel(r"$x \, / \, \Lambda_{ac}$ ")
#         ax3.set_ylim(-2.5,2.5)

        
    if Hxt is not None:
        
        ax4 = fig.add_subplot(515)  
        ax4.ticklabel_format(axis='x', style='scientific', scilimits=(-3, 4))
        ax4.ticklabel_format(axis='y', style='scientific', scilimits=(-3, 4))
        divider = make_axes_locatable(ax4)
        cf4 = ax4.pcolormesh(tt/2/np.pi,xx/law,Hxt, cmap='rainbow')#,vmin=-0.3,vmax=0.5) 
        cax = divider.append_axes("right",size= "3%",pad=0.1)
        cf4 = plt.colorbar(cf4,ax=ax4,orientation="vertical",cax=cax)
        cf4.formatter.set_powerlimits((-2, 2))
        cf4.update_ticks()
        cf4.set_label(r"$\langle \Delta T \rangle $")
        ax4.set_xlabel(r"$t \, / \, \tau_{ac}$")
        ax4.set_ylabel(r"$x \, / \, \Lambda_{ac}$ ")
#         ax4.set_ylim(-2.5,2.5)

    plt.show()

plotmap(Uxt,Fxt,Hxt,Qxt,figure=2)

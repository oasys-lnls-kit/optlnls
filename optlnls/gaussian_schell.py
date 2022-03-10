#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 15:29:25 2022

@author: lordano
"""

###### References:

# [1] Coherence properties of hard x-ray synchrotron sources and x-ray free-electron lasers
# [2] Coherence properties of focused X-ray beams at high-brilliance synchrotron sources


import numpy as np
from matplotlib import pyplot as plt
from optlnls.source import und_source


PLANCK = 4.135667433e-15; 
C = 2.99792458e+8;



#%%
def coherenceProperties(size, div, energy):
    
    d = dict()

    d['size'] = size    
    d['div'] = div
    d['energy'] = energy
    d['wl'] = PLANCK*C/energy #[m]
    d['wn'] = 2*np.pi/d['wl'] #[1/m]
    d['emm'] = size*div
    
    ### coherence length
    d['coh_len'] = 2*size/(np.sqrt(4*(d['wn']**2)*(d['emm']**2)-1)) # -----> Eq. 33, Ref. [1]
    # Source Degree of Coherence (constant for propagation in free space):
    d['q'] = d['coh_len']/size # -----> Eq. 34, Ref. [1]
    # Source Normalized Degree of Coherence  (constant for propagation in free space):
    d['coh_deg'] = d['q']/np.sqrt(4+d['q']**2) # -----> Eq. 35, Ref. [1]
    ### Rayleigh length
    d['delta'] = np.sqrt(1/((1/((2*size)**2))+(1/(d['coh_len']**2)))) # -----> Eq. 18, Ref. [1]
    d['zeff'] = d['wn']*size*d['delta'] # -----> Eq. 22, Ref. [1]
    
    return d
    
def printProperties(dictGS):
    
    d = dictGS
    
    print('\n')
    print('Size (sigma) [um]: \t\t\t\t %.2f' %(1e+6*d['size']))
    # print('Source Div (sigma) [urad]: \t\t %.2f' %(1e+6*d['div']))
    print('Transv coh length [um]: \t\t\t %.2f' %(1e+6*d['coh_len']))
    print('Degree of coherence (q): \t\t\t %.3f' %(d['q']))
    print('Normalized Degree of coh: \t\t %.3f' %(d['coh_deg']))
    print('Effective length (z_eff) [m]: \t\t %.2f' %(d['zeff']))
    print('\n')
    
    
def propagate(z, dictGS):

    d = dict()    
    d['wn'] = dictGS['wn']

    exp_fctr = (1 + (z / dictGS['zeff'])**2)**0.5 # -----> Eq. 20, Ref. [1]
    d['radius'] = z * (1 + (dictGS['zeff'] / z)**2) # -----> Eq. 21, Ref. [1]
    d['size'] = exp_fctr * dictGS['size'] # -----> Eq. 24, Ref. [1]
    d['coh_len'] = exp_fctr * dictGS['coh_len'] # -----> Eq. 28, Ref. [1]
    
    d['q'] = d['coh_len']/d['size'] # -----> Eq. 34, Ref. [1]
    d['coh_deg'] = d['q']/np.sqrt(4+d['q']**2) # -----> Eq. 35, Ref. [1]
    d['delta'] = np.sqrt(1/((1/((2*d['size'])**2))+(1/(d['coh_len']**2)))) # -----> Eq. 18, Ref. [1]
    d['zeff'] = d['wn']*d['size']*d['delta'] # -----> Eq. 22, Ref. [1]
 
    return d
    
def propagateAperture(diameter, dictGS):
    
    d = dictGS
    
    d['size'] = np.sqrt(1/(1/d['size']**2 + 1/diameter**2)) # -----> Eq. 24, Ref. [2], Size Right After Lens
    
    d['q'] = d['coh_len']/d['size'] # -----> Eq. 34, Ref. [1]
    d['coh_deg'] = d['q']/np.sqrt(4+d['q']**2) # -----> Eq. 35, Ref. [1]
    d['delta'] = np.sqrt(1/((1/((2*d['size'])**2))+(1/(d['coh_len']**2)))) # -----> Eq. 18, Ref. [1]
    d['zeff'] = d['wn']*d['size']*d['delta'] # -----> Eq. 22, Ref. [1]

    return d
    

def propagateLens(f, dictGS):
    
    dictGS['radius'] =  1 / (1/dictGS['radius'] - 1/f)
    return dictGS



#%%

if __name__ == '__main__':


    energy = 3000.0
    harmonic = 1
    
    size, div = und_source(emmitance=250e-12, 
                           beta=1.5, 
                           e_spread=0.085e-2, 
                           und_length=2.4, 
                           und_period=21, 
                           ph_energy=energy, 
                           harmonic=harmonic)
    
    size *= 1e-6
    div *= 1e-6
    
    
    d = coherenceProperties(size, div, energy)
    printProperties(d)
    
    z = 10.0
    
    d = propagate(z, d)
    
    printProperties(d)


#%%
















#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 15:29:25 2022

@author: lordano
"""

###### References:

# [1] A. Vartanyants and A. Singer, 2010 New J. Phys. 12 035004
# [2] A. Vartanyants and A. Singer, 2014 J. Synchrotron Rad. 21, 5-15


import numpy as np
from matplotlib import pyplot as plt
from optlnls.source import und_source
from copy import deepcopy
from scipy.special import erf

PLANCK = 4.135667433e-15; 
C = 2.99792458e+8;

#%% CLASS IMPLEMENTATION OF THE GAUSSIAN-SCHELL EQUATIONS
### ** the class implementation is more up-to-date than the dict one 
class GSbeam(object):
    """Ray Element"""
    
    def __init__(self, _energy=12000, _size_rms=20e-6, 
                 _coh_len=10e-6, _radius=0, _z=0, _intensity=1.0):
        
        self.energy = _energy
        self.size_rms = _size_rms
        self.coh_len = _coh_len
        self.radius = _radius
        self.z = _z
        self.intensity = _intensity
        self.update_wavelength()
        self.calc_coherence()

    def update_wavelength(self):
        
        self.wavelength = PLANCK * C / self.energy
        self.wavenumber = 2 * np.pi / self.wavelength
        

    def calc_coherence(self):

        # Source Degree of Coherence (constant for propagation in free space):
        self.q = self.coh_len / self.size_rms # -----> Eq. 34, Ref. [1]
        # Source Normalized Degree of Coherence  (constant for propagation in free space):
        self.coh_deg = self.q / np.sqrt( 4 + self.q**2) # -----> Eq. 35, Ref. [1]
        ### Rayleigh length
        delta = np.sqrt(1 / ((1/((2*self.size_rms)**2)) + (1/(self.coh_len**2)))) # -----> Eq. 18, Ref. [1]
        self.z_eff = self.wavenumber * self.size_rms * delta # -----> Eq. 22, Ref. [1]
        
    def undulator_source(self, _size_rms=20e-6, _div_rms=20e-6, _energy=12000):

        self.energy = _energy
        self.update_wavelength()
        self.z = 0
        emittance = _size_rms * _div_rms
        self.radius = np.infty
        self.size_rms = _size_rms
        
        ### coherence length
        self.coh_len = 2 * self.size_rms / (np.sqrt( 4*(self.wavenumber**2)*(emittance**2) - 1) ) # -----> Eq. 33, Ref. [1]
        
        self.calc_coherence()
        
    def propagate_from_source(self, distance):
        
        exp_fctr = (1 + (distance / self.z_eff)**2)**0.5 # -----> Eq. 20, Ref. [1]
        self.radius = distance * (1 + (self.z_eff / distance)**2) # -----> Eq. 21, Ref. [1]
        self.size_rms = exp_fctr * self.size_rms # -----> Eq. 24, Ref. [1]
        self.coh_len = exp_fctr * self.coh_len # -----> Eq. 28, Ref. [1]
        self.z = self.z + distance
        
        self.calc_coherence()
        
    def propagate_aperture(self, aperture):
        
        self.intensity = self.intensity * erf(aperture/self.size_rms / (2 * np.sqrt(2)))
        self.size_rms = np.sqrt(1 / (1/self.size_rms**2 + 1/aperture**2)) # -----> Eq. 24, Ref. [2], Size Right After Lens
        self.calc_coherence()
        
    def propagate_focusing_lens(self, f):
        
        self.radius = 1 / (1/self.radius - 1/f)
        
    def get_focus_position(self):
        
        if(self.radius < 0):
            zf = - self.radius / (1 + (self.radius/self.z_eff)**2)
        else:
            zf = 0
        return zf
    
    def propagate_to_focus(self):
        
        divisor = np.sqrt(1 + (self.z_eff / self.radius)**2)
        self.size_rms = self.size_rms / divisor
        self.coh_len = self.coh_len / divisor
        focus_z = self.get_focus_position()
        self.z = self.z + focus_z
        self.calc_coherence()
        self.radius = np.infty
        
    def get_spectral_density(self, x, normalized=1):
        
        if(normalized):
            amplitude = self.intensity / (self.size_rms * np.sqrt(2*np.pi))
        else:
            amplitude = 1
            
        return amplitude * np.exp(-x**2/(2*self.size_rms**2))
        
        
    def print_properties(self, _label=''):
                
        print('\n')
        print(_label)
        print('Position [m]: \t\t\t\t\t %.2f' %(self.z))
        print('Size (sigma) [um]: \t\t\t\t %.2f' %(self.size_rms*1e6))
        print('Transv coh length [um]: \t\t\t %.2f' %(self.coh_len*1e6))
        print('Degree of coherence (q): \t\t\t %.3f' %(self.q))
        print('Normalized Degree of coh: \t\t %.3f' %(self.coh_deg))
        print('Effective length (z_eff) [m]: \t\t %.2f' %(self.z_eff))
        print('Radius of curvature [m]: \t\t\t %.2f' %(self.radius))
        print('\n')

    def get_copy(self):
        return deepcopy(self)

def test_GSbeam():
    
    energy = 9000.0
    harmonic = 1
    
    size, div = und_source(emmitance=1 * 250e-12, 
                           beta=1.5, 
                           e_spread=0.085e-2, 
                           und_length=2.4, 
                           und_period=21, 
                           ph_energy=energy, 
                           harmonic=harmonic)
    
    size *= 1e-6
    div *= 1e-6
    
    #######################
    ### GS class propagation
    
    beam = GSbeam()
    beam.undulator_source(_size_rms=size,
                          _div_rms=div,
                          _energy=energy)

    beam.print_properties(_label='AT SOURCE POSITION')
    
    beam.propagate_from_source(distance=57.0)
    beam.print_properties(_label='AFTER 57 M')        
    
    beam.propagate_aperture(aperture=140e-6/4.55)        
    beam.print_properties(_label='AFTER APERTURE')
    

    f = 28
    beam.propagate_focusing_lens(f=f)
    beam.print_properties(_label='AFTER LENS')
    
    beam.propagate_to_focus()
    beam.print_properties(_label='AT FOCUS')
    print('focus position = ', beam.focus_z)    

    x = np.linspace(-100, 100, 1000)*1e-6
    intensity = beam.get_spectral_density(x)
    
    plt.figure()
    plt.plot(x*1e6, intensity)

#%% DICT IMPLEMENTATION OF THE GAUSSIAN-SCHELL EQUATIONS
### ** the dict implementation is deprecated. The GSbeam class is recommended. 
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
    
def printProperties(dictGS, label=''):
    
    d = dictGS
    
    print('\n')
    print(label)
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

def test_GSdict():
    
    energy = 9000.0
    harmonic = 1
    
    size, div = und_source(emmitance=1 * 250e-12, 
                           beta=1.5, 
                           e_spread=0.085e-2, 
                           und_length=2.4, 
                           und_period=21, 
                           ph_energy=energy, 
                           harmonic=harmonic)
    
    size *= 1e-6
    div *= 1e-6
    
    d = coherenceProperties(size, div, energy)
    print('source properties')
    printProperties(d)
    
    
    ################################
    ### propagating 10 m downstream
        
    z = 10.0
    
    d = propagate(z, d)
    print('beam at z = {0} m'.format(z))
    printProperties(d)

    ################################
    ### including an aperture
    
    aperture = 100e-6
    
    d = propagateAperture(diameter=aperture, dictGS=d)
    print('after aperture')
    printProperties(d) 


    ################################
    ### including a lens
    
    f = 5
    
    d = propagateLens(f=f, dictGS=d)
    print('after lens')
    printProperties(d) 


#%%

# if __name__ == '__main__':


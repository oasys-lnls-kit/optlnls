#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 13:52:00 2022

@author: sergio.lordano
"""


import numpy as np
import xraylib as xrl


class FZP(object):
    
    def __init__(self,
                 material : str = 'Au', 
                 energy : float = 10e3,
                 f : float = 0.1,
                 resolution : float = 200e-9,
                 thickness : float = 0):
  
        self.material = material
        self.energy = energy
        self.f = f
        self.resolution = resolution
        self.wavelength = 1.239842e-6 / energy
        self.k = 2 * np.pi / self.wavelength
        
        self.Z = xrl.SymbolToAtomicNumber(self.material)
        self.density = xrl.ElementDensity(self.Z)
        
        self.delta = 1 - xrl.Refractive_Index_Re(compound=material, E=energy*1e-3, density=self.density)
        self.beta = xrl.Refractive_Index_Im(compound=material, E=energy*1e-3, density=self.density)
        self.mu = 4 * np.pi * self.beta / self.wavelength
        self.attenuationLength = 1 / self.mu
        
        
        #### FZP equations
        
        self.rn = self.resolution / 1.22
        self.N_zones = int(self.wavelength * self.f / (4 * self.rn**2))
        self.diameter = 4 * self.N_zones * self.rn
        self.numerical_aperture = self.wavelength / (2 * self.rn)
        self.dof = self.wavelength / (2 * self.numerical_aperture**2)

        if(thickness == 0):
            self.thickness = self.wavelength / (2 * self.delta)
        else:
            self.thickness = thickness

        self.aspect_ratio = self.thickness / self.resolution
    

    def transmission1D(self, 
                       r : np.ndarray = np.linspace(0, 0.5e-3, 10000),
                       beam_stop : bool = False):
        
        ### REF: Vela-Comamala et al, J. synchrotron Rad. 20, 397-404 (2013)
        
        self.r = r
        wl = self.wavelength
        f = self.f
        
        amp = np.exp(1j * self.k * (self.delta + 1j*self.beta) * self.thickness)
        
        tr = np.ones(len(self.r), dtype=complex)
        
        n = np.arange(0, self.N_zones+1)
        
        n1 = n[::2]
        n2 = n[1::2]
        
        for i in range(len(n2)):
            
            c1_odd = np.sqrt(n1[i] * wl * f + (n1[i] * wl / 2)**2)
            c2_odd = np.sqrt(n2[i] * wl * f + (n2[i] * wl / 2)**2)
            
            r_odd = np.logical_and(r > c1_odd, r <= c2_odd)
            
            if((i == 0) & (beam_stop)):
                tr[r_odd] = 0
            else:
                tr[r_odd] = amp #                 
        
        ## external part of the FZP
        tr[r >= self.diameter/2] = 0 
        
        ## fix for the first value
        if(beam_stop):
            tr[0] = 0
        else:
            tr[0] = amp
                              
        return tr
            
    
if __name__ == '__main__':
    
    from plot import plot_xy_list
    
    fzp = FZP(resolution=100e-9, material='Au')
    
    px = 10e-9
    rmax = 1.5*fzp.diameter/2
    nr = int(rmax/px + 1)
    
    r = np.linspace(0, rmax, nr)
    
    fzp1D = fzp.transmission1D(r, True)
    
    fzp1D_intensity = np.abs(fzp1D)**2
    fzp1D_phase = np.angle(fzp1D)
    
    plot_xy_list(x=[r*1e6], y=[fzp1D_intensity])
    plot_xy_list(x=[r*1e6], y=[fzp1D_phase])    
    
            
            
            
        
        
        
    
    







































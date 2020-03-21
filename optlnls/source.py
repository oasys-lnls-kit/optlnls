#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 15:26:15 2020

@author: sergio.lordano
"""

import numpy as np
from matplotlib import pyplot as plt
import scipy.special as spc

def get_k(Period, what_harmonic, Energy, k_ext):
    E = 3.0e9; e = 1.60217662e-19; m_e = 9.10938356e-31; pi = 3.141592654; c = 299792458; h_cut = 6.58211915e-16;
    gamma = E*e/(m_e*c**2)
    n = [2*i+1 for i in range(50)]
    for h_n in n:
        K2 =  (8*h_n*pi*h_cut*c*(gamma**2)/Period/Energy-2)
        if K2 > 0:
            if what_harmonic == 'max':            
                if K2**0.5 < k_ext:
                    har = h_n
                    k = K2**0.5
                else:                 
                    #print("Maximum Harmonic = ", har, ', K = ', k)
                    break
            elif what_harmonic == 'min':
                if K2**0.5 > k_ext:
                    har = h_n
                    k = K2**0.5
                    #print("Minimum Harmonic = ", har, ', K = ', k)
                    break     
    B = 2*pi*m_e*c*k/(e*Period)
    return har, k, B


def und_source(emmitance, Beta, e_spread, und_length, und_period, ph_energy, harmonic):
#    import scipy.special as spc
    
    emmitance = emmitance*1e12          #Vertical Emmitance [pm.rad]
    und_period = und_period*1e2         #Undulator period length[cm]
    und_length = und_length*1e2         #Undulator Length [cm]
    
    x = 2*np.pi*harmonic*und_length/und_period*e_spread
    a1 = np.sqrt(2*np.pi)*x*spc.erf(np.sqrt(2)*x)
    Qax = np.sqrt(2*x**2/(-1+np.exp(-2*x**2)+a1))
    sphp_h = np.sqrt(emmitance/Beta+(12.398e7/ph_energy)/(2*und_length)*Qax**2)             #Divergence
    
    a1s = np.sqrt(2*np.pi)*(x/4)*spc.erf(np.sqrt(2)*(x/4))
    Qas = (np.sqrt(2*(x/4)**2/(-1+np.exp(-2*(x/4)**2)+a1s)))**(2./3.)
    sph_h = np.sqrt(emmitance*Beta+und_length*12398/(2*(np.pi**2)*ph_energy)*(Qas)**2)      #Size
    
    return sph_h, sphp_h


def srw_undulator_spectrum(mag_field=[], electron_beam=[], energy_grid=[], sampling_mesh=[], precision=[]):
    """
    Calls SRW to calculate spectrum for a planar or elliptical undulator\n
    :mag_field: list containing: [period [m], length [m], Bx [T], By [T], phase Bx = 0, phase By = 0, Symmetry Bx = +1, Symmetry By = -1]
    :electron_beam: list containing: [Sx [m], Sy [m], Sx' [rad], Sy'[rad], Energy [GeV], Energy Spread [dE/E], Current [A]]
    :energy_grid: list containing: [initial energy, final energy, number of energy points]
    :sampling_mesh: list containing: [observation plane distance from source [m], range X [m], range Y [m]]
    :precision: list containing: [h_max: maximum harmonic number to take into account, longitudinal precision factor, azimuthal precision factor (1 is standard, >1 is more accurate]
    """    
        
    from srwlib import SRWLMagFldU, SRWLMagFldH, SRWLPartBeam, SRWLStokes
    from srwlpy import CalcStokesUR
    from numpy import array as nparray

    #***********Undulator
    und = SRWLMagFldU([SRWLMagFldH(1, 'v', mag_field[3], mag_field[5], mag_field[7], 1), 
                       SRWLMagFldH(1, 'h', mag_field[2], mag_field[4], mag_field[6], 1)], 
                       mag_field[0], int(round(mag_field[1]/mag_field[0])))
       
    #***********Electron Beam
    eBeam = SRWLPartBeam()
    eBeam.Iavg = electron_beam[6] #average current [A]
    eBeam.partStatMom1.x = 0. #initial transverse positions [m]
    eBeam.partStatMom1.y = 0.
    eBeam.partStatMom1.z = -(mag_field[1]/2 + mag_field[0]*2) #initial longitudinal positions (set in the middle of undulator)
    eBeam.partStatMom1.xp = 0 #initial relative transverse velocities
    eBeam.partStatMom1.yp = 0
    eBeam.partStatMom1.gamma = electron_beam[4]/0.51099890221e-03 #relative energy
    sigEperE = electron_beam[5] #0.00089 #relative RMS energy spread
    sigX = electron_beam[0] #33.33e-06 #horizontal RMS size of e-beam [m]
    sigXp = electron_beam[2] #16.5e-06 #horizontal RMS angular divergence [rad]
    sigY = electron_beam[1] #2.912e-06 #vertical RMS size of e-beam [m]
    sigYp = electron_beam[3] #2.7472e-06 #vertical RMS angular divergence [rad]
    #2nd order stat. moments:
    eBeam.arStatMom2[0] = sigX*sigX #<(x-<x>)^2> 
    eBeam.arStatMom2[1] = 0 #<(x-<x>)(x'-<x'>)>
    eBeam.arStatMom2[2] = sigXp*sigXp #<(x'-<x'>)^2> 
    eBeam.arStatMom2[3] = sigY*sigY #<(y-<y>)^2>
    eBeam.arStatMom2[4] = 0 #<(y-<y>)(y'-<y'>)>
    eBeam.arStatMom2[5] = sigYp*sigYp #<(y'-<y'>)^2>
    eBeam.arStatMom2[10] = sigEperE*sigEperE #<(E-<E>)^2>/<E>^2
    
    #***********Precision Parameters
    arPrecF = [0]*5 #for spectral flux vs photon energy
    arPrecF[0] = precision[0] #initial UR harmonic to take into account
    arPrecF[1] = precision[1] #final UR harmonic to take into account
    arPrecF[2] = precision[2] #longitudinal integration precision parameter
    arPrecF[3] = precision[3] #azimuthal integration precision parameter
    arPrecF[4] = 1 #calculate flux (1) or flux per unit surface (2)
        
    #***********UR Stokes Parameters (mesh) for Spectral Flux
    stkF = SRWLStokes() #for spectral flux vs photon energy
    stkF.allocate(energy_grid[2], 1, 1) #numbers of points vs photon energy, horizontal and vertical positions
    stkF.mesh.zStart = sampling_mesh[0] #longitudinal position [m] at which UR has to be calculated
    stkF.mesh.eStart = energy_grid[0] #initial photon energy [eV]
    stkF.mesh.eFin = energy_grid[1] #final photon energy [eV]
    stkF.mesh.xStart = -sampling_mesh[1]/2.0 #initial horizontal position [m]
    stkF.mesh.xFin = sampling_mesh[1]/2.0 #final horizontal position [m]
    stkF.mesh.yStart = -sampling_mesh[2]/2.0 #initial vertical position [m]
    stkF.mesh.yFin = sampling_mesh[2]/2.0 #final vertical position [m]
           
    
    #**********************Calculation (SRWLIB function calls)
    print('   Performing Spectral Flux (Stokes parameters) calculation ... ')
    CalcStokesUR(stkF, eBeam, und, arPrecF)
    print('done')
    
    return nparray(stkF.arS[0:energy_grid[2]])





















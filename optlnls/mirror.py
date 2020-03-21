#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 12:46:58 2020

@author: sergio.lordano
"""

import sys
import os
import numpy as np
from scipy.interpolate import interp1d
#from matplotlib import pyplot as plt

# ======= MATERIALS ================ #
#
# Au:   A=196.967   ro=19.3    Z=79
# Rh:   A=102.9     ro=12.41   Z=45
# Pt:   A=195.08    ro=21.45   Z=78
# Si:   A=28.09     ro=2.32    Z=14
# Cr:   A=51.996    ro=7.18    Z=24
# Ni:   A=58.69     ro=8.9     Z=28
# ================================== #

def reflectivity_xrays(material, density, atomic_mass, energy_eV, angle_normal_deg, folder=''):
    # Reference: 2001, Elements of Modern X-ray Physics, Als Nielsen, section 3.4
    r0 = 2.82e-15 # [m];
    ro = density # [g/cm3]
    A = atomic_mass # [g/mol]
    Na = 6.022e23

    if(folder == ''):
        optlnls_path = __file__
        optlnls_path = optlnls_path.split(os.path.sep)[:-1]
        optlnls_path = os.path.sep.join(optlnls_path)
        f1_file = os.path.join(optlnls_path, 'materials', material+'_f1_nist.txt')
        f2_file = os.path.join(optlnls_path, 'materials', material+'_f2_nist.txt')
        # print(f1_file)
        # print(__file__)
    else:
        f1_file = material+'_f1_nist.txt'
        f2_file = material+'_f2_nist.txt'
    
    f1_nist = np.genfromtxt(f1_file, dtype='float',skip_header=2)
    f2_nist = np.genfromtxt(f2_file, dtype='float',skip_header=2)
    f1_nist_ip = interp1d(f1_nist[:,0]*1e3, f1_nist[:,1])
    f2_nist_ip = interp1d(f2_nist[:,0]*1e3, f2_nist[:,1])
    angle = (90 - angle_normal_deg)*np.pi/180.0
    wl = 1239.842/energy_eV*1e-9 # [m]
    delta_h = (r0/(2*np.pi))*(wl**2)*(ro*1e6/A*Na)*f1_nist_ip(energy_eV)
    beta_h = (r0/(2*np.pi))*(wl**2)*(ro*1e6/A*Na)*f2_nist_ip(energy_eV)
    Qcm = (4*np.pi/wl)*(2*delta_h)**0.5
    b_mu = beta_h/(2*delta_h)
    Qsm = (4*np.pi/wl)*np.sin(angle)
    Qpm = (Qsm**2 - (Qcm**2)*(1-1j*2*b_mu))**0.5
    return np.abs((Qsm-Qpm)/(Qsm+Qpm))**2  

def amplitude_reflectivity_xrays(material, density, atomic_mass, energy_eV, angle_normal_deg, folder=''):
    # Reference: 2001, Elements of Modern X-ray Physics, Als Nielsen, section 3.4
    r0 = 2.82e-15 # [m];
    ro = density # [g/cm3]
    A = atomic_mass # [g/mol]
    Na = 6.022e23
    
    if(folder != ''):
        this_file_path = sys.argv[0]
        f1_file = os.path.join(this_file_path, 'materials', material+'_f1_nist.txt')
        f2_file = os.path.join(this_file_path, 'materials', material+'_f2_nist.txt')
    else:
        f1_file = material+'_f1_nist.txt'
        f2_file = material+'_f2_nist.txt'
    
    f1_nist = np.genfromtxt(f1_file, dtype='float',skip_header=2)
    f2_nist = np.genfromtxt(f2_file, dtype='float',skip_header=2)
    f1_nist_ip = interp1d(f1_nist[:,0]*1e3, f1_nist[:,1])
    f2_nist_ip = interp1d(f2_nist[:,0]*1e3, f2_nist[:,1])
    angle = angle_normal_deg*np.pi/180.0
    wl = 1239.842/energy_eV*1e-9 # [m]
    delta_h = (r0/(2*np.pi))*(wl**2)*(ro*1e6/A*Na)*f1_nist_ip(energy_eV)
    beta_h = (r0/(2*np.pi))*(wl**2)*(ro*1e6/A*Na)*f2_nist_ip(energy_eV)
    Qcm = (4*np.pi/wl)*(2*delta_h)**0.5
    b_mu = beta_h/(2*delta_h)
    Qsm = (4*np.pi/wl)*np.sin(angle)
    Qpm = (Qsm**2 - (Qcm**2)*(1-1j*2*b_mu))**0.5
    return (Qsm-Qpm)/(Qsm+Qpm)




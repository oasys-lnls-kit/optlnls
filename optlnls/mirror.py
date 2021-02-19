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
from matplotlib import pyplot as plt

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


def read_IMD(filename, wl_range=[0,0]):
    
    wl, n, k = np.genfromtxt(filename, comments=';', unpack=True)
    wl *= 1e-4
    nc = n + 1j*k
    
    if(wl_range != [0,0]):
        wl_min = wl_range[0]
        wl_max = wl_range[1]
        if(wl_max <= 0.0):
            wl_max = 1e20

        nc = nc[(wl <= wl_max) & (wl >= wl_min)]       
        wl = wl[(wl <= wl_max) & (wl >= wl_min)]
    
    return wl, nc

def read_RefractiveIndexInfo(filename, wl_range=[0,0]):

    n = []
    k = []

    j=0

    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split(',')
            try:
                line = np.array(line, dtype=np.float)
                if(j==1): 
                    n.append(line.tolist())
                else:
                    k.append(line.tolist())
            except:
                j += 1
                
    n = np.array(n)
    k = np.array(k)
    
    if(len(k)) == 0: # Some files do not have k. Consider equal zero instead.
        k = np.array([n[:,0], [0.0]*len(n)]).transpose()

    # check if wavelengths for n, k are the same
    if all(n[:,0] == k[:,0]):
        wl = n[:,0]
        nc = n[:,1] + 1j*k[:,1]
        
    else:
        # implement interpolation option
        print("wavelength points are different for 'n' and 'k'! ")
        return 0
                
    if(wl_range != [0,0]):
        wl_min = wl_range[0]
        wl_max = wl_range[1]
        if(wl_max <= 0.0):
            wl_max = 1e20

        nc = nc[(wl <= wl_max) & (wl >= wl_min)]       
        wl = wl[(wl <= wl_max) & (wl >= wl_min)]
    
        
    return wl, nc
    
	
def fresnel_reflectivity(n1, n2, theta_surface_deg):
    
    n1 = n1.astype(np.complex) if isinstance(n1, (np.ndarray)) else complex(n1)
    n2 = n2.astype(np.complex) if isinstance(n2, (np.ndarray)) else complex(n2)

    θi = np.deg2rad(90-theta_surface_deg) # incidence angle (radians)
    θt = np.arcsin(n1/n2*np.sin(θi)) # refraction angle (radians)

    rs = (n1*np.cos(θi)-n2*np.cos(θt)) / (n1*np.cos(θi)+n2*np.cos(θt))
    rp = (n2*np.cos(θi)-n1*np.cos(θt)) / (n1*np.cos(θt)+n2*np.cos(θi))

    Rs = np.abs(rs)**2
    Rp = np.abs(rp)**2
    Runpol = (Rs + Rp)/2
    
    #Psi_s = np.rad2deg(np.angle(rs))
    #Psi_p = np.rad2deg(np.angle(rp))

    return Rs, Rp, Runpol




def optical_properties(filelist, theta, wl_range=[0,0]):
    
    Rs = []
    Rp = []
    Ru = []
    n  = []
    k  = []
    wavelength = []

    for filename in filelist:
        
        fname, ext = os.path.splitext(filename)
        
        if(ext == '.csv'):
            wl, n2 = read_RefractiveIndexInfo(filename)
        elif(ext == '.nk'):
            wl, n2 = read_IMD(filename)
            
        if(wl_range != [0,0]):
            wl_min = wl_range[0]
            wl_max = wl_range[1]
            if(wl_max <= 0.0):
                wl_max = 1e20

            n2 = n2[(wl <= wl_max) & (wl >= wl_min)]       
            wl = wl[(wl <= wl_max) & (wl >= wl_min)]
     
        R_s, R_p, R_u = fresnel_reflectivity(n1=1, n2=n2, theta_surface_deg=theta)
        Rs.append(R_s)
        Rp.append(R_p)
        Ru.append(R_u)
        wavelength.append(wl)
        n.append(n2.real)
        k.append(n2.imag)

    return wavelength, n, k, Rs, Rp, Ru
        
def plot_nk(wavelength, n, k, filelist, prefix):
    


    plt.figure(figsize=(4.5,3))
    plt.subplots_adjust(0.15, 0.15, 0.95, 0.85)
    for i, filename in enumerate(filelist):
        fname, ext = os.path.splitext(filename)
        label = fname.split('/')[-1]
        energy = 1.239842/wavelength[i]
        plt.plot(energy, n[i], marker='.', label=label)
    plt.xlabel('Energy [eV]')
    plt.ylabel('Refraction index (n)')
    plt.xlim(1, 1e2)
    #plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc='best', fontsize=6)
    plt.minorticks_on()
    plt.grid(which='both', alpha=0.2)
    plt.tick_params(which='both', axis='both', direction='in', top=False, right=True)
    #ax = plt.gca()
    #secax = ax.secondary_xaxis('top', functions=(um2eV, eV2um))
    #secax.set_xlabel('Energy [eV]')
    plt.savefig(prefix + '_refrIndex.png', dpi=1200)

    plt.figure(figsize=(4.5,3))
    plt.subplots_adjust(0.15, 0.15, 0.95, 0.85)
    for i, filename in enumerate(filelist):
        fname, ext = os.path.splitext(filename)
        label = fname.split('/')[-1]
        energy = 1.239842/wavelength[i]
        plt.plot(energy, k[i], marker='.', label=label)
    plt.xlabel('Energy [eV]')
    plt.ylabel('Extinction Coefficient (\u03BA)')
    plt.xlim(1, 1e2)
    #plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc='best', fontsize=6)
    plt.minorticks_on()
    plt.grid(which='both', alpha=0.2)
    plt.tick_params(which='both', axis='both', direction='in', top=False, right=True)
    #ax = plt.gca()
    #secax = ax.secondary_xaxis('top', functions=(um2eV, eV2um))
    #secax.set_xlabel('Energy [eV]')
    plt.savefig(prefix + '_extiCoeff.png', dpi=1200)

def plot_reflectances(wavelength, Rs, Rp, filelist, theta, prefix):
    
    plt.figure(figsize=(4.5,3))
    plt.subplots_adjust(0.15, 0.15, 0.95, 0.85)
    for i, filename in enumerate(filelist):
        fname, ext = os.path.splitext(filename)
        label = fname.split('/')[-1]
        energy = 1.239842/wavelength[i]
        plt.plot(energy, Rs[i], marker='.', label=label)
    # plt.title('Au Reflectance (s-pol) at {0:.1f} deg'.format(theta))
    plt.xlabel('Energy [eV]')
    plt.ylabel('Reflectance  \u03C3-pol  (\u03B8  = %.1f \u00b0)' % theta)
    plt.ylim(-0.05, 1.05)
    plt.xlim(1, 1e2)
    #plt.xscale('log')
    plt.legend(loc='best', fontsize=6)
    plt.minorticks_on()
    plt.grid(which='both', alpha=0.2)
    plt.tick_params(which='both', axis='both', direction='in', top=False, right=True)
    #ax = plt.gca()
    #secax = ax.secondary_xaxis('top', functions=(um2eV, eV2um))
    #secax.set_xlabel('Energy [eV]')
    plt.savefig(prefix + '_refl_S.png', dpi=1200)

    plt.figure(figsize=(4.5,3))
    plt.subplots_adjust(0.15, 0.15, 0.95, 0.85)
    for i, filename in enumerate(filelist):
        fname, ext = os.path.splitext(filename)
        label = fname.split('/')[-1]
        energy = 1.239842/wavelength[i]
        plt.plot(energy, Rp[i], marker='.', label=label)
    # plt.title('Au Reflectance (p-pol) at {0:.1f} deg'.format(theta))
    plt.xlabel('Energy [eV]')
    plt.ylabel('Reflectance  \u03C0-pol  (\u03B8  = %.1f \u00b0)' % theta)
    plt.ylim(-0.05, 1.05)
    plt.xlim(1, 1e2)
    #plt.xscale('log')
    plt.legend(loc='best', fontsize=6)
    plt.minorticks_on()
    plt.grid(which='both', alpha=0.2)
    plt.tick_params(which='both', axis='both', direction='in', top=False, right=True)
    #ax = plt.gca()
    #secax = ax.secondary_xaxis('top', functions=(um2eV, eV2um))
    #secax.set_xlabel('Energy [eV]')
    plt.savefig(prefix + '_refl_P.png', dpi=1200)

    plt.show()
    
    
def write_refl_files(wavelength, Rs, Rp, prefix):
    
    from optlnls.math import common_region_average
    
    lists_Rs=[]
    lists_Rp=[]
    for i in range(len(wavelength)):
        if(len(wavelength[i])>0):
            lists_Rs.append([wavelength[i], Rs[i]])
            lists_Rp.append([wavelength[i], Rp[i]])

    wavelength_Rs_avg, Rs_avg = common_region_average(lists_Rs)
    wavelength_Rp_avg, Rp_avg = common_region_average(lists_Rp)

    np.savetxt(prefix+'_Rs_avg.txt', np.array([wavelength_Rs_avg, Rs_avg]).transpose(), header='wl[um],Rs', fmt='%.4e')
    np.savetxt(prefix+'_Rp_avg.txt', np.array([wavelength_Rp_avg, Rp_avg]).transpose(), header='wl[um],Rp', fmt='%.4e')
    
def write_average_nk(wavelength, n, k, filename, step=0):

    from optlnls.math import common_region_average
    
    lists_n=[]
    lists_k=[]
    for i in range(len(wavelength)):
        if(len(wavelength[i])>0):
            lists_n.append([wavelength[i], n[i]])
            lists_k.append([wavelength[i], k[i]])

    wavelength_n_avg, n_avg = common_region_average(lists_n, step)
    wavelength_k_avg, k_avg = common_region_average(lists_k, step)
    
    if((wavelength_n_avg == wavelength_k_avg).all()):
        
        with open(filename, 'w+') as f:
            f.write('wl,n \n')
            for i in range(len(wavelength_n_avg)):
                f.write('{0:.8f},{1:.8f}\n'.format(wavelength_n_avg[i],n_avg[i]))
            f.write('\n')
            f.write('wl,k \n')
            for i in range(len(wavelength_n_avg)):
                f.write('{0:.8f},{1:.8f}\n'.format(wavelength_n_avg[i],k_avg[i]))
    
    
def calc_reflectivity_fresnel(energy, theta, input_file='Si.nk', unpolarized=0):
    
    wavelength, n, k, Rs, Rp, Ru = optical_properties([input_file], theta)
    
    Rs_interp = interp1d(wavelength[0], Rs[0], kind='linear')
    Rp_interp = interp1d(wavelength[0], Rp[0], kind='linear')
    
    if(isinstance(energy, (int, float))):
        
        refl = np.array([Rs_interp(eV2um(energy)), Rp_interp(eV2um(energy))])
        
    else:
        ne = len(energy)
    
        refl = np.zeros((2, ne))
        
        for i in range(ne):
            wl = eV2um(energy[i])
            refl[:,i] = [Rs_interp(wl), Rp_interp(wl)]
        
    if(unpolarized):
        return np.average(refl, axis=0)
    
    else:
        return refl
    
	
def um2eV(wl):
    return 1.23984198433/wl
        
def eV2um(e):
    return 1.23984198433/e	
	

def test_cedro_refl():
    
    filelist = ['inputs/Si.nk']
    
    theta = 45  
    
    wavelength, n, k, Rs, Rp, Ru = optical_properties(filelist, theta)
    
    energy = []
    for i in range(len(filelist)):
        energy.append(um2eV(wavelength[i]))
    
    
    array_to_save = np.array([energy[0], Ru[0], Rs[0], Rp[0]]).transpose()
    np.savetxt('Si_refl_45deg_fresnel_eq.txt', array_to_save, fmt='%.6e', delimiter='\t')


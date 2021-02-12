#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 12:47:43 2020

@author: sergio.lordano
"""

# Library:

import xraylib
from scipy.constants import physical_constants 
import matplotlib.pyplot as plt
import numpy as np
    

# Functions:

def calc_Bragg_angle(crystal='Si', energy=8, h=1, k=1, l=1, corrected=False):
    
    '''
    Calculates Bragg angle using xraylib.
    
    Parameters:
        
    - crystal: crystal material. [str]
    - energy: energy in keV. [float]
    - h, k, l: Miller indexes. [int]
    
    Returns:
    
    - bragg: Bragg angle in rad. [float] 
    
    '''
    
    cryst = xraylib.Crystal_GetCrystal(crystal)

    bragg = xraylib.Bragg_angle(cryst, energy, h, k, l)

    if(corrected):
        w0 = calc_rocking_curve_shift(crystal=crystal, energy=energy, h=h, k=k, l=l, rel_angle=1, debye_temp_factor=1)
        bragg += w0
    
    return bragg


def calc_Darwin_width(crystal='Si', energy=8, h=1, k=1, l=1, rel_angle=1, debye_temp_factor=1):
    
    '''
    Calculates Darwin width and intrinsic resolution for p and s polarizations. It uses xraylib.
    
    Parameters:
        
    - crystal: crystal material. [str]
    - energy: energy in keV. [float]
    - h, k, l: Miller indexes. [int]
    - rel_angle: relative angle [float] 
    - debye_temp_factor: Debye Temperature Factor. [float]
    
    Returns:
    
    - dw_s: Angular Darwin width for s polarization in rad. [float] 
    - dw_p: Angular Darwin width for p polarization in rad. [float] 
    - resolution_s: Intrinsic resolution for s polarization. [float] 
    - resolution_p: Intrinsic resolution for p polarization. [float] 
    
    References:
        
    A simple formula to calculate the x-ray flux after a double-crystal monochromator - M. S. del Rio; O. MATHON.
    
    '''
    
    cryst = xraylib.Crystal_GetCrystal(crystal)
    
    bragg = xraylib.Bragg_angle(cryst, energy, h, k, l)
            
    FH = xraylib.Crystal_F_H_StructureFactor(cryst, energy, h, k, l, debye_temp_factor, rel_angle)
    
    FHbar = xraylib.Crystal_F_H_StructureFactor(cryst, energy, -h, -k, -l, debye_temp_factor, rel_angle)
    
    C = (np.cos(2*bragg))
    
    dw_p = 1e10 * 2 * C * (xraylib.R_E / cryst['volume']) * (xraylib.KEV2ANGST * xraylib.KEV2ANGST/ (energy * energy)) * np.sqrt(abs(FH * FHbar)) / np.pi / np.sin(2*bragg)
    
    dw_s = 1e10 * 2 * (xraylib.R_E / cryst['volume']) * (xraylib.KEV2ANGST * xraylib.KEV2ANGST/ (energy * energy)) * np.sqrt(abs(FH * FHbar)) / np.pi / np.sin(2*bragg)
    
    resolution_p = dw_p/np.tan(bragg)
    
    resolution_s = dw_s/np.tan(bragg)
    
    return dw_s, dw_p, resolution_s, resolution_p


def calc_Darwin_width_vs_E(crystal='Si', energy_array=np.linspace(2.8, 30, 1361), h=1, k=1, l=1, rel_angle=1, debye_temp_factor=1, save_txt=True, save_fig=True, filename_to_save_dw='Darwin_width', filename_to_save_resol='Intrinsic_resolution'):    

    '''
    Calculates the Darwin width and the intrinsic resolution for p and s polarizations as function of energy. It uses xraylib.
    
    Parameters:
        
    - crystal: crystal material. [str]
    - energy: energy in keV. [float]
    - h, k, l: Miller indexes. [int]
    - rel_angle: relative angle [float] 
    - debye_temp_factor: Debye Temperature Factor. [float]
    - save_txt: if True, saves two .txt files with the Darwin width and the intrinsic resolution as function of energy. [boolean]
    - save_fig: if True, saves two figures with the Darwin width and the intrinsic resolution as function of energy. [boolean]
    - filename_to_save_dw: name to save the figure and the .txt file with the Darwin width as function of energy. [string]
    - filename_to_save_resol: name to save the figure and the .txt file with the intrinsic resolution as function of energy. [string]
    
    Returns:
    
    - darwin_width_s: Angular Darwin width for s polarization in rad as function of energy. [float]
    - darwin_width_p: Angular Darwin width for p polarization in rad as function of energy. [float] 
    - resolution_s: Intrinsic resolution for s polarization as function of energy. [float] 
    - resolution_p: Intrinsic resolution for p polarization as function of energy. [float] 
    
    References:
        
    A simple formula to calculate the x-ray flux after a double-crystal monochromator - M. S. del Rio; O. MATHON.
    
    '''
    
    # Calculating:
    
    resolution_s = []
    
    darwin_width_s = []
    
    resolution_p = []
    
    darwin_width_p = []
    
    for energy in energy_array:
    
        dw_s, dw_p, resol_s, resol_p = calc_Darwin_width(crystal, energy, h, k, l, rel_angle=1, debye_temp_factor=1)
        
        darwin_width_p.append(dw_p)
        
        resolution_p.append(resol_p)
        
        darwin_width_s.append(dw_s)
        
        resolution_s.append(resol_s)
    
    resolution_p = np.array(resolution_p)
    
    darwin_width_p = np.array(darwin_width_p)
    
    resolution_s = np.array(resolution_s)
    
    darwin_width_s = np.array(darwin_width_s)
    
    
    # Darwin Width:

    plt.figure()                                   
    plt.plot(energy_array, 1.0E6*darwin_width_s, label='s - pol', linewidth=2, markersize=9)                                                       
    plt.plot(energy_array, 1.0E6*darwin_width_p, label='p - pol', linewidth=2, markersize=9)
    plt.ylabel('Darwin Width [$\mu$rad]', fontsize=15)
    plt.xlabel('Energy [keV]', fontsize=15)
    plt.xscale('linear')
    plt.yscale('linear')
    plt.xlim(energy_array[0], energy_array[-1])
    plt.minorticks_on()
    plt.tick_params(which='both', axis='both', direction='in', right=True, top=True, labelsize=15)
    plt.grid(which='both', alpha=0.2)
    plt.legend(fontsize=12)
    plt.tight_layout()
    if(save_fig):
        plt.savefig(filename_to_save_dw+'.png', dpi=600)


    # Resolution:

    plt.figure()                                                                                          
    plt.plot(energy_array, resolution_s, label='s - pol', linewidth=2, markersize=9)
    plt.plot(energy_array, resolution_p, label='p - pol', linewidth=2, markersize=9)
    plt.ylabel('Resolution [$\Delta\Theta/tan(\Theta)$]', fontsize=15) # [$\dfrac{\Delta\Theta}{tan(\Theta)}$]
    plt.xlabel('Energy [keV]', fontsize=15)
    plt.xscale('linear')
    plt.yscale('linear')
    plt.xlim(energy_array[0], energy_array[-1])
    plt.minorticks_on()
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.tick_params(which='both', axis='both', direction='in', right=True, top=True, labelsize=15)
    plt.grid(which='both', alpha=0.2)
    plt.legend(fontsize=12, loc='best')
    plt.tight_layout()   
    if(save_fig):
        plt.savefig(filename_to_save_resol+'.png', dpi=600)
    
    
    if(save_txt):    
    
        # Writing .txt file:
        
        filename = filename_to_save_dw+'.txt'
        
        with open(filename, 'w') as f:
            
            f.write('#Energy[keV]\tDarwin_Width_s[urad]\tDarwin_Width_p[urad] \n')
            
            for i in range(len(energy_array)):
                
                f.write('%.2f\t%.6E\t%.6E \n' %(energy_array[i], 1.0E6*darwin_width_s[i], 1.0E6*darwin_width_p[i]))
        
        
        # Writing .txt file:
        
        filename = filename_to_save_resol+'.txt'
        
        with open(filename, 'w') as f:
            
            f.write('#Energy[keV]\tResolution_s\tResolution_p \n')
            
            for i in range(len(energy_array)):
                
                f.write('%.2f\t%.6E\t%.6E \n' %(energy_array[i], resolution_s[i], resolution_p[i]))
              
                
    return darwin_width_s, darwin_width_p, resolution_s, resolution_p
    
    
def calc_rocking_curve_shift(crystal='Si', energy=8, h=1, k=1, l=1, rel_angle=1, debye_temp_factor=1):
    
    '''
    Calculates the angular shift of the Rocking Curve. It uses xraylib.
    
    Parameters:
        
    - crystal: crystal material. [str]
    - energy: energy in keV. [float]
    - h, k, l: Miller indexes. [int]
    - rel_angle: relative angle [float] 
    - debye_temp_factor: Debye Temperature Factor. [float]
    
    Returns:

    - w0: Angular shift of the Rocking Curve in rad. [float]
        
    References:
        
    Elements of modern X-ray physics / Jens Als-Nielsen, Des McMorrow – 2nd ed. Cap. 6.
        
    '''
    
    # Function:

    def calc_g(d, r0, Vc, F):
        return abs((2*d*d*r0/(Vc))*F)
    
    
    # Calculating rocking curve shift:
    
    cryst = xraylib.Crystal_GetCrystal(crystal)
    
    bragg = xraylib.Bragg_angle(cryst, energy, h, k, l)
    
    r0 = physical_constants['classical electron radius'][0]
    
    F0 = xraylib.Crystal_F_H_StructureFactor(cryst, energy, 0, 0, 0, debye_temp_factor, rel_angle)
    
    d = 1e-10*xraylib.Crystal_dSpacing(cryst, h, k, l)
    
    V = 1e-30*cryst['volume']
    
    g0 = calc_g(d, r0, V, F0)
    
    w0 = (g0/(np.pi))*np.tan(bragg)    
    
    return w0

    
def calc_Darwin_curve(delta_theta=np.linspace(-0.00015, 0.00015, 5000), crystal='Si', energy=8, h=1, k=1, l=1, rel_angle=1, debye_temp_factor=1, use_correction=True, save_txt=True, save_fig=True, filename_to_save='Darwin_curve'):
    
    '''
    Calculates the Darwin curve. It does not considers absortion. Valid for s-polarization only.
    
    Parameters:
        
    - delta_theta: array containing values of (Theta - Theta_Bragg) in rad. [array]
    - crystal: crystal material. [str]
    - energy: energy in keV. [float]
    - h, k, l: Miller indexes. [int]
    - rel_angle: relative angle [float] 
    - debye_temp_factor: Debye Temperature Factor. [float]
    - use_correction: if True, considers the corrected Bragg angle due to refraction. [boolean] 
    - save_txt: if True, saves the Darwin Curve in a .txt file. [boolean]
    - save_fig: if True, saves a figure with the Darwin Curve in .png. [boolean]
    - filename_to_save: name to save the figure and the .txt file. [string]
    
    Returns:
    
    - delta_theta: Array containing values of (Theta - Theta_Bragg) in rad. [array] 
    - R: Intensity reflectivity array. [array]
    - zeta_total: Total Darwin width (delta_lambda/lambda). [float]
    - zeta_FWHM: Darwin width FWHM (delta_lambda/lambda). [float]
    - w_total: Total Angular Darwin width (delta_Theta) in rad. [float]
    - w_FWHM: Angular Darwin width FWHM (delta_Theta) in rad. [float]
    - w0: Angular shift of the Rocking Curve in rad. [float]
        
    References:
        
    Elements of modern X-ray physics / Jens Als-Nielsen, Des McMorrow – 2nd ed. Cap. 6.
        
    '''
    
    # Functions:

    def calc_g(d, r0, Vc, F):
        return abs((2*d*d*r0/(Vc))*F)
    
    
    def calc_xc(zeta, g, g0):
        return np.pi*zeta/g - g0/g
    
    
    def calc_zeta_total(d, r0, Vc, F):
        return (4/np.pi)*(d)*(d)*(r0*abs(F)/Vc)
        
    
    def Darwin_curve(xc):
        
        R = [] 
        
        for x in xc:
        
            if(x >= 1):
                
                r = (x - np.sqrt(x*x-1))*(x - np.sqrt(x*x-1))
                
            if(x <= 1):
                
                r = 1
                
            if(x <= -1):
                
                r = (x + np.sqrt(x*x-1))*(x + np.sqrt(x*x-1))
                
            R.append(r)
            
        return np.array(R)
    
    
    # Calculating Darwin curve:
    
    cryst = xraylib.Crystal_GetCrystal(crystal)
    
    bragg = xraylib.Bragg_angle(cryst, energy, h, k, l)
    
    zeta = delta_theta/np.tan(bragg)
    
    r0 = physical_constants['classical electron radius'][0]
    
    FH = xraylib.Crystal_F_H_StructureFactor(cryst, energy, h, k, l, debye_temp_factor, rel_angle)
    
    F0 = xraylib.Crystal_F_H_StructureFactor(cryst, energy, 0, 0, 0, debye_temp_factor, rel_angle)
    
    d = 1e-10*xraylib.Crystal_dSpacing(cryst, h, k, l)
    
    V = 1e-30*cryst['volume']
    
    g = calc_g(d, r0, V, FH)
    
    g0 = calc_g(d, r0, V, F0)
    
    xc = calc_xc(zeta, g, g0)
    
    R = Darwin_curve(xc)
    
    zeta_total = calc_zeta_total(d, r0, V, FH)
    
    zeta_FWHM = (3/(2*np.sqrt(2)))*zeta_total
    
    w_total = zeta_total*np.tan(bragg)
    
    w_FWHM = (3/(2*np.sqrt(2)))*w_total
    
    w0 = (g0/(np.pi))*np.tan(bragg)
    
    
    # Correcting curve offset (due to refraction):
    
    if (use_correction):
        
        delta_theta = delta_theta - w0
        
    
    # Saving .txt file:
    
    if(save_txt):
        
        filename = filename_to_save+'.txt'
    
        with open(filename, 'w') as f:
        
            f.write('#Delta_Theta[rad] Intensity_Reflectivity \n')
            
            for i in range(len(R)):
                
                f.write('%.6E\t%.6E \n' %(delta_theta[i], R[i]))
                
            
    # Plotting Graph:
    
    plt.figure()
    plt.plot(delta_theta, R, linewidth=1.8, color='black')
    plt.fill_between(delta_theta, R, alpha=0.9, color='C0')
    plt.ylabel('Intensity reflectivity', fontsize=13)
    plt.xlabel('$\Delta$'+'$\Theta$'+' [rad]', fontsize=13)
    plt.xscale('linear')
    plt.yscale('linear')
    plt.minorticks_on()
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.tick_params(which='both', axis='both', direction='in', right=True, top=True, labelsize=12)
    plt.grid(which='both', alpha=0.2)
    plt.tight_layout()
    textstr = '\n'.join((
    r'$\zeta_{total}=$%.4E' % (zeta_total, ),
    r'$\zeta_{FWHM}=$%.4E' % (zeta_FWHM, ),
    r'$\omega_{total}=$%.4E rad' % (w_total, ),
    r'$\omega_{FWHM}=$%.4E rad' % (w_FWHM, )))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5) # wheat # gray
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=props)
    plt.show()
    if(save_fig):
        plt.savefig(filename_to_save+'.png', dpi=600)
    
    
    return delta_theta, R, zeta_total, zeta_FWHM, w_total, w_FWHM, w0


if __name__ == '__main__':
    
    
    # Example 1:
    
    bragg = calc_Bragg_angle(crystal='Si', energy=8, h=1, k=1, l=1)
    
    print('\n')
    print('Bragg angle: Rad: {} Deg: {}'.format(bragg, bragg*180/np.pi))
    
    
    # Example 2: 
    
    dw_s, dw_p, resolution_s, resolution_p = calc_Darwin_width(crystal='Si', energy=8, h=1, k=1, l=1, rel_angle=1, debye_temp_factor=1)
    
    print('\n')
    print('dw_s = %.4f urad' %(1e+6*dw_s))
    print('dw_p = %.4f urad' %(1e+6*dw_p))
    print('resol_s = %.6E' %(resolution_s))
    print('resol_p = %.6E' %(resolution_p))
    
    
    # Example 3:
    
    darwin_width_s, darwin_width_p, resolution_s, resolution_p = calc_Darwin_width_vs_E(crystal='Si', energy_array=np.linspace(2.8, 30, 1361), h=1, k=1, l=1, rel_angle=1, debye_temp_factor=1, save_txt=False, save_fig=False, filename_to_save_dw='Darwin_width', filename_to_save_resol='Intrinsic_resolution')
    
    
    # Example 4:
    
    w0 = calc_rocking_curve_shift(crystal='Si', energy=8, h=1, k=1, l=1, rel_angle=1, debye_temp_factor=1)
    
    print('\n')
    print('shift = %.4f urad' %(1e+6*w0))

    
    # Example 5:
    
    delta_theta, R, zeta_total, zeta_FWHM, w_total, w_FWHM, w0 = calc_Darwin_curve(delta_theta=np.linspace(-0.00015, 0.00015, 5000), crystal='Si', energy=8, h=1, k=1, l=1, rel_angle=1, debye_temp_factor=1, use_correction=False, save_txt=False, save_fig=False, filename_to_save='Darwin_curve')
    
    print('\n')
    print('w_total = %.4f urad' %(1e+6*w_total))
    print('w_FWHM = %.4f urad' %(1e+6*w_FWHM))
    print('zeta_total = %.6E' %(zeta_total))
    print('zeta_FWHM = %.6E' %(zeta_FWHM))
    print('shift = %.4f urad' %(1e+6*w0))

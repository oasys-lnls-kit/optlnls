#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 12:47:19 2020

@author: sergio.lordano
"""

import numpy as np
from operator import itemgetter
import matplotlib.pyplot as plt
import scipy
from scipy.interpolate import interp1d
from scipy.special import kv

def pimpale_func (alfa1,alfa2,alfa3,beam_offset,deg_mrad):
    """
    Pimpale optimization
    
    """
    
    if deg_mrad == 'mrad':
          
        a1 = np.tan(alfa1*1e-3)
        a2 = np.tan(alfa2*1e-3)
        a3 = np.tan(alfa3*1e-3)
    
    if deg_mrad == 'deg':
        
        a1 = np.tan(np.deg2rad(alfa1))
        a2 = np.tan(np.deg2rad(alfa2))
        a3 = np.tan(np.deg2rad(alfa3))
        
    b = beam_offset/2
    
    D = (a2-a1)*np.sqrt(1+a3**2)+(a3-a2)*np.sqrt(1+a1**2)-(a3-a1)*np.sqrt(1+a2**2);
    
    R = (b/D)*(a3-a2)*(a3-a1)*(a2-a1);
    XL = (b/D)*((a3**2)*np.sqrt(1+a2**2)-(a3**2-a2**2)*np.sqrt(1+a1**2)-(a2**2-a1**2)*np.sqrt(1+a3**2));
    H = (b/D)*((a3-a2)*(1+a2*a3)*np.sqrt(1+a1**2)-(a3-a1)*(1+a1*a3)*np.sqrt(1+a2**2)+(a2-a1)*(1+a1*a2)*np.sqrt(1+a3**2));
    
    return (R,XL,H)    


### Calculates alfa, beta and gamma angles for PGM on sx700 mounting ### 
def sx700_angles(energy, cff, line_density, normal_or_surface):
    r2d = 180.0/np.pi     
    wavelength_mm = (1.23984198433e-3)/(energy)
    alpha_f = np.arcsin(np.sqrt(1.0+(line_density*wavelength_mm*cff/(1-cff**2))**2)+wavelength_mm*line_density/(1-cff**2))
    alpha_f_graz = np.pi/2 - alpha_f
    beta_f = np.arcsin(-np.sqrt(1+((line_density*wavelength_mm*cff)/(1-cff**2))**2)-line_density*wavelength_mm*cff**2/(1-cff**2))
    beta_f_graz = np.pi/2 + beta_f
    gamma_f = (alpha_f-beta_f)/2
    gamma_f_graz = np.pi/2 - gamma_f
    if normal_or_surface == 0:
        return [alpha_f*r2d, beta_f*r2d, gamma_f*r2d]
    if normal_or_surface == 1:
        return [alpha_f_graz*r2d, beta_f_graz*r2d, gamma_f_graz*r2d]
	
	
def sx700_distance_mirror_grating(gap=20e-3, gamma=80.0):
	
	return gap/np.abs(np.sin(np.pi-2*np.deg2rad(gamma))) 
	
	
def calc_grating_beta(wavelength=200e-9, alpha=83.7480078, k0=75, m=-1, energy=0):
    
    if energy != 0:
        wavelength = 1.23984198433e-6/energy
    
    return np.arcsin(wavelength*1e3 * m * k0 - np.sin(alpha*np.pi/180)) * 180 / np.pi
	
	
def calc_constant_included_angle(wavelength=200e-9, k0=75, m=1, two_theta=162, energy=0):
    ### equation from x-ray data booklet
    
    if energy != 0:
        wavelength = 1.23984198433e-6/energy
    
    beta = np.arcsin((m * k0 * wavelength*1e3)/(2 * np.cos((two_theta/2)*np.pi/180)))*180/np.pi - two_theta/2
    alpha = two_theta + beta	
    return alpha, beta
    

def TGM_optimize_radii(alpha=83.7480078, beta=78.2519922, p=1.0016, q=1.41428):
    ### equations from Peatman's book
    
    f = 1/(1/p+1/q)

    cos_alpha = np.cos(alpha*np.pi/180)
    cos_beta = np.cos(beta*np.pi/180)
    
    Rs = f * (cos_alpha + cos_beta) 
    Rm = (cos_alpha + cos_beta) / (cos_alpha**2 / p + cos_beta**2 / q)    	

    return Rm, Rs


def TGM_focal_position(R=7.977, r=0.18228, alpha=83.7480078, beta=78.2519922, p=1.0016):
    
    cos_alpha = np.cos(alpha*np.pi/180)
    cos_beta = np.cos(beta*np.pi/180)
    
    # meridional focusing
    qm = (p*R*cos_beta**2) / ( p * (cos_alpha+cos_beta) - R*cos_alpha**2 )
        
    # sagittal focusing
    fs = r / (cos_alpha + cos_beta)
    qs = 1 / (1/fs - 1/p)
        
    return qm, qs

	

def calc_grating_angle_from_cff(energy, k0, cff, m):

    wavelength = 1.23984198433*1e-6/energy # [m]
        
    
    wavelength_mm = wavelength*1e3 # used in mm to match k0 dimension in lines/mm
    m_shadow = -1*m # shadow uses negative value
    
    sin_alpha = (-m_shadow*k0*wavelength_mm/(cff**2 - 1)) + \
                np.sqrt(1 + (m_shadow*m_shadow*cff*cff*k0*k0*wavelength_mm*wavelength_mm)/((cff**2 - 1)**2))
    
    alpha = np.arcsin(sin_alpha)
    beta = -np.arcsin(sin_alpha - m_shadow*k0*wavelength_mm)
    gamma = (alpha - beta)/2
    
    alpha_deg = alpha*180/np.pi
    beta_deg = beta*180/np.pi
    gamma_deg = gamma*180/np.pi
    
    return alpha_deg, beta_deg, gamma_deg
    



def calc_VLS_polynomial_srw(energy, k0, m, alpha, beta, r_a, r_b):

    wavelength = 1.23984198433*1e-6/energy # [m]
    wavelength_mm = wavelength*1e3 # used in mm to match k0 dimension in lines/mm
    m_shadow = -1*m # shadow uses negative value    

    cos_alpha = np.cos(alpha * np.pi / 180)
    sin_alpha = np.sin(alpha * np.pi / 180)
    cos_beta = np.cos(beta * np.pi / 180)
    sin_beta = np.sin(beta * np.pi / 180)
    
    ### VLS
    b2 = (((cos_alpha**2)/r_a) + ((cos_beta**2)/r_b))/(-2*m_shadow*k0*wavelength_mm)
    b3 = ((sin_alpha*cos_alpha**2)/r_a**2 - \
         (sin_beta*cos_beta**2)/r_b**2)/(-2*m_shadow*k0*wavelength_mm)
    b4 = (((4*sin_alpha**2 - cos_alpha**2)*cos_alpha**2)/r_a**3 + \
         ((4*sin_beta**2 - cos_beta**2)*cos_beta**2)/r_b**3)/(-8*m_shadow*k0*wavelength_mm)
    
    srw_coeff_0 = round(k0, 8)
    srw_coeff_1 = round(-2*b2, 8)
    srw_coeff_2 = round(3*b3, 8)
    srw_coeff_3 = round(-4*b4, 8)
    
    return [srw_coeff_0, srw_coeff_1, srw_coeff_2, srw_coeff_3]
        

def calc_PGM_distances(r_a, fixed_height, gamma):

    d_source_to_plane_mirror = r_a - (fixed_height/np.abs(np.tan(np.pi-2*gamma)))
    d_mirror_to_grating = fixed_height/np.abs(np.sin(np.pi-2*gamma))
    
    return [d_source_to_plane_mirror, d_mirror_to_grating]

def calc_VLS_polynomial_shadow(energy, k0, m, alpha, beta, r_a, r_b):

    wavelength = 1.23984198433*1e-6/energy # [m]
    wavelength_mm = wavelength*1e3 # used in mm to match k0 dimension in lines/mm
    m_shadow = -1*m # shadow uses negative value    

    cos_alpha = np.cos(alpha * np.pi / 180)
    sin_alpha = np.sin(alpha * np.pi / 180)
    cos_beta = np.cos(beta * np.pi / 180)
    sin_beta = np.sin(beta * np.pi / 180)
    
    ### VLS
    b2 = (((cos_alpha**2)/r_a) + ((cos_beta**2)/r_b)) / (-2*m_shadow*k0*wavelength_mm)
    
    b3 = ((sin_alpha*cos_alpha**2)/r_a**2 - \
         (sin_beta*cos_beta**2)/r_b**2)/(-2*m_shadow*k0*wavelength_mm)
        
    b4 = (((4*sin_alpha**2 - cos_alpha**2)*cos_alpha**2)/r_a**3 + \
         ((4*sin_beta**2 - cos_beta**2)*cos_beta**2)/r_b**3)/(-8*m_shadow*k0*wavelength_mm)
    
    shadow_coeff_0 = round(k0, 8)
    shadow_coeff_1 = round(-2*k0*b2, 8)
    shadow_coeff_2 = round(3*k0*b3, 8)
    shadow_coeff_3 = round(-4*k0*b4, 8)

    return [shadow_coeff_0, shadow_coeff_1, shadow_coeff_2, shadow_coeff_3]


def test_vls_sape():


    energy_reference = 10.0 # eV
    wavelength = 1.23984198433*1e-6/energy_reference # [m]
    k0 = 180.0 # lines/mm
    cff = 3.0
    m = -1 # diffraction order
    
    alpha_deg, beta_deg, gamma_deg = calc_grating_angle_from_cff(energy=energy_reference, k0=k0, cff=cff, m=m)
    
    alpha_deg = round(alpha_deg, 8)
    beta_deg = round(beta_deg, 8)
    gamma_deg = round(gamma_deg, 8)
    
   
    r_a = -6000.0 # mm
    r_b = 6000.0 # mm
        
    poly = calc_VLS_polynomial_shadow(energy=energy_reference, k0=k0, m=m, alpha=alpha_deg, beta=-beta_deg, r_a=r_a, r_b=r_b)
    # print(poly)
    
    if(1):

        print('alpha =', alpha_deg)        
        print('beta =', beta_deg)
        print('gamma =', gamma_deg)
        
        print('shadow c0 =', poly[0])
        print('shadow c1 =', poly[1])
        print('shadow c2 =', poly[2])
        print('shadow c3 =', poly[3])
    
    



if __name__ == '__main__':
    
    pass
    test_vls_sape()











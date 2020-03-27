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
    wavelength_mm = (1.2398e-3)/(energy)
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
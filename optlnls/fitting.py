#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 11:49:02 2020

@author: sergio.lordano
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter 
from optlnls.math import lorentz_function, lorentz_gauss_function
from optlnls.math import gauss_function, pseudo_voigt_asymmetric, get_fwhm 

def fit_gauss(x, y, p0, maxfev):

    try:
        popt, pcov = curve_fit(gauss_function, x, y, p0=p0, maxfev=maxfev)
        
    except ValueError:
        popt, pcov, perr = [0]*3, [0]*3, [0]*3        
        print("Could not fit data\n") 
    except RuntimeError:
        pcov = [0]*5      
        popt = p0
        print("Could not fit data\n") 
        
    perr = np.sqrt(np.diag(pcov))
    return popt, perr  
        
def fit_lorentz(x, y, p0, maxfev=20000):

    try:
        popt, pcov = curve_fit(lorentz_function, x, y, p0=p0, maxfev=maxfev)
        
    except ValueError:
        popt, pcov = [0]*3, [0]*3        
        print("Could not fit data\n") 
    except RuntimeError:
        pcov = [0]*5      
        popt = p0
        print("Could not fit data\n") 
    
    perr = np.sqrt(np.diag(pcov))
    return popt, perr  

def fit_lorentz_gauss(x, y, p0, maxfev=20000):

    try:
        popt, pcov = curve_fit(lorentz_gauss_function, x, y, p0=p0, maxfev=maxfev)
        
    except ValueError:
        popt, pcov = [0]*3, [0]*3        
        print("Could not fit data\n") 
    except RuntimeError:
        pcov = [0]*5      
        popt = p0
        print("Could not fit data\n") 
    
    perr = np.sqrt(np.diag(pcov))
    return popt, perr


def fit_pseudo_voigt_asymmetric(x, y, p0=[0]*6, autoguess=1, maxfev=20000, plot=0):
    
    if(autoguess):
        y_savgol = savgol_filter(y, 51, 5)
        fwhm = get_fwhm(x, y, oversampling=200)[0]
        mean = np.average(x, weights=y)
        peak = np.max(y_savgol)        
        p0 = [mean, peak, fwhm, 0.5, 0.5, 0.5]
        
    try:
        popt, pcov = curve_fit(pseudo_voigt_asymmetric, x, y, p0=p0, maxfev=maxfev)
        
        if(plot):
            y_fit = pseudo_voigt_asymmetric(x, *popt)
            plt.figure()
            plt.plot(x, y, 'o', label='data')
            plt.plot(x, y_savgol, '--', label='savgol filter')
            plt.plot(x, y_fit, '-', label='pseudo-voigt')
            plt.legend()
            
            
        
    except ValueError:
        popt, pcov = [0]*6, [0]*6        
        print("Could not fit data\n")
        
    except RuntimeError:
        pcov = [0]*6      
        popt = p0
        print("Could not fit data\n") 
    
    perr = np.sqrt(np.diag(pcov))
    return popt, perr  



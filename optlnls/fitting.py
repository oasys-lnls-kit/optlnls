#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 11:49:02 2020

@author: sergio.lordano
"""

import numpy as np
from scipy.optimize import curve_fit
from optlnls.math import gauss_function, lorentz_function, lorentz_gauss_function

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
        
def fit_lorentz(x, y, p0, maxfev):

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

def fit_lorentz_gauss(x, y, p0, maxfev):

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
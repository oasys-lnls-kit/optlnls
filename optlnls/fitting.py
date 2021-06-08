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
from optlnls.math import get_fwhm 

def fit_gauss(x, y, p0=[0]*3, maxfev=20000, autoguess=1, filtered=0, window_length=11, poly_order=5):

    if(autoguess):
        mean, peak, fwhm = fit_autoguess(x, y, filtered, window_length, poly_order)
        p0 = [peak, mean, fwhm/2]     

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

def fit_lorentz(x, y, p0=[0]*3, maxfev=20000, autoguess=1, filtered=0, window_length=11, poly_order=5):

    if(autoguess):
        mean, peak, fwhm = fit_autoguess(x, y, filtered, window_length, poly_order)
        p0 = [peak, mean, fwhm/2]        

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

def fit_lorentz_gauss(x, y, p0=[0]*5, maxfev=20000, autoguess=1, filtered=0, window_length=11, poly_order=5):

    if(autoguess):
        mean, peak, fwhm = fit_autoguess(x, y, filtered, window_length, poly_order)
        p0 = [mean, peak, fwhm/2, peak, fwhm/2]    

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

def fit_autoguess(x, y, filtered=0, window_length=11, poly_order=5):

    fwhm = get_fwhm(x, y, oversampling=200)[0]
    mean = np.average(x, weights=y)
    
    if(filtered):
        y_savgol = savgol_filter(y, window_length, poly_order)
        peak = np.max(y_savgol) 
    else:
        peak = np.max(y)

    return mean, peak, fwhm
    
    

def fit_pseudo_voigt_asymmetric(x, y, p0=[0]*6, autoguess=1, maxfev=20000, plot=0, filtered=0, window_length=11, poly_order=5):
    
    if(autoguess):
        mean, peak, fwhm = fit_autoguess(x, y, filtered, window_length, poly_order)
        p0 = [mean, peak, fwhm, 0.5, 0.5, 0.5]
        
    try:
        popt, pcov = curve_fit(pseudo_voigt_asymmetric, x, y, p0=p0, maxfev=maxfev)
        
        if(plot):
            y_fit = pseudo_voigt_asymmetric(x, *popt)
            plt.figure()
            plt.plot(x, y, 'o', label='data')
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


def linear_function(x, a, b):
    return a*x + b

def gauss_function(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def lorentz_function(x, a, x0, sigma):
    return a / (sigma * (1 + ((x - x0) / sigma )**2 ) )

def lorentz_gauss_function(x, x0, a, sigma, b, gamma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2)) + (b / (gamma * (1 + ((x - x0) / gamma )**2)))

def error_function(x, a, x0, sigma, y0):
    
    gaussian = gauss_function(x, a, x0, sigma)
    
    sign = 1 if a > 0 else -1
    
    y = np.zeros((len(x)))
    y[0] = y0
    for i in range(int(len(x)-1)):
        y[i+1] = y[i] + sign*np.abs(gaussian[i+1] + gaussian[i])/2

    x_step = np.abs(x[1] - x[0])
    y *= x_step
    
    return y

def pseudo_voigt_asymmetric_normalized(x, x0, sigma, alpha, beta, m):
    
    ln2 = np.log(2)
    pi = np.pi
    x = x - x0
    sigma_x = 2*sigma / (1 + np.exp(-alpha * (x - beta)))
    term1  = (1-m) * np.sqrt( 4 * ln2 / (pi * sigma_x**2) )
    term1 *= np.exp( -(4 * ln2 / sigma_x**2) * x**2 )
    term2  = (m / (2 * pi)) * sigma_x / ( (sigma_x/2)**2 + 4*x**2 )
    pseudov_asymmetric = term1 + term2
    return pseudov_asymmetric


def pseudo_voigt_asymmetric(x, x0, a, sigma, alpha, beta, m):
    
    ln2 = np.log(2)
    pi = np.pi
    x = x - x0
    sigma_x = 2*sigma / (1 + np.exp(-alpha * (x - beta)))
    term1  = (1-m) * np.sqrt( 4 * ln2 / (pi * sigma_x**2) )
    term1 *= np.exp( -(4 * ln2 / sigma_x**2) * x**2 )
    term2  = (m / (2 * pi)) * sigma_x / ( (sigma_x/2)**2 + 4*x**2 )
    pseudov_asymmetric = term1 + term2
    pseudov_asymmetric *= a / np.max(pseudov_asymmetric)
    return pseudov_asymmetric

if __name__ == '__main__':
    
    x = np.linspace(-6, 6, 100)
    y = gauss_function(x, 5, 2, 1.5)
    
    y += np.random.random(size=len(x))
    
    ### gaussian
    # popt, perr = fit_gauss(x, y, autoguess=1)
    # y_fit = gauss_function(x, *popt)

    ### lorentzian
    # popt, perr = fit_lorentz(x, y, autoguess=1)
    # y_fit = lorentz_function(x, *popt)

    ### lorentzian + gaussian
    # popt, perr = fit_lorentz_gauss(x, y, autoguess=1)
    # y_fit = lorentz_gauss_function(x, *popt)
    
    ### pseudo-voigt asymmetric
    popt, perr = fit_pseudo_voigt_asymmetric(x, y, autoguess=1)
    y_fit = pseudo_voigt_asymmetric(x, *popt)
    
    plt.figure()
    plt.plot(x, y, '.')
    plt.plot(x, y_fit, '-')    
    
    
    
    


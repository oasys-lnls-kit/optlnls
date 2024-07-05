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


def fit_cauchy_gauss(x, y, p0=[0]*5, maxfev=20000, autoguess=1, filtered=0, window_length=11, poly_order=5):

    if(autoguess):
        mean, peak, fwhm = fit_autoguess(x, y, filtered, window_length, poly_order)
        p0 = [mean, peak, fwhm/2, peak, fwhm/2]    

    try:
        popt, pcov = curve_fit(cauchy_gauss_function, x, y, p0=p0, maxfev=maxfev)
        
    except ValueError:
        popt, pcov = [0]*3, [0]*3        
        print("Could not fit data\n") 
    except RuntimeError:
        pcov = [0]*5      
        popt = p0
        print("Could not fit data\n") 
    
    perr = np.sqrt(np.diag(pcov))
    return popt, perr

def fit_cauchy_lorentz(x, y, p0=[0]*5, maxfev=20000, autoguess=1, filtered=0, window_length=11, poly_order=5):

    if(autoguess):
        mean, peak, fwhm = fit_autoguess(x, y, filtered, window_length, poly_order)
        p0 = [mean, peak, fwhm/2, peak, fwhm/2]    

    try:
        popt, pcov = curve_fit(cauchy_lorentz_function, x, y, p0=p0, maxfev=maxfev)
        
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


def fit_cauchy(x, y, p0=[0]*3, maxfev=20000, autoguess=1, filtered=0, window_length=11, poly_order=5):

    if(autoguess):
        mean, peak, fwhm = fit_autoguess(x, y, filtered, window_length, poly_order)
        p0 = [peak, mean, fwhm/2]        

    try:
        popt, pcov = curve_fit(cauchy_function, x, y, p0=p0, maxfev=maxfev)
        
    except ValueError:
        popt, pcov = [0]*3, [0]*3        
        print("Could not fit data\n") 
    except RuntimeError:
        pcov = [0]*5      
        popt = p0
        print("Could not fit data\n") 
    
    perr = np.sqrt(np.diag(pcov))
    return popt, perr  


def fit_pearson_vii(x, y, p0=[0]*4, maxfev=20000, autoguess=1, filtered=0, window_length=11, poly_order=5):

    if(autoguess):
        mean, peak, fwhm = fit_autoguess(x, y, filtered, window_length, poly_order)
        p0 = [peak, mean, fwhm/2, 4.0]        

    try:
        popt, pcov = curve_fit(pearson_vii_function, x, y, p0=p0, maxfev=maxfev)
        
    except ValueError:
        popt, pcov = [0]*3, [0]*3        
        print("Could not fit data\n") 
    except RuntimeError:
        pcov = [0]*5      
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

def cauchy_gauss_function(x, x0, a1, sigma1, a2, sigma2):
    return gauss_function(x, a1, x0, sigma1) + cauchy_function(x, x0, a2, sigma2)

def cauchy_lorentz_function(x, x0, a1, sigma1, a2, sigma2):
    return lorentz_function(x, a1, x0, sigma1) + cauchy_function(x, x0, a2, sigma2)


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
    '''
    Calculates the Asymmetric Pseudo-Voigt Profile.
    
    Parameters
    ----------
    x : array_like
        Input value.
    x0 : float
         Peak center position.
    a : float
        Normalization constant, peak value.
    sigma : float
            This is actually the full width at half maximum (FWHM) in the symmetric case.
    alpha : float
            Asymmetry parameter. Controls how the FWHM varies sigmoidally.
    beta : float
           Displacement of the sigmoidal step relative to the peak.
    m : float
        Fraction of Lorentzian character contributing to the net line shape. When f = 1,
        the shape is a pure Lorentzian. When f = 0, the shape is a pure Gaussian.
    
    Returns
    -------
    pseudov_asymmetric : array_like
                          Asymmetric Pseudo-Voigt Profile
    
    References
    ----------
    [1]: Aaron L. Stancik and Eric B. Brauns, "A simple asymmetric lineshape for fitting
    infrared absorption spectra", Vibrational Spectroscopy, Volume 47, Issue 1, 2008,
    Pages 66-69. https://doi.org/10.1016/j.vibspec.2008.02.009 \n
    [2]: Schmid, M., Steinrück, H.-P. and Gottfried, J.M. (2014), "A new asymmetric Pseudo-Voigt
    function for more efficient fitting of XPS lines", Surf. Interface Anal., 46: 505-511.
    https://doi.org/10.1002/sia.5521
    '''
    
    ln2 = np.log(2)
    pi = np.pi
    x = x - x0
    sigma_x = 2*sigma / (1 + np.exp(-alpha * (x - beta)))
    term1  = (1-m) * np.sqrt( 4 * ln2 / (pi * sigma_x**2) )
    term1 *= np.exp( -(4 * ln2 / sigma_x**2) * x**2 )
    # term2  = (m / (2 * pi)) * sigma_x / ( (sigma_x/2)**2 + 4*x**2 )
    term2 = (m / (pi * sigma_x)) / (1 + 4 * (x/sigma_x)**2)
    pseudov_asymmetric = term1 + term2
    pseudov_asymmetric *= a / np.max(pseudov_asymmetric)
    return pseudov_asymmetric


def cauchy_function(x, x0, a, sigma):
    
    return a / (1 + ((x-x0)/sigma)**2)

def pearson_vii_function(x, x0, a, sigma, m):
    
    return a * (1 + ((x-x0)**2) / (m * sigma**2))



def calc_rsquared(y, yfit):
    """
    Parameters
    ----------
    y : float, array
        y data points.
    yfit : float, array
        y fitted points in the same x-coordinates.

    Returns
    -------
    r_squared : float
        R^2 - the fitting quality parameter.
    """

    residuals = y - yfit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y- np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    return r_squared




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
    
    
    
    


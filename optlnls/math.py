#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 11:47:24 2020

@author: sergio.lordano
"""

import numpy as np
from scipy.interpolate import interp1d

def gauss_function(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def lorentz_function(x, a, x0, sigma):
    return a / (sigma * (1 + ((x - x0) / sigma )**2 ) )

def lorentz_gauss_function(x, x0, a, sigma, b, gamma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2)) + (b / (gamma * (1 + ((x - x0) / gamma )**2)))

def calc_rms(x, f_x):
    return np.sqrt(np.sum(f_x*np.square(x))/np.sum(f_x) - (np.sum(f_x*x)/np.sum(f_x))**2)

def add_zeros(array, n):
    aux = []
    for i in range(len(array)):
        aux.append(array[i])
    for k in range(n):
        aux.insert(0, 0)
        aux.append(0)
    return np.array(aux)

def add_steps(array, n):
    aux = []
    step = (np.max(array)-np.min(array))/(len(array)-1)
    for i in range(len(array)):
        aux.append(array[i])
    for k in range(n):
        aux.insert(0, array[0] - (k+1)*step)
        aux.append(array[-1] + (k+1)*step)
    return np.array(aux)

def resample_distribution(array_x, array_y, oversampling=2, n_points=0):
    dist = interp1d(array_x, array_y)
    if(n_points == 0):
        x_int = np.linspace(np.min(array_x), np.max(array_x), int(len(array_x)*oversampling))
    else:
        x_int = np.linspace(np.min(array_x), np.max(array_x), n_points)
    y_int = dist(x_int)
    return x_int, y_int 


def derivate(x, y):    
    diffy = [y[i+1]-y[i] for i in range(len(y)-1)]
    diffx = [x[i+1]-x[i] for i in range(len(x)-1)]
    der = [diffy[i]/diffx[i] for i in range(len(diffx))]
    der.append(der[-1])
    return np.array(der)

def get_fwhm(x, y, oversampling=1, zero_padding=True, avg_correction=False, 
             inmost_outmost=0, threshold=0.5, npoints=5):
    
    if(oversampling > 1.0):
        array_x, array_y = resample_distribution(x, y, oversampling)
    else:
        array_x, array_y = x, y
        
    if(zero_padding):
        array_x = add_steps(x, 3)
        array_y = add_zeros(y, 3)
        
    try: 
        
        ### FIRST SEARCH (ROUGH) ###
        
        y_peak = np.max(array_y)
        threshold = threshold * y_peak
        idx_peak = (np.abs(array_y-y_peak)).argmin()
        
        if(idx_peak==0):
            left_hwhm_idx = 0
        else:
            if(inmost_outmost == 0): # INMOST
                for i in range(idx_peak,0,-1):
                    if np.abs(array_y[i]-threshold)<np.abs(array_y[i-1]-threshold) and (array_y[i-1]-threshold)<0:
                        break                
                left_hwhm_idx = i            
            else: # OUTMOST
                for i in range(0,idx_peak):
                    if np.abs(array_y[i]-threshold)>np.abs(array_y[i-1]-threshold) and (array_y[i-1]-threshold)>0:
                        break                
                left_hwhm_idx = i 
            
        if(idx_peak==len(array_y)-1):
            right_hwhm_idx = len(array_y)-1
        else:
            if(inmost_outmost == 0): # INMOST
                for j in range(idx_peak,len(array_y)-2):
                    if np.abs(array_y[j]-threshold)<np.abs(array_y[j+1]-threshold) and (array_y[j+1]-threshold)<0:
                        break              
                right_hwhm_idx = j
            else: # OUTMOST
                for j in range(len(array_y)-2, idx_peak, -1):
                    if np.abs(array_y[j]-threshold)>np.abs(array_y[j+1]-threshold) and (array_y[j+1]-threshold)>0:
                        break              
                right_hwhm_idx = j    
        
        fwhm = array_x[right_hwhm_idx] - array_x[left_hwhm_idx] 
        
        ### SECOND SEARCH (FINE) ###
#            npoints = 5 # to use for each side
        left_min = left_hwhm_idx-npoints if left_hwhm_idx-npoints >=0 else 0
        left_max = left_hwhm_idx+npoints+1 if left_hwhm_idx+npoints+1 < len(array_x) else -1
        right_min = right_hwhm_idx-npoints if right_hwhm_idx-npoints >=0 else 0
        right_max = right_hwhm_idx+npoints+1 if right_hwhm_idx+npoints+1 < len(array_x) else -1
        
#            left_fine_x, left_fine_y = interp_distribution(array_x[left_hwhm_idx-npoints: left_hwhm_idx+npoints+1], array_y[left_hwhm_idx-npoints: left_hwhm_idx+npoints+1], oversampling=int(oversampling*50))
#            right_fine_x, right_fine_y = interp_distribution(array_x[right_hwhm_idx-npoints: right_hwhm_idx+npoints+1], array_y[right_hwhm_idx-npoints: right_hwhm_idx+npoints+1], oversampling=int(oversampling*50))
        
        left_fine_x, left_fine_y = resample_distribution(array_x[left_min: left_max], array_y[left_min: left_max], oversampling=int(oversampling*50))
        right_fine_x, right_fine_y = resample_distribution(array_x[right_min: right_max], array_y[right_min: right_max], oversampling=int(oversampling*50))
        
        
        if(inmost_outmost == 0): # INMOST
            for i in range(len(left_fine_y)-1, 0, -1):
                if np.abs(left_fine_y[i]-threshold)<np.abs(left_fine_y[i-1]-threshold) and (left_fine_y[i-1]-threshold)<0:
                        break                
            left_hwhm_idx = i 
            
            for j in range(0,len(right_fine_y)-2):
                if np.abs(right_fine_y[j]-threshold)<np.abs(right_fine_y[j+1]-threshold) and (right_fine_y[j+1]-threshold)<0:
                    break              
            right_hwhm_idx = j
                
        elif(inmost_outmost == 1): # OUTMOST
            for i in range(0, len(left_fine_y)-1):
                if np.abs(left_fine_y[i]-threshold)<np.abs(left_fine_y[i+1]-threshold) and (left_fine_y[i+1]-threshold)>0:
                    break                
            left_hwhm_idx = i
            
            for j in range(len(right_fine_y)-2, 0, -1):
                if np.abs(right_fine_y[j]-threshold)<np.abs(right_fine_y[j-1]-threshold) and (right_fine_y[j-1]-threshold)>0:
                    break              
            right_hwhm_idx = j
        
        
        fwhm = right_fine_x[right_hwhm_idx] - left_fine_x[left_hwhm_idx]               
            
        if(avg_correction):
            avg_y = (left_fine_y[left_hwhm_idx]+ right_fine_y[right_hwhm_idx])/2.0
            popt_left = np.polyfit(left_fine_x[left_hwhm_idx-1 : left_hwhm_idx+2], left_fine_y[left_hwhm_idx-1 : left_hwhm_idx+2] , 1) 
            popt_right = np.polyfit(right_fine_x[right_hwhm_idx-1 : right_hwhm_idx+2], right_fine_y[right_hwhm_idx-1 : right_hwhm_idx+2] , 1) 
            
            x_left = (avg_y-popt_left[1])/popt_left[0]
            x_right = (avg_y-popt_right[1])/popt_right[0]
            fwhm = x_right - x_left 

            return [fwhm, x_left, x_right, avg_y, avg_y]
        else:

            return [fwhm, left_fine_x[left_hwhm_idx], right_fine_x[right_hwhm_idx], left_fine_y[left_hwhm_idx], right_fine_y[right_hwhm_idx]]
#                return [fwhm, left_fine_x[left_hwhm_idx], right_fine_x[right_hwhm_idx], left_fine_y[left_hwhm_idx], right_fine_y[right_hwhm_idx], right_fine_x, right_fine_y, left_fine_x, left_fine_y]
        
    except ValueError:
        fwhm = 0.0        
        print("Could not calculate fwhm\n")   
        return [fwhm, 0, 0, 0, 0]


def psd(xx, yy, onlyrange = None):
    """
     psd: Calculates the PSD (power spectral density) from a profile

      INPUTS:
           x - 1D array of (equally-spaced) lengths.
           y - 1D array of heights.
      OUTPUTS:
           f - 1D array of spatial frequencies, in units of 1/[x].
           s - 1D array of PSD values, in units of [y]^3.
      KEYWORD PARAMETERS:
           onlyrange - 2-element array specifying the min and max spatial
               frequencies to be considered. Default is from
               1/(length) to 1/(2*interval) (i.e., the Nyquist
               frequency), where length is the length of the scan,
               and interval is the spacing between points.

      PROCEDURE
            Use FFT

    """
    import numpy
    n_pts = xx.size
    if (n_pts <= 1):
        print ("psd: Error, must have at least 2 points.")
        return 0

    window=yy*0+1.
    length=xx.max()-xx.min()  # total scan length.
    delta = xx[1] - xx[0]

    # psd from windt code
    # s=length*numpy.absolute(numpy.fft.ifft(yy*window)**2)
    # s=s[0:(n_pts/2+1*numpy.mod(n_pts,2))]  # take an odd number of points.

    #xianbo + luca:
    s0 = numpy.absolute(numpy.fft.fft(yy*window))
    s =  2 * delta * s0[0:int(len(s0)/2)]**2/s0.size # uniformed with IGOR, FFT is not symmetric around 0
    s[0] /= 2
    s[-1] /= 2


    n_ps=s.size                       # number of psd points.
    interval=length/(n_pts-1)         # sampling interval.
    f_min=1./length                   # minimum spatial frequency.
    f_max=1./(2.*interval)            # maximum (Nyquist) spatial frequency.
    # spatial frequencies.
    f=numpy.arange(float(n_ps))/(n_ps-1)*(f_max-f_min)+f_min

    if onlyrange != None:
        roi =  (f <= onlyrange[1]) * (f >= onlyrange[0])
        if roi.sum() > 0:
            roi = roi.nonzero()
            f = f[roi]
            s = s[roi]

    return s,f












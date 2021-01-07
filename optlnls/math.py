#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 11:47:24 2020

@author: sergio.lordano
"""

import numpy as np
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt

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


def derivate(x, y, return_centered_values=True):
    """

    Parameters
    ----------
    x : array_like
        array containing the coordinates.
    y : array_like
        array containing the values of the function to be derived.
    return_centered_values : boolean (optional)
        if True, will, calculate a new coordinate array with values from the
        mean positios, otherwise will discard the last value.
        The default is True.

    Returns
    -------
    x_new : TYPE
        DESCRIPTION.
    dy_dx : TYPE
        DESCRIPTION.

    """
    dy_dx = np.diff(y) / np.diff(x)
    if return_centered_values:
        x_new = np.array([(x[i] + x[i + 1]) / 2 for i in range(len(x) - 1)])
    else:
        x_new = x[:-1]

    return x_new, dy_dx

def derivate_keeping_size(x, y):  
    """creates derivated array, but keeping same dimensions by copying last value"""
    diffy = [y[i+1]-y[i] for i in range(len(y)-1)]
    diffx = [x[i+1]-x[i] for i in range(len(x)-1)]
    der = [diffy[i]/diffx[i] for i in range(len(diffx))]
    der.append(der[-1])
    return np.array(der)


def bin_matrix(matrix, binning_y, binning_x):

    yn, xn = matrix.shape

    if ((xn % binning_x != 0) or (yn % binning_y != 0)):
        print('array of shape ({0} x {1}) cannot be binned by factor ({2},{3})'.format(yn, xn, binning_y, binning_x))
        return matrix
    
    else:
        print('binning matrix of shape({0},{1}) by factors ({2},{3})'.format(yn, xn, binning_y, binning_x))
        xn = int(xn / binning_x)
        yn = int(yn / binning_y)
            
        matrix_binned = np.zeros((yn,xn), dtype=float)
        
        count_y = 0
        for iy in range(yn):
    
            count_x = 0
            for ix in range(xn):
                
                matrix_binned[iy,ix] = np.sum(matrix[count_y:count_y+binning_y,
                                                     count_x:count_x+binning_x])
    
                count_x += binning_x
            count_y += binning_y
            
        matrix_binned /= binning_x*binning_y
      
        if(0):
            fig, ax = plt.subplots(figsize=(12,4), ncols=2)
            im0 = ax[0].imshow(matrix, origin='lower')
            im1 = ax[1].imshow(matrix_binned, origin='lower')
            fig.colorbar(im0, ax=ax[0])
            fig.colorbar(im1, ax=ax[1])
        
            plt.show()
     
        return matrix_binned



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


def common_region_average(lines, force_step=0.0):
    
    '''
    lines = [[x0, y0], [x1, y1], ..., [xn-1, yn-1]]
    
    '''
        
    if(force_step != 0):

        step = force_step
		
    else:
	
	    # Finding the best step:

        step = abs(lines[0][0][1] - lines[0][0][0])

        for i in range(len(lines)):
            
            if (abs(lines[i][0][1] - lines[i][0][0]) < step):
                
                step = abs(lines[i][0][1] - lines[i][0][0])
            
    
    # Finding the position array:
    
    x_max = np.max(lines[0][0])
    
    x_min = np.min(lines[0][0])
    
    for i in range(len(lines)):
        
        if (np.max(lines[i][0]) > x_max):
            
            x_max = np.max(lines[i][0])
        
        if (np.min(lines[i][0]) < x_min):
            
            x_min = np.min(lines[i][0])
            
    x_average = np.arange(x_min, x_max, step)
     

    # Calculating average:
       
    f_list = []

    y_average = []
    
    for i in range(len(lines)):
        
        f = interp1d(lines[i][0], lines[i][1])
        
        f_list.append(f)
    
    
    sum_y = 0
    
    n = 0
    
    for i in range(len(x_average)):
        
        for j in range(len(lines)):
            
            try:
            
                f = f_list[j]
                
                y_interp = f(x_average[i])
            
                sum_y = sum_y + y_interp
                
                n = n + 1
                
            except(ValueError):
                
                pass
        
        y_average.append(sum_y/n)
        
        sum_y = 0
        
        n = 0    
               
        
    return x_average, np.array(y_average)


def zero_padding(mtx1, zero_pad_x=0, zero_pad_y=0, debug=False):
    
    if not(float(zero_pad_x).is_integer()):
        zero_pad_x = int(np.ceil(zero_pad_x))
        print("WARNING: zero_pad_x is not an integer and will be rounded to next integer")
   
    if not(float(zero_pad_y).is_integer()):
        zero_pad_y = int(np.ceil(zero_pad_y))
        print("WARNING: zero_pad_y is not an integer and will be rounded to next integer")
   
    
    step_y = np.mean(np.diff(mtx1[1:,0]))
    step_x = np.mean(np.diff(mtx1[0,1:]))
    
    range_y = mtx1[-1,0] - mtx1[1,0]
    range_x = mtx1[0,-1] - mtx1[0,1]
    
    ny = len(mtx1[1:,0])
    nx = len(mtx1[0,1:])
    
    if(debug):
    
        print('zero_pad_y, zero_pad_x')
        print(zero_pad_y, zero_pad_x)
        print('inital (step / range / npoints) y,x')
        print(step_y, step_x)
        print(range_y, range_x)
        print(ny, nx)    
        print('initial start values y,x')
        print(mtx1[1,0], mtx1[0,1])
        print('initial final values y,x')
        print(mtx1[-1,0], mtx1[0,-1])

    
    mtx2_xStart = mtx1[0,1] - int(zero_pad_x * nx) * step_x
    mtx2_xFin = mtx1[0,-1] + int(zero_pad_x * nx) * step_x
    mtx2_x = np.arange(mtx2_xStart, mtx2_xFin + 0.5*step_x, step=step_x)
    step_x_mtx2 = np.mean(np.diff(mtx2_x))
    
    mtx2_yStart = mtx1[1,0] - int(zero_pad_y * ny) * step_y
    mtx2_yFin = mtx1[-1,0] + int(zero_pad_y * ny) * step_y
    mtx2_y = np.arange(mtx2_yStart, mtx2_yFin + 0.5*step_y, step=step_y)
    step_y_mtx2 = np.mean(np.diff(mtx2_y))

    if(debug):
        range_x_mtx2 = mtx2_xFin - mtx2_xStart
        range_y_mtx2 = mtx2_yFin - mtx2_yStart

        print('final (step / range / npoints) y,x')        
        print(step_y_mtx2, step_x_mtx2)
        print(range_y_mtx2, range_x_mtx2)
        print(len(mtx2_y), len(mtx2_x))
        print('final start values y,x')
        print(mtx2_yStart, mtx2_xStart)
        print('final final values y,x')
        print(mtx2_yFin, mtx2_xFin)
    
    idx_xStart = np.abs(mtx1[0,1] - mtx2_x).argmin()
    idx_xFin = np.abs(mtx1[0,-1] - mtx2_x).argmin()
    idx_yStart = np.abs(mtx1[1,0] - mtx2_y).argmin()
    idx_yFin = np.abs(mtx1[-1,0] - mtx2_y).argmin()
    
    mtx2 = np.zeros((int((2*zero_pad_y + 1) * ny + 1), int((2*zero_pad_x + 1) * nx + 1)))
    mtx2[1:,0] = mtx2_y
    mtx2[0,1:] = mtx2_x
    mtx2[idx_yStart+1:idx_yFin+2, idx_xStart+1:idx_xFin+2] = mtx1[1:,1:]

    if(debug):
        
        plt.figure()
        plt.imshow(mtx1[1:,1:], extent=[mtx1[0,1], mtx1[0,-1], mtx1[1,0], mtx1[-1,0]])
        
        plt.figure()
        plt.imshow(mtx2[1:,1:], extent=[mtx2[0,1], mtx2[0,-1], mtx2[1,0], mtx2[-1,0]])
        
    return mtx2








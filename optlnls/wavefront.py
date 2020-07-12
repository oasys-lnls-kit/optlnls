#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 11:09:54 2020

@author: lordano


"""
import numpy as np
from scipy.optimize import curve_fit
from optlnls.constants import * 



def wrap_phase(phase_2D, phase_shift=1.0):
    wrapped_phase = (phase_2D + pi - phase_shift) % (2*pi) 
    return wrapped_phase - pi
    

def spherical_phase(xy, R, energy_eV, converging=-1, wrapped=0, flatten=0):
    wavelength = hc/energy_eV
    wavenumber = 2*pi/wavelength
    [x, y] = xy
    z = converging * wavenumber*(np.sqrt(R**2 - x**2 - y**2) - R)
    
    if(wrapped):
        z = wrap_phase(z)
    
    if(flatten):
        z = z.flatten()
        
    return z


# def spherical_phase_to_fit(xy, R, *args):

    # energy_eV = args[0]
    # converging = args[1]
    # wrapped = args[2]
    # flatten = args[3]    
    
    # return lambda xy, R : 


def cylindrical_phasex(xy, R, energy_eV, converging=-1, wrapped=0, flatten=0):
    wavelength = hc/energy_eV
    wavenumber = 2*pi/wavelength
    [x, y] = xy
    z = converging * wavenumber*(np.sqrt(R**2 - x**2) - R)

    if(wrapped):
        z = wrap_phase(z)
    
    if(flatten):
        z = z.flatten()

    return z

def cylindrical_phasey(xy, R, energy_eV, converging=-1, wrapped=0, flatten=0):
    wavelength = hc/energy_eV
    wavenumber = 2*pi/wavelength
    [x, y] = xy
    z = converging * wavenumber*(np.sqrt(R**2 - y**2) - R)
    
    if(wrapped):
        z = wrap_phase(z)
    
    if(flatten):
        z = z.flatten()
    
    return z


def write_phase_to_file(phase2D, filename='test.dat'):
    np.savetxt(filename, phase2D, fmt='%.6e')
    
def read_phase_from_file(filename):
    return np.genfromtxt(filename)

def crop_matrix(matrix2D, xmin, xmax, ymin, ymax):
    
    x = matrix2D[0,1:]
    y = matrix2D[1:,0]
    matrix = matrix2D[1:,1:]
    
    x_idx = np.where(np.logical_and(x >= xmin, x <= xmax))[0]
    y_idx = np.where(np.logical_and(y >= ymin, y <= ymax))[0]
    matrix_crop = matrix[y_idx[0]:y_idx[-1]+1, x_idx[0]:x_idx[-1]+1]
    
    matrix_cropped = np.zeros((len(y_idx)+1, len(x_idx)+1))
    matrix_cropped[0,1:] = x[x_idx[0]:x_idx[-1]+1]
    matrix_cropped[1:,0] = y[y_idx[0]:y_idx[-1]+1]
    matrix_cropped[1:,1:] = matrix_crop
    
    return matrix_cropped
    

def fit_spherical_phase(phase2D, energy_eV, converging=-1, wrapped=0, R_guess=2.0):

    x = phase2D[0,1:]*1e-3
    y = phase2D[1:,0]*1e-3
    phase = phase2D[1:,1:]

    xx, yy = np.meshgrid(x, y)    
    
    popt, pcov = curve_fit(lambda xy, R: spherical_phase(xy, R, energy_eV, converging, wrapped, 1), 
                           xdata=[xx, yy], ydata=phase.flatten(), p0=[R_guess],
                           method='trf', xtol=1e-15, max_nfev=50000)

    #spherical_fitted = spherical_phase([xx,yy], *popt).reshape((len(y),len(x)))

    return popt[0]


if __name__ == '__main__':
    
    from matplotlib import pyplot as plt

    if(0):

        x = np.linspace(-0.2e-3, 0.2e-3, 101)
        y = np.linspace(-0.1e-3, 0.1e-3, 51)
        
        xx, yy = np.meshgrid(x, y)
    
        phase = spherical_phase(xy=[xx, yy], R=2.0, energy_eV=3000, converging=1, wrapped=1)

        plt.figure()
        plt.imshow(phase)
        plt.show()

    if(1):    

        
        ############################
        ## READ PHASE FROM FILE
        ############################

        phase_test = read_phase_from_file('/home/lordano/Oasys/SPIE2020_simulations/Manaca/zernikes/test_phase2.dat') 
        phase_test[1:,1:] -= np.mean(phase_test[1:,1:])
    
        x = phase_test[0,1:]*1e-3
        y = phase_test[1:,0]*1e-3
        
        xx, yy = np.meshgrid(x, y)
    
        ############################
        ## PHASE WITH THEORETICAL RADIUS
        ############################
    
        phase = spherical_phase(xy=[xx, yy], R=2.0, energy_eV=12000, converging=1, wrapped=0)
        phase -= np.mean(phase)

        residual = phase_test[1:,1:] - phase
        residual -= np.mean(residual)
        
        ############################
        ## FIT PHASE
        ############################

        R_fitted = fit_spherical_phase(phase_test, energy_eV=12000, converging=1, wrapped=0, R_guess=1.0)
        print('Fitted R = ', R_fitted)

        phase_fitted = spherical_phase([xx,yy], R=R_fitted, energy_eV=12000, converging=1, wrapped=0)
        phase_fitted -= np.mean(phase_fitted)

        residual_fitted = phase_test[1:,1:] - phase_fitted
        residual_fitted -= np.mean(residual_fitted)

        ############################
        ## PLOTS
        ############################

        plt.figure()
        plt.title("SRW DATA")
        plt.imshow(phase_test[1:,1:], aspect='auto', origin='lower',
                    extent=[x.min()*1e6, x.max()*1e6, y.min()*1e6, y.max()*1e6])
        plt.colorbar()
        plt.show()

        plt.figure("Theoretical Radius")
        plt.imshow(phase, aspect='auto', origin='lower',
                    extent=[x.min()*1e6, x.max()*1e6, y.min()*1e6, y.max()*1e6])
        plt.colorbar()
        plt.show()

        plt.figure()
        plt.title("Fitted Phase")
        plt.imshow(phase_fitted, aspect='auto', origin='lower',
                    extent=[x.min()*1e6, x.max()*1e6, y.min()*1e6, y.max()*1e6])
        plt.colorbar()
        plt.show()
   


        plt.figure()
        plt.title("Theoretical Radius Residual")
        plt.imshow(residual, aspect='auto', origin='lower',
                    extent=[x.min()*1e6, x.max()*1e6, y.min()*1e6, y.max()*1e6])
        plt.colorbar()
        plt.show()

        plt.figure()
        plt.title("Fitted Residual")
        plt.imshow(residual_fitted, aspect='auto', origin='lower',
                    extent=[x.min()*1e6, x.max()*1e6, y.min()*1e6, y.max()*1e6])
        plt.colorbar()
        plt.show()








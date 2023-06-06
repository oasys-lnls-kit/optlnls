#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 11:09:54 2020

@author: lordano


"""
import numpy as np
from scipy.optimize import curve_fit
from optlnls.constants import * 
from matplotlib import pyplot as plt
import pickle


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


class xfw_transmission1D(object):
    
    def __init__(self, 
                 wavelength : float = 1e-10,
                 f : float = 0.5):
    
        self.f = f
        self.wavelength = wavelength
        
    def load_from_file(self, filename : str = '',        
                       pixel : float = 100e-9,
                       width_factor : float = 0.1,
                       sum_two_pi : bool = False):
        
        self.filename = filename
        self.pixel = pixel
        self.width_factor = width_factor
        
        # wavenumber = 2 * np.pi / wavelength
        colAmp = 3
        colPhase = 4
        
        # ### read data
        data = np.genfromtxt(filename)

        R = 1*data[0,1] # minimum aperture radius of the lens 
        da = self.width_factor * self.wavelength * self.f / R # slit width
        self.slit_width = da
        # Lmax = R * f / data[0,1]

        # ### limits that define slits region and map resolution
        rInf = data[-1,1] - 2*da
        rSup = data[0,1] + 2*da
        rInt = rSup - rInf
        rMed = 0.5*(rSup + rInf)
        
        #resolucao do mapa
        mapaRes = round(rSup/pixel)
        
        # ### define x coordinates
        self.x = np.linspace(0, rSup, mapaRes)

        mapaAmp = np.zeros(mapaRes)
        mapaPhase = np.zeros(mapaRes)

        # ### generate amp and phase map from table
        for i in range(mapaRes):
            #em cada pixel, calcular distancia ao centro (0,0) e aplicar perfil
            r = pixel*i
            if self.sqpeak(r,rMed,rInt) == 1:
                mapaAmp[i] = self.perfilr(r, da, data, colAmp)
                phase = self.perfilr(r, da, data, colPhase)
                if(sum_two_pi):
                    if(phase < 0):
                        phase += 2 * np.pi
                mapaPhase[i] = phase
                
        self.amplitude = mapaAmp
        self.phase = mapaPhase
        self.n = mapaRes

    #pico quadrado
    def sqpeak(self, x, x0, d):
        if abs(x-x0) > 0.5*d:
            return 0
        else:
            return 1    
        
    #perfil radial
    def perfilr(self, r, d, tabelaFendas, col):
        perfil = 0
        for linha in tabelaFendas:
            perfil += linha[col] * self.sqpeak(r,linha[1],d)
        return perfil


class wfr1D(object):
        
    def __init__(self, x : np.ndarray = None, 
                 amplitude : np.ndarray = None, 
                 phase : np.ndarray = None, 
                 field : np.ndarray = None,
                 z : int = 0,
                 wavelength : float = None,
                 energy : float = 10e3):
    
        self.x = x
        self.amplitude = amplitude
        self.phase = phase
        self.field = field
        self.z = z
        self.wavelength = wavelength
        self.energy = energy
        
        has_field = self.field is not None
        has_amplitude = self.field is not None
        has_phase = self.phase is not None
        
        no_field = not has_field
        no_amplitude = not has_amplitude
        no_phase = not has_phase
        
        if(no_field & no_amplitude & no_phase):
            self.amplitude = np.ones(1)
            self.phase = np.zeros(1)
            has_amplitude = True
            has_phase = True

        if(no_field & (has_amplitude & has_phase)):
            self.field = self.amplitude * np.exp(-1j * self.phase)
        if(has_field & (no_amplitude & no_phase)):
            self.amplitude = np.abs(self.field)
            self.phase = np.angle(self.field)
        
        if(self.wavelength is None):
            self.wavelength = 1.239842e-6 / self.energy
        else:
            self.energy = 1.239842e-6 / self.wavelength
    
        self.k = 2 * np.pi / self.wavelength # wavenumber
        self.intensity = self.amplitude**2
        
    def update_field(self):
        self.amplitude = np.abs(self.field)
        self.phase = np.angle(self.field)
        self.intensity = self.amplitude**2

    def gaussian_field(self, peak, sigma, x0=0):
        self.field = peak * np.exp(-(self.x - x0)**2 / (2*sigma**2)) + 0j       
        self.update_field()

    def uniform_field(self, peak):
        self.field = np.ones(len(self.x)) * peak  + 0j       
        self.update_field()
        
    def propagate_thin_lens(self, f):
        lens = np.exp(-1.0j * self.k * (self.x**2 / f) / 2)
        self.field = self.field * lens
        self.update_field()
        
    def propagate_distance_hankel(self, z):
        from pyhank import HankelTransform
        H = HankelTransform(order=0, radial_grid=self.x)
        field_H = H.to_transform_r(self.field) 
        field_Hk = H.qdht(field_H)
 
        kz = np.sqrt(self.k**2 - H.kr**2)
        phi = kz * z  # Propagation phase
        field_Hkz = field_Hk * np.exp(1j * phi)  # Apply propagation
        field_Hz = H.iqdht(field_Hkz)  # iQDHT
        self.field = H.to_original_r(field_Hz)  # Interpolate output

        self.z = self.z + z
        self.update_field()
        
    def run_caustics(self, zmin, zmax, nz):
        from pyhank import HankelTransform
        z = np.linspace(zmin, zmax, nz)
        nx = len(self.x)
        
        H = HankelTransform(order=0, radial_grid=self.x)
        field_H = H.to_transform_r(self.field) 
        field_Hk = H.qdht(field_H)
        field_z = np.zeros((nx, nz), dtype=complex)
        kz = np.sqrt(self.k**2 - H.kr**2)
        for i, zi in enumerate(z):
            phi = kz * zi  # Propagation phase
            field_Hkz = field_Hk * np.exp(1j * phi)  # Apply propagation
            field_Hz = H.iqdht(field_Hkz)  # iQDHT
            field_z[:,i] = H.to_original_r(field_Hz)  # Interpolate output
        
        caustics = np.zeros((nx+1, nz+1))
        caustics[0,1:] = z
        caustics[1:,0] = self.x
        caustics[1:,1:] = np.abs(field_z)**2

        return caustics
        
        
    def propagate_distance_fresnel(self, z):
        exp_ikz = np.exp( 1j * self.k * z)
        exp_quad_phase = np.exp( 1j * self.k / ( 2 * z ) * self.x**2 )
        kernel = (1 / (1j * self.wavelength * z)) * exp_ikz * exp_quad_phase
        self.field = fftconvolve(self.field, kernel, 'same')
        self.z = self.z + z
        self.update_field()
        
    def propagate_aperture(self, half_aperture):
        self.field[self.x >= half_aperture] = 0 + 0j
        self.update_field()
        
    def propagate_transmission(self, amplitude, phase):
        transmission = amplitude * np.exp(-1j * phase)
        self.field = self.field * transmission
        self.update_field()
        
    def pad_wfr(self, pad_factor, pad_value):

        x0 = self.x
        field = self.field
        nx0 = len(x0)

        step = np.mean(np.diff(x0))
        xmin = x0.min()
        x_pos = x0 - xmin

        nxf = int(round(x_pos.max() * pad_factor / step) + 1)
        nx_add = nxf - nx0 
                
        for i in range(nx_add):
            x_pos = np.append(x_pos, x_pos[-1] + step)
            field = np.append(field, pad_value)
            
        x_pos = x_pos + xmin
        
        self.x = x_pos
        self.field = field
        self.update_field()
        
    def plot_wfr(self, quantity='intensity', mirrored=1, 
                 title='', x_unit_factor=1e6, xlim=[],
                 yscale='linear', savepath=''):

        x = self.x        
        if(quantity == 'intensity'):
            y = self.intensity
            ylabel = 'Intensity [a.u.]'
        if(quantity == 'phase'):
            y = self.phase
            ylabel = 'Phase [rad]'
            
        if(x_unit_factor == 1e6):
            xlabel = 'X [\u03BCm]'
        if(x_unit_factor == 1e3):
            xlabel = 'X [mm]'
        if(x_unit_factor == 1):
            xlabel = 'X [m]'
        
        if(mirrored):
            x = np.concatenate((-1*x[::-1], x[1:]))
            y = np.concatenate((y[::-1], y[1:]))
        
        plt.figure(figsize=(4.5, 3.0))
        plt.subplots_adjust(0.15,0.15,0.97,0.92)
        plt.plot(x*x_unit_factor, y)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.minorticks_on()
        if(xlim != []):
            plt.xlim(xlim[0]*x_unit_factor, 
                     xlim[1]*x_unit_factor)
        plt.yscale(yscale)
        if(savepath != ''):
            plt.savefig(savepath, dpi=300)
            
    def save(self, filename_pkl):
        with open(filename_pkl, 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)
    
    def load(self, filename_pkl):
        with open(filename_pkl, 'rb') as inp:
            wfr = pickle.load(inp)
            return wfr




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








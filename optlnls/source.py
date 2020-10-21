#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 15:26:15 2020

@author: sergio.lordano
"""

import numpy
import numpy as np
from matplotlib import pyplot as plt
import scipy.special as spc

from scipy.interpolate import interp1d
from scipy.integrate import simps
from scipy.special import kv
from scipy.integrate import quad
import scipy.constants as codata


def get_k(Period, what_harmonic, Energy, k_ext):
    E = 3.0e9; e = 1.60217662e-19; m_e = 9.10938356e-31; pi = 3.141592654; c = 299792458; h_cut = 6.58211915e-16;
    gamma = E*e/(m_e*c**2)
    n = [2*i+1 for i in range(50)]
    for h_n in n:
        K2 =  (8*h_n*pi*h_cut*c*(gamma**2)/Period/Energy-2)
        if K2 > 0:
            if what_harmonic == 'max':            
                if K2**0.5 < k_ext:
                    har = h_n
                    k = K2**0.5
                else:                 
                    #print("Maximum Harmonic = ", har, ', K = ', k)
                    break
            elif what_harmonic == 'min':
                if K2**0.5 > k_ext:
                    har = h_n
                    k = K2**0.5
                    #print("Minimum Harmonic = ", har, ', K = ', k)
                    break     
    B = 2*pi*m_e*c*k/(e*Period)
    return har, k, B


def und_source(emmitance, Beta, e_spread, und_length, und_period, ph_energy, harmonic):
#    import scipy.special as spc
    
    emmitance = emmitance*1e12          #Vertical Emmitance [pm.rad]
    und_period = und_period*1e2         #Undulator period length[cm]
    und_length = und_length*1e2         #Undulator Length [cm]
    
    x = 2*np.pi*harmonic*und_length/und_period*e_spread
    a1 = np.sqrt(2*np.pi)*x*spc.erf(np.sqrt(2)*x)
    Qax = np.sqrt(2*x**2/(-1+np.exp(-2*x**2)+a1))
    sphp_h = np.sqrt(emmitance/Beta+(12.398e7/ph_energy)/(2*und_length)*Qax**2)             #Divergence
    
    a1s = np.sqrt(2*np.pi)*(x/4)*spc.erf(np.sqrt(2)*(x/4))
    Qas = (np.sqrt(2*(x/4)**2/(-1+np.exp(-2*(x/4)**2)+a1s)))**(2./3.)
    sph_h = np.sqrt(emmitance*Beta+und_length*12398/(2*(np.pi**2)*ph_energy)*(Qas)**2)      #Size
    
    return sph_h, sphp_h


def srw_undulator_spectrum(mag_field=[], electron_beam=[], energy_grid=[], sampling_mesh=[], precision=[]):
    """
    Calls SRW to calculate spectrum for a planar or elliptical undulator\n
    :mag_field: list containing: [period [m], length [m], Bx [T], By [T], phase Bx = 0, phase By = 0, Symmetry Bx = +1, Symmetry By = -1]
    :electron_beam: list containing: [Sx [m], Sy [m], Sx' [rad], Sy'[rad], Energy [GeV], Energy Spread [dE/E], Current [A]]
    :energy_grid: list containing: [initial energy, final energy, number of energy points]
    :sampling_mesh: list containing: [observation plane distance from source [m], range X [m], range Y [m]]
    :precision: list containing: [h_max: maximum harmonic number to take into account, longitudinal precision factor, azimuthal precision factor (1 is standard, >1 is more accurate]
    """    
        
    from srwlib import SRWLMagFldU, SRWLMagFldH, SRWLPartBeam, SRWLStokes
    from srwlpy import CalcStokesUR
    from numpy import array as nparray

    #***********Undulator
    und = SRWLMagFldU([SRWLMagFldH(1, 'v', mag_field[3], mag_field[5], mag_field[7], 1), 
                       SRWLMagFldH(1, 'h', mag_field[2], mag_field[4], mag_field[6], 1)], 
                       mag_field[0], int(round(mag_field[1]/mag_field[0])))
       
    #***********Electron Beam
    eBeam = SRWLPartBeam()
    eBeam.Iavg = electron_beam[6] #average current [A]
    eBeam.partStatMom1.x = 0. #initial transverse positions [m]
    eBeam.partStatMom1.y = 0.
    eBeam.partStatMom1.z = -(mag_field[1]/2 + mag_field[0]*2) #initial longitudinal positions (set in the middle of undulator)
    eBeam.partStatMom1.xp = 0 #initial relative transverse velocities
    eBeam.partStatMom1.yp = 0
    eBeam.partStatMom1.gamma = electron_beam[4]/0.51099890221e-03 #relative energy
    sigEperE = electron_beam[5] #0.00089 #relative RMS energy spread
    sigX = electron_beam[0] #33.33e-06 #horizontal RMS size of e-beam [m]
    sigXp = electron_beam[2] #16.5e-06 #horizontal RMS angular divergence [rad]
    sigY = electron_beam[1] #2.912e-06 #vertical RMS size of e-beam [m]
    sigYp = electron_beam[3] #2.7472e-06 #vertical RMS angular divergence [rad]
    #2nd order stat. moments:
    eBeam.arStatMom2[0] = sigX*sigX #<(x-<x>)^2> 
    eBeam.arStatMom2[1] = 0 #<(x-<x>)(x'-<x'>)>
    eBeam.arStatMom2[2] = sigXp*sigXp #<(x'-<x'>)^2> 
    eBeam.arStatMom2[3] = sigY*sigY #<(y-<y>)^2>
    eBeam.arStatMom2[4] = 0 #<(y-<y>)(y'-<y'>)>
    eBeam.arStatMom2[5] = sigYp*sigYp #<(y'-<y'>)^2>
    eBeam.arStatMom2[10] = sigEperE*sigEperE #<(E-<E>)^2>/<E>^2
    
    #***********Precision Parameters
    arPrecF = [0]*5 #for spectral flux vs photon energy
    arPrecF[0] = precision[0] #initial UR harmonic to take into account
    arPrecF[1] = precision[1] #final UR harmonic to take into account
    arPrecF[2] = precision[2] #longitudinal integration precision parameter
    arPrecF[3] = precision[3] #azimuthal integration precision parameter
    arPrecF[4] = 1 #calculate flux (1) or flux per unit surface (2)
        
    #***********UR Stokes Parameters (mesh) for Spectral Flux
    stkF = SRWLStokes() #for spectral flux vs photon energy
    stkF.allocate(energy_grid[2], 1, 1) #numbers of points vs photon energy, horizontal and vertical positions
    stkF.mesh.zStart = sampling_mesh[0] #longitudinal position [m] at which UR has to be calculated
    stkF.mesh.eStart = energy_grid[0] #initial photon energy [eV]
    stkF.mesh.eFin = energy_grid[1] #final photon energy [eV]
    stkF.mesh.xStart = -sampling_mesh[1]/2.0 #initial horizontal position [m]
    stkF.mesh.xFin = sampling_mesh[1]/2.0 #final horizontal position [m]
    stkF.mesh.yStart = -sampling_mesh[2]/2.0 #initial vertical position [m]
    stkF.mesh.yFin = sampling_mesh[2]/2.0 #final vertical position [m]
           
    
    #**********************Calculation (SRWLIB function calls)
    print('   Performing Spectral Flux (Stokes parameters) calculation ... ')
    CalcStokesUR(stkF, eBeam, und, arPrecF)
    print('done')
    
    return nparray(stkF.arS[0:energy_grid[2]])



def AuxReadInMagFld3D(filePath, sCom):
    
    """Function from SRW examples"""
    
    from array import array
    from srwlib import SRWLMagFld3D
    
    f = open(filePath, 'r')
    f.readline() #1st line: just pass
    global xStart,xStep,xNp,yStart,yStep,yNp,zStart,zStep,zNp
    xStart = float(f.readline().split(sCom, 2)[1]) #2nd line: initial X position [m]; it will not actually be used
    xStep = float(f.readline().split(sCom, 2)[1]) #3rd line: step vs X [m]
    xNp = int(f.readline().split(sCom, 2)[1]) #4th line: number of points vs X
    yStart = float(f.readline().split(sCom, 2)[1]) #5th line: initial Y position [m]; it will not actually be used
    yStep = float(f.readline().split(sCom, 2)[1]) #6th line: step vs Y [m]
    yNp = int(f.readline().split(sCom, 2)[1]) #7th line: number of points vs Y
    zStart = float(f.readline().split(sCom, 2)[1]) #8th line: initial Z position [m]; it will not actually be used
    zStep = float(f.readline().split(sCom, 2)[1]) #9th line: step vs Z [m]
    zNp = int(f.readline().split(sCom, 2)[1]) #10th line: number of points vs Z
    totNp = xNp*yNp*zNp
    locArBx = array('d', [0]*totNp)
    locArBy = array('d', [0]*totNp)
    locArBz = array('d', [0]*totNp)
    for i in range(totNp):
        curLineParts = f.readline().split('\t')
        locArBx[i] = float(curLineParts[0])
        locArBy[i] = float(curLineParts[1])
        locArBz[i] = float(curLineParts[2])
    f.close()
    xRange = xStep
    if xNp > 1: xRange = (xNp - 1)*xStep
    yRange = yStep
    if yNp > 1: yRange = (yNp - 1)*yStep
    zRange = zStep
    if zNp > 1: zRange = (zNp - 1)*zStep
        
    return SRWLMagFld3D(locArBx, locArBy, locArBz, xNp, yNp, zNp, xRange, yRange, zRange, 1)


def BM_spectrum(E, I, B, ph_energy, hor_acc_mrad=1.0):
    """
    Calculates the emitted spectrum of a Bending Magnet (vertically integrated) whithin a horizontal acceptance\n
    Units: [ph/s/0.1%bw]\n
    :E: Storage Ring energy [GeV]
    :I: Storage Ring current [A]
    :B: Magnetic Field value [T]    
    :ph_energy: Array of Photon Energies [eV]
    :hor_acc_mrad: Horizontal acceptance [mrad]
    """
    
    
    
    def bessel_f(y):
        return kv(5.0/3.0, y)    
        
    e_c = 665*(E**2)*B # eV
    y = ph_energy/e_c
    int_K53 = numpy.zeros((len(y)))
    for i in range(len(y)):
        int_K53[i] = quad(lambda x: kv(5.0/3.0, x), y[i], numpy.inf)[0]
    G1_y = y*int_K53
    BM_Flux = (2.457*1e13)*E*I*G1_y*hor_acc_mrad
    
    return BM_Flux


def Wiggler_spectrum(E, I, B, N_periods, ph_energy, hor_acc_mrad=1.0):
        """
        Calculates the emitted spectrum of a Wiggler (vertically integrated) whithin a horizontal acceptance\n
        Units: [ph/s/0.1%bw]\n
        :E: Storage Ring energy [GeV]
        :I: Storage Ring current [A]
        :B: Magnetic Field value [T]    
        :N_periods: Number of Periods
        :ph_energy: Array of Photon Energies [eV]
        :hor_acc_mrad: Horizontal acceptance [mrad]
        """
        
        def bessel_f(y):
            return kv(5.0/3.0, y)    
            
        e_c = 665*(E**2)*B # eV
        y = ph_energy/e_c
        int_K53 = numpy.zeros((len(y)))
        for i in range(len(y)):
            int_K53[i] = quad(lambda x: kv(5.0/3.0, x), y[i], numpy.inf)[0]
        G1_y = y*int_K53
        W_Flux = (2.457*1e13)*E*I*G1_y*hor_acc_mrad*(2*N_periods)
        
        return W_Flux    


def BM_vertical_acc(E=3.0, B=3.2, ph_energy=1915.2, div_limits=[-1.0e-3, 1.0e-3], e_beam_vert_div=0.0, plot=False):

    """
    Calculates the vertical angular flux probability density function (pdf) for \
    a Bending Magnet or Wiggler and compares it to divergence limits to calculate \
    the relative vertcal acceptance.\n
    Return: Dictionary containing vertical angular distribution, fwhm and acceptance factor (energy-dependent)  \n
    :E: Storage Ring energy [GeV]
    :B: Magnetic Field value [T]    
    :ph_energy: Photon Energy - single value or array - [eV]
    :div_limits: Divergence limits array for which acceptance must be calculated [rad]
    :e_beam_vert_div: electron beam vertical divergence sigma [rad]. Not taken into account if equal to None.
    :plot: boolean: True or False if you want the distribution to be shown.
    """
    import numpy
    from scipy.special import kv
    from scipy.integrate import simps
    
    def gaussian_pdf(x, x0, sigma): # gaussian probability density function (PDF)
        return (1/(numpy.sqrt(2*numpy.pi*sigma**2)))*numpy.exp(-(x-x0)**2/(2*sigma**2))
    
    def calc_vert_dist(e_relative):
        G = (e_relative/2.0)*(gamma_psi**(1.5))    
        K13_G = kv(1.0/3.0, G)
        K23_G = kv(2.0/3.0, G)
          
        dN_dOmega  = (1.33e13)*(E**2)*I*(e_relative**2)*(gamma_psi**2)
        dN_dOmega *= ( (K23_G**2) + (((gamma**2) * (psi**2))/(gamma_psi))*(K13_G**2) )   
        
        return dN_dOmega
    
    if(not(hasattr(ph_energy, "__len__"))): # for single energies
        ph_energy = numpy.array([ph_energy])
    
    I = 0.1 # [A] -> result independent
    gamma = E/0.51099890221e-03
    e_c = 665*(E**2)*B # [eV]    
    energy_relative = ph_energy/e_c
    
    # calculate graussian approximation to define psi mesh
    int_K53 = quad(lambda x: kv(5.0/3.0, x), energy_relative[0], numpy.inf)[0]
    K23 = kv(2.0/3.0, energy_relative[0]/2)
    vert_angle_sigma = numpy.sqrt(2*numpy.pi/3)/(gamma*energy_relative[0])*int_K53/((K23)**2)
    if(e_beam_vert_div > 0.0):
        vert_angle_sigma = numpy.sqrt(vert_angle_sigma**2 + e_beam_vert_div**2) # calculates gaussian PDF of the e-beam vertical divergence
    
    psi = numpy.linspace(-vert_angle_sigma*2, vert_angle_sigma*2, 1000) # vertical angle array
    gamma_psi = 1 + (gamma**2) * (psi**2) # factor dependent on gamma and psi
    psi_minus = numpy.abs(psi - div_limits[0]).argmin() # first psi limit index
    psi_plus = numpy.abs(psi - div_limits[1]).argmin() # second psi limit index
    
    vert_pdf = numpy.zeros((len(ph_energy), len(psi)))
    vert_acceptance = numpy.zeros((len(ph_energy)))
    lwhm = numpy.zeros((len(ph_energy)))
    rwhm = numpy.zeros((len(ph_energy)))
    fwhm = numpy.zeros((len(ph_energy)))
    
    if(e_beam_vert_div > 0.0):
        e_beam_pdf = gaussian_pdf(psi, 0, e_beam_vert_div) # calculates gaussian PDF of the e-beam vertical divergence
        
    for i in range(len(ph_energy)):
        vert_pdf[i] = calc_vert_dist(energy_relative[i]) 
        vert_pdf[i] /= simps(y=vert_pdf[i], x=psi)
        
        if(e_beam_vert_div > 0.0): # convolves radiation and e-beam angular distributions
            
            conv_dist = numpy.convolve(vert_pdf[i], e_beam_pdf, mode='same')
            conv_dist_norm = simps(y=conv_dist, x=psi)
            conv_pdf = conv_dist / conv_dist_norm # convolved PDF
            vert_pdf[i] = conv_pdf
            
        vert_acceptance[i] = simps(vert_pdf[i][psi_minus:psi_plus+1], x=psi[psi_minus:psi_plus+1])
        # calculates FWHM 
        peak = numpy.max(vert_pdf[i])
        peak_idx = numpy.abs(vert_pdf[i]-peak).argmin()
        lwhm[i] = psi[numpy.abs(vert_pdf[i][:peak_idx] - peak/2).argmin()]
        rwhm[i] = psi[numpy.abs(vert_pdf[i][peak_idx:] - peak/2).argmin() + peak_idx]
        fwhm[i] = rwhm[i] - lwhm[i]

   
    if(plot==True and len(vert_pdf)==1):
        from matplotlib import pyplot as plt
        plt.figure()
        plt.plot(psi*1e3, vert_pdf[0], 'C0.-')
        plt.ylabel('$Flux \ PDF$')
        plt.xlabel('$\psi \ [mrad]$')
        plt.ylim(0, numpy.max(vert_pdf)*1.1)
        plt.fill_between(psi*1e3, vert_pdf[i], where=numpy.logical_and(psi>=psi[psi_minus], psi<=psi[psi_plus]))
        plt.axvline(x=psi[psi_minus]*1e3)
        plt.axvline(x=psi[psi_plus]*1e3)
        plt.plot(lwhm*1e3, peak/2, 'C1+', markersize=12)
        plt.plot(rwhm*1e3, peak/2, 'C1+', markersize=12)
        plt.show()
        
    if(plot==True and len(vert_pdf)>1):
        from matplotlib import pyplot as plt
        plt.figure()
        plt.imshow(vert_pdf.transpose(), extent=[ph_energy[0], ph_energy[-1], psi[0]*1e3, psi[-1]*1e3], aspect='auto')
        plt.xlabel('$Energy \ [eV]$')
        plt.ylabel('$\psi \ [mrad]$')
        plt.plot(ph_energy, lwhm*1e3, '--', color='white', linewidth=1.0, alpha=0.4)
        plt.plot(ph_energy, rwhm*1e3, '--', color='white', linewidth=1.0, alpha=0.4)
        plt.axhline(y=div_limits[0]*1e3, color='gray', alpha=0.5)
        plt.axhline(y=div_limits[1]*1e3, color='gray', alpha=0.5)
        plt.minorticks_on()
        plt.ylim([-vert_angle_sigma*1.75e3, vert_angle_sigma*1.75e3])
        plt.show()

    output = {"Psi": psi,
              "PDF": vert_pdf,
              "acceptance": vert_acceptance,
              "lwhm": lwhm,
              "rwhm": rwhm,
              "fwhm": fwhm}

    return output


def undulator_B_to_K(B, period):
    
    e = 1.60217662e-19; m_e = 9.10938356e-31; pi = 3.141592654; c = 299792458; 
    return e * period * B / (2 * pi * m_e * c)

def undulator_K_to_B(K, period):
    
    e = 1.60217662e-19; m_e = 9.10938356e-31; pi = 3.141592654; c = 299792458; 
    return (2 * pi * m_e * c * K) /  (e * period)    

def undulator_K_to_E1(K, period, E_GeV):
    
    e = 1.60217662e-19; m_e = 9.10938356e-31; pi = 3.141592654; c = 299792458; h_cut = 6.58211915e-16;
    gamma = E_GeV*1e9*e/(m_e*c**2)
    E1 = 2 * gamma**2 * (2*pi*h_cut) * c / ( period * (1 + K**2 / 2) )
    
    return E1

def undulator_B_to_E1(B, period, E_GeV):
    
    e = 1.60217662e-19; m_e = 9.10938356e-31; pi = 3.141592654; c = 299792458; h_cut = 6.58211915e-16;
    gamma = E_GeV*1e9*e/(m_e*c**2)
    K = undulator_B_to_K(B, period)
    E1 = 2 * gamma**2 * (2*pi*h_cut) * c / ( period * (1 + K**2 / 2) )
    
    return E1


def undulator_E_to_K(energy, harmonic, period, E_GeV):
    
    e = 1.60217662e-19; m_e = 9.10938356e-31; pi = 3.141592654; c = 299792458; h_cut = 6.58211915e-16;
    gamma = E_GeV*1e9*e/(m_e*c**2)
    try:
        K = np.sqrt( (4 * harmonic * (2*pi*h_cut) * c * gamma**2 )/(period * energy) - 2 )
        return K
    except:
        print("invalid combination of parameters!")
        return 0
    
def undulator_E_to_phase(energy, harmonic, period, E_GeV, B_max, z0_mm=0):
    
    K = undulator_E_to_K(energy, harmonic, period, E_GeV)
    B = undulator_K_to_B(K, period)
    z = z0_mm + (period / np.pi) * np.arccos(B / B_max) * 1e3
    
    return z
    

def undulator_phase_to_B(phase, phase_offset, period, B_peak):
       
    B = B_peak * np.abs( np.cos(np.pi * (phase - phase_offset) / period) )
    
    return B



if __name__ == '__main__':
    
    print(undulator_E_to_phase(energy=14353, harmonic=5, period=0.022, E_GeV=2.955, B_max=0.7064, z0=-0.3165))
    
    # B = 0.71033
    # print(undulator_B_to_E1(B=B, period=0.022, E_GeV=3.0))
    
    # print(undulator_B_to_K(B=B, period=0.022))
    # print(undulator_K_to_B(K=1.438, period=0.022))
    # print(undulator_K_to_E1(K=1.438, period=0.022, E_GeV=3.0))


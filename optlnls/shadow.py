
from optlnls.source import srw_undulator_spectrum, BM_spectrum
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps

def calc_und_flux(beam, nbins, eBeamEnergy, eSpread, current, 
                  und_per, und_length, B, min_harmonic, max_harmonic, 
                  source_beam, accept_hor, accept_ver, 
                  z_srw=10.0, show_plots=0, verbose=1):
    

    
    beam_bl = beam # beam after beamline
    
    ##################################
    #### SPECTRUM AFTER BEAMLINE #####
    E_bl = beam_bl.getshonecol(11, nolost=1) # energy column
    I_bl = beam_bl.getshonecol(23, nolost=1) # intensity column
    E_b = np.linspace(np.min(E_bl), np.max(E_bl), nbins) # energy coordinates
    Spec_b = np.histogram(E_bl, nbins, weights=I_bl)[0] # ray intensity spectrum after beamline
    
    
    ##################################
    #### SPECTRUM AT SOURCE ##########
    E_s0 = beam_bl.getshonecol(11, nolost=0) # energy column
    # Discarding the rays outside energy limits defined by beamline:
    # Use only the rays where min(E_bl) < E_s0 < max(E_bl)     
    E_s = E_s0[np.logical_and(E_s0>=np.min(E_bl),E_s0<=np.max(E_bl))]
    I_s = np.ones((len(E_s))) # source rays with weight 1
    Spec_s = np.histogram(E_s, nbins, weights=I_s)[0] # ray intensity spectrum at source
    
    ##################################
    #### SHOW SPECTRA ################
    if(show_plots):
        plt.figure()
        plt.loglog(E_b, Spec_s)
        plt.loglog(E_b, Spec_b)
        plt.title('Energy histograms at Source and after BL')
        # plt.show()
    
    ##########################################
    #### CALCULATE BEAMLINE TRANSMITTANCE ####
    T_E = Spec_b / Spec_s 
    
    if(show_plots):
        plt.figure()
        plt.plot(E_b, T_E)
        plt.title('beamline transmittance T(E)')
        # plt.show()
    
    ###########################################
    #### CALCULATE UNDULATOR SPECTRUM - SRW ###
    
    # Setup SRW_UNDULATOR_SPECTRUM() parameters
    
    #   :mag_field: list containing: [period [m], length [m], Bx [T], By [T], phase Bx = 0, phase By = 0, Symmetry Bx = +1, Symmetry By = -1]
    #   :electron_beam: list containing: [Sx [m], Sy [m], Sx' [rad], Sy'[rad], Energy [GeV], Energy Spread [dE/E], Current [A]]
    #   :energy_grid: list containing: [initial energy, final energy, number of energy points]
    #   :sampling_mesh: list containing: [observation plane distance from source [m], range -X [m], , range+X [m], range -Y [m], range +Y [m]]
    #   :precision: list containing: [h_max: maximum harmonic number to take into account, longitudinal precision factor, azimuthal precision factor (1 is standard, >1 is more accurate]
    
    #und_N = int(round(und_length/und_per))
    mag_field=[und_per, und_length, 0, B, 0, 0, +1, +1]
    electron_beam=[source_beam[0], source_beam[1], source_beam[2], source_beam[3], eBeamEnergy, eSpread, current]
    energy_grid=[np.min(E_b), np.max(E_b), nbins]
    sampling_mesh=[z_srw, -z_srw*accept_hor/2, z_srw*accept_hor/2, -z_srw*accept_ver/2, z_srw*accept_ver/2]  # [56, -56*3.3e-05, 56*3.3e-05, -56*3.3e-05, 56*3.3e-05]    
    precision = [min_harmonic, max_harmonic, 1.0, 1.0]   
    
    
    und_spec = srw_undulator_spectrum(mag_field, electron_beam, energy_grid, sampling_mesh, precision)   
    BL_spec = und_spec*T_E
    
    if(show_plots):
        plt.figure()
        plt.plot(E_b, und_spec)
        plt.plot(E_b, BL_spec)
        plt.yscale('log')
        plt.title('Spectrum in [ph/s/0.1%/100mA]')
        plt.show()
    
    ##########################################
    #### CALCULATE TOTAL FLUX ################
    total_flux_s = simps(und_spec*(1000/E_b), x=E_b) # converting to 1 eV bandwidth 
    total_flux_b = simps(BL_spec*(1000/E_b), x=E_b) # converting to 1 eV bandwidth
    if(verbose):
        print("Total Flux at Source: {0:.3e} ph/s".format(total_flux_s))
        print("Total Flux after Beamline: {0:.3e} ph/s".format(total_flux_b))
    
    ##########################################
    #### CALCULATE TOTAL POWER ###############
    e_charge = 1.6021766208e-19
    total_power_s = simps(und_spec*(1000/E_b)*E_b*e_charge, x=E_b) # converting to 1 eV bandwidth 
    total_power_b = simps(BL_spec*(1000/E_b)*E_b*e_charge, x=E_b) # converting to 1 eV bandwidth
    if(verbose):
        print("Total Power at Source: {0:.3e} W".format(total_power_s))
        print("Total Power after Beamline: {0:.3e} W".format(total_power_b))
    
    outputs = dict()
    outputs['total flux propagated'] = total_flux_b
    outputs['total flux at source'] = total_flux_s
    outputs['total power propagated'] = total_power_b
    outputs['total power at source'] = total_power_s
    outputs['energy array'] = E_b
    outputs['transmission array'] = T_E    
    outputs['source spectrum'] = und_spec
    
    return outputs


def calc_BM_flux(beam, E, I, B, hor_acc_mrad, nbins, show_plots=0, verbose=0):
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import simps
    
    beam_bl = beam # beam after beamline
    
    ##################################
    #### SPECTRUM AFTER BEAMLINE #####
    E_bl = beam_bl.getshonecol(11, nolost=1) # energy column
    I_bl = beam_bl.getshonecol(23, nolost=1) # intensity column
    E_b = np.linspace(np.min(E_bl), np.max(E_bl), nbins) # energy coordinates
    Spec_b = np.histogram(E_bl, nbins, weights=I_bl)[0] # ray intensity spectrum after beamline
    
    
    ##################################
    #### SPECTRUM AT SOURCE ##########
    E_s0 = beam_bl.getshonecol(11, nolost=0) # energy column
    # Discarding the rays outside energy limits defined by beamline:
    # Use only the rays where min(E_bl) < E_s0 < max(E_bl)     
    E_s = E_s0[np.logical_and(E_s0>=np.min(E_bl),E_s0<=np.max(E_bl))]
    I_s = np.ones((len(E_s))) # source rays with weight 1
    Spec_s = np.histogram(E_s, nbins, weights=I_s)[0] # ray intensity spectrum at source
    
    ##################################
    #### SHOW SPECTRA ################
    if(show_plots):
        plt.figure()
        plt.loglog(E_b, Spec_s)
        plt.loglog(E_b, Spec_b)
        plt.title('Energy histograms at Source and after BL')
        # plt.show()
    
    ##########################################
    #### CALCULATE BEAMLINE TRANSMITTANCE ####
    T_E = Spec_b / Spec_s 
    
    if(show_plots):
        plt.figure()
        plt.plot(E_b, T_E)
        plt.title('beamline transmittance T(E)')
        # plt.show()
    
    ###########################################
    #### CALCULATE BENDING MAGNET SPECTRUM ####   
    BM_spec = BM_spectrum(E, I, B, E_b, hor_acc_mrad)   
    BL_spec = BM_spec*T_E
    
    if(show_plots):
        plt.figure()
        plt.plot(E_b, BM_spec)
        plt.plot(E_b, BL_spec)
        plt.yscale('log')
        plt.title('Spectrum in [ph/s/0.1%/100mA]')
        plt.show()
    
    ##########################################
    #### CALCULATE TOTAL FLUX ################
    total_flux_s = simps(BM_spec*(1000/E_b), x=E_b) # converting to 1 eV bandwidth 
    total_flux_b = simps(BL_spec*(1000/E_b), x=E_b) # converting to 1 eV bandwidth
    if(verbose):
        print("Total Flux at Source: {0:.3e} ph/s".format(total_flux_s))
        print("Total Flux after Beamline: {0:.3e} ph/s".format(total_flux_b))
    
    ##########################################
    #### CALCULATE TOTAL POWER ###############
    e_charge = 1.6021766208e-19
    total_power_s = simps(BM_spec*(1000/E_b)*E_b*e_charge, x=E_b) # converting to 1 eV bandwidth 
    total_power_b = simps(BL_spec*(1000/E_b)*E_b*e_charge, x=E_b) # converting to 1 eV bandwidth
    if(verbose):
        print("Total Power at Source: {0:.3e} W".format(total_power_s))
        print("Total Power after Beamline: {0:.3e} W".format(total_power_b))
    
    outputs = dict()
    outputs['total flux propagated'] = total_flux_b
    outputs['total flux at source'] = total_flux_s
    outputs['total power propagated'] = total_power_b
    outputs['total power at source'] = total_power_s
    outputs['energy array'] = E_b
    outputs['transmission array'] = T_E    
    outputs['source spectrum'] = BM_spec
    
    return outputs


# def srw_undulator_spectrum(mag_field=[], electron_beam=[], energy_grid=[], sampling_mesh=[], precision=[]):

#     import sys                                                                                                                                                                                          
#     sys.path.insert(0, '/home/ABTLUS/humberto.junior/SRW/env/work/srw_python')        
#     from srwlib import SRWLMagFldU, SRWLMagFldH, SRWLPartBeam, SRWLStokes
#     from srwlpy import CalcStokesUR
#     import numpy as np
    

#     """
#     Calls SRW to calculate spectrum for a planar or elliptical undulator\n
#     :mag_field: list containing: [period [m], length [m], Bx [T], By [T], phase Bx = 0, phase By = 0, Symmetry Bx = +1, Symmetry By = -1]
#     :electron_beam: list containing: [Sx [m], Sy [m], Sx' [rad], Sy'[rad], Energy [GeV], Energy Spread [dE/E], Current [A]]
#     :energy_grid: list containing: [initial energy, final energy, number of energy points]
#     :sampling_mesh: list containing: [observation plane distance from source [m], range -X [m], , range+X [m], range -Y [m], range +Y [m]]
#     :precision: list containing: [h_max: maximum harmonic number to take into account, longitudinal precision factor, azimuthal precision factor (1 is standard, >1 is more accurate]
#     """     

#     #***********Undulator
#     und = SRWLMagFldU([SRWLMagFldH(1, 'v', mag_field[3], mag_field[5], mag_field[7], 1), 
#                        SRWLMagFldH(1, 'h', mag_field[2], mag_field[4], mag_field[6], 1)], 
#                        mag_field[0], int(round(mag_field[1]/mag_field[0])))
       
#     #***********Electron Beam
#     eBeam = SRWLPartBeam()
#     eBeam.Iavg = electron_beam[6] #average current [A]
#     eBeam.partStatMom1.x = 0. #initial transverse positions [m]
#     eBeam.partStatMom1.y = 0.
#     eBeam.partStatMom1.z = -(mag_field[1]/2 + mag_field[0]*2) #initial longitudinal positions (set in the middle of undulator)
#     eBeam.partStatMom1.xp = 0 #initial relative transverse velocities
#     eBeam.partStatMom1.yp = 0
#     eBeam.partStatMom1.gamma = electron_beam[4]/0.51099890221e-03 #relative energy
#     sigEperE = electron_beam[5] #0.00089 #relative RMS energy spread
#     sigX = electron_beam[0] #33.33e-06 #horizontal RMS size of e-beam [m]
#     sigXp = electron_beam[2] #16.5e-06 #horizontal RMS angular divergence [rad]
#     sigY = electron_beam[1] #2.912e-06 #vertical RMS size of e-beam [m]
#     sigYp = electron_beam[3] #2.7472e-06 #vertical RMS angular divergence [rad]
#     #2nd order stat. moments:
#     eBeam.arStatMom2[0] = sigX*sigX #<(x-<x>)^2> 
#     eBeam.arStatMom2[1] = 0 #<(x-<x>)(x'-<x'>)>
#     eBeam.arStatMom2[2] = sigXp*sigXp #<(x'-<x'>)^2> 
#     eBeam.arStatMom2[3] = sigY*sigY #<(y-<y>)^2>
#     eBeam.arStatMom2[4] = 0 #<(y-<y>)(y'-<y'>)>
#     eBeam.arStatMom2[5] = sigYp*sigYp #<(y'-<y'>)^2>
#     eBeam.arStatMom2[10] = sigEperE*sigEperE #<(E-<E>)^2>/<E>^2
    
#     #***********Precision Parameters
#     arPrecF = [0]*5 #for spectral flux vs photon energy
#     arPrecF[0] = 1 #initial UR harmonic to take into account
#     arPrecF[1] = precision[0] #final UR harmonic to take into account
#     arPrecF[2] = precision[1] #longitudinal integration precision parameter
#     arPrecF[3] = precision[2] #azimuthal integration precision parameter
#     arPrecF[4] = 1 #calculate flux (1) or flux per unit surface (2)
        
#     #***********UR Stokes Parameters (mesh) for Spectral Flux
#     stkF = SRWLStokes() #for spectral flux vs photon energy
#     stkF.allocate(energy_grid[2], 1, 1) #numbers of points vs photon energy, horizontal and vertical positions
#     stkF.mesh.zStart = sampling_mesh[0] #longitudinal position [m] at which UR has to be calculated
#     stkF.mesh.eStart = energy_grid[0] #initial photon energy [eV]
#     stkF.mesh.eFin = energy_grid[1] #final photon energy [eV]
#     stkF.mesh.xStart = sampling_mesh[1] #initial horizontal position [m]
#     stkF.mesh.xFin = sampling_mesh[2] #final horizontal position [m]
#     stkF.mesh.yStart = sampling_mesh[3] #initial vertical position [m]
#     stkF.mesh.yFin = sampling_mesh[4] #final vertical position [m]
           
    
#     #**********************Calculation (SRWLIB function calls)
#     #print('   Performing Spectral Flux (Stokes parameters) calculation ... ')
#     CalcStokesUR(stkF, eBeam, und, arPrecF)
#     #print('done')
    
#     return np.array(stkF.arS[0:energy_grid[2]])


# def BM_spectrum(E, I, B, ph_energy, hor_acc_mrad=1.0):
    
#         """
#         Calculates the emitted spectrum of a Bending Magnet (vertically integrated) whithin a horizontal acceptance\n
#         Units: [ph/s/0.1%bw]\n
#         :E: Storage Ring energy [GeV]
#         :I: Storage Ring current [A]
#         :B: Magnetic Field value [T]    
#         :ph_energy: Array of Photon Energies [eV]
#         :hor_acc_mrad: Horizontal acceptance [mrad]
#         """
        
#         from scipy.integrate import quad
#         from scipy.special import kv
#         import numpy
        
#         def bessel_f(y):
#             return kv(5.0/3.0, y)    
            
#         e_c = 665*(E**2)*B # eV
#         y = ph_energy/e_c
#         int_K53 = numpy.zeros((len(y)))
#         for i in range(len(y)):
#             int_K53[i] = quad(lambda x: kv(5.0/3.0, x), y[i], numpy.inf)[0]
#         G1_y = y*int_K53
#         BM_Flux = (2.457*1e13)*E*I*G1_y*hor_acc_mrad
        
#         return BM_Flux
    
    


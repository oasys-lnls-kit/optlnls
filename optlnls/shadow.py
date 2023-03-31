
from optlnls.source import srw_undulator_spectrum, BM_spectrum, BM_vertical_acc
from optlnls.math import get_fwhm, weighted_avg_and_std
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.integrate import simps
import time
import h5py
import os
import glob

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


def calc_wiggler_flux(beam, E, I, B, N_periods, hor_acc_mrad, nbins, 
                      vert_acc_mrad=0, e_beam_vert_div=1e-6, show_plots=0, verbose=0):
    
    WG_flux = calc_BM_flux(beam, E, I, B, hor_acc_mrad, 
                        nbins, vert_acc_mrad, e_beam_vert_div)
    WG_flux['total flux at source'] = WG_flux['total flux at source'] * (2*N_periods)
    WG_flux['total flux propagated'] = WG_flux['total flux propagated'] * (2*N_periods)
    WG_flux['total power at source'] = np.nan
    WG_flux['total power propagated'] = np.nan
    
    return WG_flux


def calc_BM_flux(beam, E, I, B, hor_acc_mrad, nbins, 
                 vert_acc_mrad=0, e_beam_vert_div=1e-6, show_plots=0, verbose=0):
    
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
    if(vert_acc_mrad > 0):
        acc = BM_vertical_acc(E, B, E_b, 
                              div_limits=[-vert_acc_mrad/2/1000, 
                                          vert_acc_mrad/2/1000],
                              e_beam_vert_div=e_beam_vert_div)
        vert_acc_factor = acc['acceptance']
        
        # print('\n\n****************************')
        # print('Energy[eV] vs Acceptance factor')
        # for i in range(len(E_b)):
        #     print('{0:.2f}, {1:.6f}'.format(E_b[i], vert_acc_factor[i]))
    else:
        vert_acc_factor = 1
    
    BM_spec = BM_spectrum(E, I, B, E_b, hor_acc_mrad) * vert_acc_factor   
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


def get_good_ranges(beam, zStart, zFin, colh, colv):
    
    r_z0h = beam.get_good_range(icol=colh, nolost=1)
    r_z0v = beam.get_good_range(icol=colv, nolost=1)
    
    beam_copy = beam.duplicate()
    beam_copy.retrace(zStart)
    r_zStarth = beam_copy.get_good_range(icol=colh, nolost=1)
    r_zStartv = beam_copy.get_good_range(icol=colv, nolost=1)
    
    beam_copy = beam.duplicate()
    beam_copy.retrace(zFin)
    r_zFinh = beam_copy.get_good_range(icol=colh, nolost=1)
    r_zFinv = beam_copy.get_good_range(icol=colv, nolost=1)
    
    rh_min = np.min(r_z0h + r_zStarth + r_zFinh)
    rh_max = np.max(r_z0h + r_zStarth + r_zFinh)
    rv_min = np.min(r_z0v + r_zStartv + r_zFinv)
    rv_max = np.max(r_z0v + r_zStartv + r_zFinv)

    return [rh_min, rh_max, rv_min, rv_max] 


def initialize_hdf5(h5_filename, zStart, zFin, nz, zOffset, colh, colv, colref, nbinsh, nbinsv, good_rays, offsets=None):
    with h5py.File(h5_filename, 'w') as f:
        f.attrs['begin time'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        f.attrs['zStart'] = zStart
        f.attrs['zFin'] = zFin
        f.attrs['nz'] = nz
        f.attrs['zOffset'] = zOffset
        f.attrs['zStep'] = int((zFin - zStart) / (nz - 1))
        f.attrs['col_h'] = colh
        f.attrs['col_v'] = colv
        f.attrs['col_ref'] = colref
        f.attrs['nbins_h'] = nbinsh
        f.attrs['nbins_v'] = nbinsv
        f.attrs['good_rays'] = good_rays
        if offsets is not None:
            f.attrs['offsets'] = offsets
        group = f.create_group('datasets')

def append_dataset_hdf5(filename, data, z, nz, tag, t0, ndigits):
    
    mean_h, rms_h = weighted_avg_and_std(data['bin_h_center'], data['histogram_h']) 
    mean_v, rms_v = weighted_avg_and_std(data['bin_v_center'], data['histogram_v'])
    fwhm_h = get_fwhm(data['bin_h_center'], data['histogram_h'])
    fwhm_v = get_fwhm(data['bin_v_center'], data['histogram_v'])
    
    with h5py.File(filename, 'a') as f:
        dset = f['datasets'].create_dataset('step_{0:0{ndigits}d}'.format(tag, ndigits=ndigits),
                                            data=np.array(data['histogram'], dtype=np.float), 
                                            compression="gzip")
        dset.attrs['z'] = z 
        dset.attrs['xStart'] = data['bin_h_center'].min()
        dset.attrs['xFin'] = data['bin_h_center'].max()
        dset.attrs['nx'] = data['nbins_h']
        dset.attrs['yStart'] = data['bin_v_center'].min()
        dset.attrs['yFin'] = data['bin_v_center'].max()
        dset.attrs['ny'] = data['nbins_v']
        dset.attrs['mean_h'] = mean_h
        dset.attrs['mean_v'] = mean_v
        dset.attrs['rms_h'] = rms_h
        dset.attrs['rms_v'] = rms_v
        dset.attrs['fwhm_h'] = fwhm_h
        dset.attrs['fwhm_v'] = fwhm_v
        dset.attrs['ellapsed time (s)'] = round(time.time() - t0, 3)
        
        try:
            dset.attrs['fwhm_h_shadow'] = data['fwhm_h']
            dset.attrs['center_h_shadow'] = (data['fwhm_coordinates_h'][0] + data['fwhm_coordinates_h'][1]) / 2.0
        except:
            print('CAUSTIC WARNING: FWHM X could not be calculated by Shadow at z position = {0:.3f}'.format(z[0]))
            dset.attrs['fwhm_h_shadow'] = np.nan
            dset.attrs['center_h_shadow'] = np.nan
        try:
            dset.attrs['fwhm_v_shadow'] = data['fwhm_v']
            dset.attrs['center_v_shadow'] = (data['fwhm_coordinates_v'][0] + data['fwhm_coordinates_v'][1]) / 2.0
        except:
            print('CAUSTIC WARNING: FWHM Y could not be calculated by Shadow at z position = {0:.3f}'.format(z[0]))
            dset.attrs['fwhm_v_shadow'] = np.nan
            dset.attrs['center_v_shadow'] = np.nan
            
        if (tag == nz - 1):
            f.attrs['end time'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def read_caustic(filename, write_attributes=False, plot=False, plot2D=False, 
                 print_minimum=False, cmap='viridis', figprefix=''):
    
    with h5py.File(filename, 'r+') as f:
    
        g = f['datasets']
    
        dset_names = list(g.keys())
        
        center_shadow = np.zeros((len(dset_names), 2), dtype=float)
        center = np.zeros((len(dset_names), 2), dtype=float)
        rms = np.zeros((len(dset_names), 2), dtype=float)
        fwhm = np.zeros((len(dset_names), 2), dtype=float)
        fwhm_shadow = np.zeros((len(dset_names), 2), dtype=float)
        
        ###### READ DATA #######################
        zOffset = f.attrs['zOffset']        
        zStart = f.attrs['zStart']
        zFin = f.attrs['zFin']
        nz = f.attrs['nz']
                
        #if(plot2D): 
        xStart = g[dset_names[0]].attrs['xStart']
        xFin = g[dset_names[0]].attrs['xFin']
        nx = g[dset_names[0]].attrs['nx']
        yStart = g[dset_names[0]].attrs['yStart']
        yFin = g[dset_names[0]].attrs['yFin']
        ny = g[dset_names[0]].attrs['ny']
            
        histoH = np.zeros((nx, nz))
        histoV = np.zeros((ny, nz))
        
        z_points = np.linspace(zStart, zFin, nz) + zOffset
        
        for i, dset in enumerate(dset_names):
            #dset_keys = list(f[dset].attrs.keys())
        
            center_shadow[i,0] = g[dset].attrs['center_h_shadow']
            center_shadow[i,1] = g[dset].attrs['center_v_shadow']
            center[i,0] = g[dset].attrs['mean_h']
            center[i,1] = g[dset].attrs['mean_v']
            rms[i,0] = g[dset].attrs['rms_h']
            rms[i,1] = g[dset].attrs['rms_v']
            fwhm[i,0] = g[dset].attrs['fwhm_h'][0]
            fwhm[i,1] = g[dset].attrs['fwhm_v'][0]
            fwhm_shadow[i,0] = g[dset].attrs['fwhm_h_shadow']
            fwhm_shadow[i,1] = g[dset].attrs['fwhm_v_shadow']
            
            #if(plot2D):
            histo2D = np.array(g[dset])
            histoH[:,i] = histo2D.sum(axis=1)
            histoV[:,i] = histo2D.sum(axis=0)
                
    #### FIND MINIMUMS AND ITS Z POSITIONS

    rms_min = [np.min(rms[:,0]), np.min(rms[:,1])]
    fwhm_min = [np.min(fwhm[:,0]), np.min(fwhm[:,1])]
    fwhm_shadow_min = [np.min(fwhm_shadow[:,0]), np.min(fwhm_shadow[:,1])]

    rms_min_z=np.array([z_points[np.abs(rms[:,0]-rms_min[0]).argmin()],
                        z_points[np.abs(rms[:,1]-rms_min[1]).argmin()]])

    fwhm_min_z=np.array([z_points[np.abs(fwhm[:,0]-fwhm_min[0]).argmin()],
                         z_points[np.abs(fwhm[:,1]-fwhm_min[1]).argmin()]])

    fwhm_shadow_min_z=np.array([z_points[np.abs(fwhm_shadow[:,0]-fwhm_shadow_min[0]).argmin()],
                                z_points[np.abs(fwhm_shadow[:,1]-fwhm_shadow_min[1]).argmin()]])

    center_rms = np.array([center[:,0][np.abs(z_points-rms_min_z[0]).argmin()],
                           center[:,1][np.abs(z_points-rms_min_z[1]).argmin()]])

    center_fwhm = np.array([center[:,0][np.abs(z_points-fwhm_min_z[0]).argmin()],
                            center[:,1][np.abs(z_points-fwhm_min_z[1]).argmin()]])

    center_fwhm_shadow = np.array([center[:,0][np.abs(z_points-fwhm_shadow_min_z[0]).argmin()],
                                   center[:,1][np.abs(z_points-fwhm_shadow_min_z[1]).argmin()]])
 
    
    outdict = {'xStart': xStart,
               'xFin': xFin,
               'nx': nx,
               'yStart': yStart,
               'yFin': yFin,
               'ny': ny,
               'zStart': zStart,
               'zFin': zFin,
               'nz': nz,
               'zOffset': zOffset,
               'center_h_array': center[:,0], 
               'center_v_array': center[:,1],
               'center_shadow_h_array': center_shadow[:,0], 
               'center_shadow_v_array': center_shadow[:,1],
               'rms_h_array': rms[:,0], 
               'rms_v_array': rms[:,1],
               'fwhm_h_array': fwhm[:,0], 
               'fwhm_v_array': fwhm[:,1],
               'fwhm_shadow_h_array': fwhm_shadow[:,0], 
               'fwhm_shadow_v_array': fwhm_shadow[:,1],
               'rms_min_h': rms_min[0],
               'rms_min_v': rms_min[1],
               'fwhm_min_h': fwhm_min[0],
               'fwhm_min_v': fwhm_min[1],
               'fwhm_shadow_min_h': fwhm_shadow_min[0],
               'fwhm_shadow_min_v': fwhm_shadow_min[1],
               'z_rms_min_h': rms_min_z[0],
               'z_rms_min_v': rms_min_z[1],
               'z_fwhm_min_h': fwhm_min_z[0],
               'z_fwhm_min_v': fwhm_min_z[1],
               'z_fwhm_shadow_min_h': fwhm_shadow_min_z[0],
               'z_fwhm_shadow_min_v': fwhm_shadow_min_z[1],
               'center_rms_h': center_rms[0],
               'center_rms_v': center_rms[1],
               'center_fwhm_h': center_fwhm[0],
               'center_fwhm_v': center_fwhm[1],
               'center_fwhm_shadow_h': center_fwhm_shadow[0],
               'center_fwhm_shadow_v': center_fwhm_shadow[1],
               }
    
    if(write_attributes):
        with h5py.File(filename, 'a') as f:
            for key in list(outdict.keys()):
                f.attrs[key] = outdict[key]
                
            f.create_dataset('histoXZ', data=histoH, dtype=np.float, compression="gzip")
            f.create_dataset('histoYZ', data=histoV, dtype=np.float, compression="gzip")
            
    if(print_minimum):
        print('\n   ****** \n' + '   Z min (rms-hor): {0:.3e}'.format(rms_min_z[0]))
        print('   Z min (rms-vert): {0:.3e}\n   ******'.format(rms_min_z[1]))
        
    if(plot):
        
        plt.figure()
        plt.title('rms')
        plt.plot(z_points, rms[:,0], label='rms_h')
        plt.plot(z_points, rms[:,1], label='rms_v')
        plt.legend()
        plt.minorticks_on()
        plt.grid(which='both', alpha=0.2)    
    
        plt.figure()
        plt.title('fwhm')
        plt.plot(z_points, fwhm[:,0], label='fwhm_h')
        plt.plot(z_points, fwhm[:,1], label='fwhm_v')
        # plt.plot(z_points, fwhm_shadow[:,0], label='fwhm_h_shadow')
        # plt.plot(z_points, fwhm_shadow[:,1], label='fwhm_v_shadow')
        plt.legend()
        plt.minorticks_on()
        plt.grid(which='both', alpha=0.2)    
        
        plt.figure()
        plt.title('center')
        plt.plot(z_points, center[:,0], label='center_h')
        # plt.plot(z_points, center_shadow[:,0], label='center_h_shadow')
        plt.legend()
        plt.minorticks_on()
        plt.grid(which='both', alpha=0.2)    
        
        plt.figure()
        plt.title('center')
        plt.plot(z_points, center[:,1], label='center_v')
        # plt.plot(z_points, center_shadow[:,1], label='center_v_shadow')
        plt.legend()
        plt.minorticks_on()
        plt.grid(which='both', alpha=0.2)    
            
        plt.show()
        
    if(plot2D):
        
        extHZ = [zStart+zOffset, zFin+zOffset, xStart, xFin]
        extVZ = [zStart+zOffset, zFin+zOffset, yStart, yFin]
        
        plt.figure(figsize=(6,2))
        plt.subplots_adjust(0.13, 0.22, 0.97, 0.95)
        # plt.title('XZ')
        plt.imshow(histoH, extent=extHZ, origin='lower', aspect='auto', cmap=cmap)
        plt.xlabel('Z [mm]')
        plt.ylabel('Horizontal [mm]')
        plt.minorticks_on()
        plt.tick_params(which='both', axis='both', top=True, right=True)
        if(figprefix != ''):
            plt.savefig(figprefix + '_XZ.png', dpi=300)
    
        plt.figure(figsize=(6,2))
        plt.subplots_adjust(0.13, 0.22, 0.97, 0.95)    
        # plt.title('YZ')
        plt.imshow(histoV, extent=extVZ, origin='lower', aspect='auto', cmap=cmap)
        plt.xlabel('Z [mm]')
        plt.ylabel('Vertical [mm]')
        plt.minorticks_on()
        plt.tick_params(which='both', axis='both', top=True, right=True)
        if(figprefix != ''):
            plt.savefig(figprefix + '_YZ.png', dpi=300)
        
    # if(plot2D == 'log'):
        
    #     xc_min_except_0 = np.min(histoH[histoH>0])
    #     histoH[histoH<=0.0] = xc_min_except_0/2.0
        
    #     plt.figure()
    #     plt.title('XZ')
    #     plt.imshow(histoH, extent=[zStart, zFin, xStart, xFin], origin='lower', aspect='auto',
    #                norm=LogNorm(vmin=xc_min_except_0/2.0, vmax=np.max(histoH)))
    #     plt.xlabel('Z')
    #     plt.ylabel('Horizontal')

    #     yc_min_except_0 = np.min(histoV[histoV>0])
    #     histoV[histoV<=0.0] = yc_min_except_0/2.0

    #     plt.figure()
    #     plt.title('YZ')
    #     plt.imshow(histoV, extent=[zStart, zFin, yStart, yFin], origin='lower', aspect='auto',
    #                norm=LogNorm(vmin=xc_min_except_0/2.0, vmax=np.max(histoV)))
    #     plt.xlabel('Z')
    #     plt.ylabel('Vertical')
    
    
    return histoH, histoV, outdict


def plot_caustic(caustic, caustic_dict, figprefix='', cmap='viridis'):
    
    c = caustic_dict
    
    histoHZ = c['histoHZ']
    extHZ = [c['zStart']+c['zOffset'], c['zFin']+c['zOffset'], c['xStart'], c['xFin']]
        
    histoVZ = c['histoVZ']
    extVZ = [c['zStart']+c['zOffset'], c['zFin']+c['zOffset'], c['yStart'], c['yFin']]
    

    plt.figure(figsize=(6,2))
    plt.subplots_adjust(0.13, 0.15, 0.97, 0.93)
    # plt.title('XZ')
    plt.imshow(histoHZ, extent=extHZ, origin='lower', aspect='auto', cmap=cmap)
    plt.xlabel('Z')
    plt.ylabel('Horizontal')
    plt.minorticks_on()
    plt.tick_params(top=True, right=True)
    if(figprefix != ''):
        plt.savefig(figprefix + '_XZ.png', dpi=300)

    plt.figure(figsize=(6,2))
    plt.subplots_adjust(0.13, 0.15, 0.97, 0.93)    
    # plt.title('YZ')
    plt.imshow(histoVZ, extent=extVZ, origin='lower', aspect='auto', cmap=cmap)
    plt.xlabel('Z')
    plt.ylabel('Vertical')
    plt.minorticks_on()
    plt.tick_params(top=True, right=True)
    if(figprefix != ''):
        plt.savefig(figprefix + '_YZ.png', dpi=300)


        
    
    
            
def run_shadow_caustic(filename, beam, zStart, zFin, nz, zOffset, colh, colv, colref, nbinsh, nbinsv, xrange, yrange):

    t0 = time.time()
    good_rays = beam.nrays(nolost=1)
    initialize_hdf5(filename, zStart, zFin, nz, zOffset, colh, colv, colref, nbinsh, nbinsv, good_rays)
    
    z_points = np.linspace(zStart, zFin, nz) + zOffset
    for i in range(nz):        
        beam.retrace(z_points[i]);
        
        # (col_v,col_h,weights) = beam.getshcol((colv,colh,colref),nolost=1)
        # (hh,yy,xx) = np.histogram2d(col_v, col_h, bins=[nbinsv,nbinsh], range=[yrange,xrange], 
        #                             normed=False, weights=weights)

        # histo = dict()
        # histo['col_h'] = colh
        # histo['col_v'] = colv
        # histo['nolost'] = 1
        # histo['nbins_h'] = nbinsh
        # histo['nbins_v'] = nbinsv
        # histo['ref'] = colref
        # histo['xrange'] = xrange
        # histo['yrange'] = yrange
        # histo['bin_h_edges'] = xx
        # histo['bin_v_edges'] = yy
        # histo['bin_h_left'] = np.delete(xx,-1)
        # histo['bin_v_left'] = np.delete(yy,-1)
        # histo['bin_h_right'] = np.delete(xx,0)
        # histo['bin_v_right'] = np.delete(yy,0)
        # histo['bin_h_center'] = 0.5*(histo['bin_h_left']+histo['bin_h_right'])
        # histo['bin_v_center'] = 0.5*(histo['bin_v_left']+histo['bin_v_right'])
        # histo['histogram'] = hh
        # histo['histogram_h'] = hh.sum(axis=0)
        # histo['histogram_v'] = hh.sum(axis=1)
        # histo['intensity'] = beam.intensity(nolost=1)
        # histo['nrays'] = beam.nrays(nolost=0)
        # histo['good_rays'] = beam.nrays(nolost=1)
        
        histo = beam.histo2(col_h=colh, col_v=colv, nbins_h=nbinsh, nbins_v=nbinsv, nolost=1, ref=colref, xrange=xrange, yrange=yrange);
        append_dataset_hdf5(filename, data=histo, z=z_points[i], nz=nz, tag=i+1, t0=t0, ndigits=len(str(nz)))
    read_caustic(filename, write_attributes=True)



def clean_shadow_files(path, print_files=1, delete_files=1,
                       prefixlist=['mirr', 'rmir','optax', 'effic', 
                                   'screen', 'star', 'angle']):
    
    removed_files = []
    
    for name in prefixlist:
        
        files = os.path.join(path, name+'*')
        
        fnames = glob.glob(files)
        
        for fnm in fnames:         
            
            removed_files.append(os.path.basename(fnm))
            if(print_files):
                print(removed_files[-1])
              
            if(delete_files):
                try:
                    os.remove(fnm)
                except:
                    pass
    
    
    return removed_files
    














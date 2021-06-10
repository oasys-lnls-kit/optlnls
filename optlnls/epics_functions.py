#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 15:29:36 2020

@author: sergio.lordano
"""

import numpy as np
from matplotlib import pyplot as plt
import epics
import h5py
import time
from optlnls.math import poly_any_degree
from optlnls.image import bin_matrix
from scipy.ndimage import rotate



            
def put_DCM(value, pv_DCM, pv_DCM_move, sleep_time=1.0):
    
    epics.caput(pv_DCM, value, wait=True) 

    if(sleep_time > 0 ): time.sleep(1.0)
    epics.caput(pv_DCM_move, 0, wait=True)
    if(sleep_time > 0 ): time.sleep(1.0)
    epics.caput(pv_DCM_move, 1, wait=True)
    if(sleep_time > 0 ): time.sleep(1.0)
    epics.caput(pv_DCM_move, 0, wait=True)    

def put_undulator(value, pv_und, pv_und_start, sleep_time=0.5):
    
    epics.caput(pv_und, value)
    if(sleep_time > 0 ): time.sleep(0.5)
    epics.caput(pv_und_start, 3)



def move_DCM(pv_to_move='MNC:A:DCM01:GonRxR', pv_RBV='MNC:A:DCM01:GonRxEnergy_RBV', pv_value=6.0, pv_delta=0.5e-3):
    """
    pv_to_move (type=str): name of the pv to be actuated (setpoint)
    pv_RBV (type=str): name of the read-back-value pv that will be compared to the setpoint to check movement
    pv_value (type=float): desired setpoint of the pv *pv_to_move* (e.g keV for energy)
    pv_delta (type=float): interval to compare setpoint to read-back-value and define if movement is finished (e.g keV for energy)
    """
    
    pv_trajMove = 'MNC:A:DCM01:TrajMove'

    # put value to PV and operate trajMove
    put_DCM(pv_value, pv_to_move, pv_trajMove)

    print('Moving DCM...')
    t0 = time.time()
    moving = 1
    while moving:
        time.sleep(0.2)
        DCM_diff = np.abs(epics.caget(pv_RBV) - pv_value)
        moving = DCM_diff >= pv_delta

        # in case DCM does not move, try again in 10 seconds
        if(time.time() - t0 > 10):
            put_DCM(pv_value, pv_to_move, pv_trajMove)
            t0 = time.time()

    print('finished moving!')

    return 1

    
def move_UND_and_DCM_at_MNC(energy_value, harmonic):
    
    #pv_und_phase_read = 'SI-09SA:ID-APU22:Phase-Mon'
    pv_und_phase_write = 'SI-09SA:ID-APU22:Phase-SP'
    pv_und_phase_moving = 'SI-09SA:ID-APU22:Moving-Mon'
    pv_und_phase_start = 'SI-09SA:ID-APU22:DevCtrl-Cmd'

    pv_DCM_energy_read = 'MNC:A:DCM01:GonRxEnergy_RBV'
    pv_DCM_energy_write = 'MNC:A:DCM01:GonRxR'
    pv_DCM_trajMove = 'MNC:A:DCM01:TrajMove'

    phase = get_phase_from_energy(energy_value, harmonic, verbose=0)
    
    DCM_value = energy_value
    UND_value = phase


    if((DCM_value >= 5.5) & (DCM_value <= 25.0)):

        put_DCM(DCM_value, pv_DCM_energy_write, pv_DCM_trajMove)

        put_undulator(UND_value, pv_und_phase_write, pv_und_phase_start)

        print('Moving DCM (Energy)...')

        delta_DCM = 0.5e-3

        t0 = time.time()
        moving = 1
        while moving:
            time.sleep(0.1)
            DCM_diff = np.abs(epics.caget(pv_DCM_energy_read) - DCM_value)
            DCM_moving = DCM_diff >= delta_DCM
            UND_moving = epics.caget(pv_und_phase_moving)
            moving = UND_moving or DCM_moving

            if(time.time() - t0 > 10):
                put_DCM(DCM_value, pv_DCM_energy_write, pv_DCM_trajMove)
                t0 = time.time()

        print('finished moving!')
        return 1

    else:
        print('NOT MOVING! COMMANDED DCM ENERGY IS OUT OF LIMITS!')
        return 0


def put_foil_into_beam(foil_pv, foil_pv_in, sleep_time):
    
    epics.caput(foil_pv, 0)
    t0 = time.time()
    while epics.caget(foil_pv_in):
        time.sleep(0.1)
        if time.time()-t0 > 5:
            print('Jammed')
            epics.caput(foil_pv, 1)
            if(sleep_time > 0 ): time.sleep(1)
            epics.caput(foil_pv, 0)

def put_foil_out_of_beam(foil_pv, foil_pv_out, sleep_time):
    
    epics.caput(foil_pv, 1)
    t0 = time.time()
    while epics.caget(foil_pv_out):
        time.sleep(0.1)
        if time.time()-t0 > 5:
            print('Jammed')
            epics.caput(foil_pv, 0)
            if(sleep_time > 0 ): time.sleep(1)
            epics.caput(foil_pv, 1)

def get_current_value():
    
    return epics.caget('SI-Glob:AP-CurrInfo:Current-Mon')
    
    
def get_image_max_value(image_pv, shape=[1024, 1280, 1], 
                        binning=[1,1], plot_image=0):
    
    import epics
    
    ### read PVs
    img = np.array(epics.caget(image_pv))            # get image from pv
    img = img.reshape(shape[0], shape[1], shape[2])  # reshape array
    img = img[:,:,0]                                 # get the RGB channel
    
    ### bin image to avoid saturating noise
    if(binning != [1,1]):
        img = bin_matrix(img, binning[0], binning[1])
        
    ### get maximum value
    img_max = np.max(img)

    if(plot_image):
        fig, ax = plt.subplots(figsize=(12,8))
        im = ax.imshow(img, origin='lower')
        fig.colorbar(im, ax=ax)
        plt.show()
    
    return img_max

def adjust_exposure_time(image_pv, exp_time_pv, saturation=256, 
                         threshold=[0.75, 0.85], shape=[1024, 1280, 1], 
                         binning=[1,1], exp_time_max=1, debug=0):
    
    #import epics
    
    img_max = get_image_max_value(image_pv, shape, binning)
    
    ### check if exposure time is ok
    exp_time_is_bad = ((img_max < threshold[0]*saturation) or  
                       (img_max > threshold[1]*saturation))

    exp_time = float(epics.caget(exp_time_pv))    
    ### optimize exposure time
    if(exp_time_is_bad):
        
        print("   exposure time is bad. trying to optimize... ")
        
 
        if(debug): print("   exposure time is ", round(exp_time, 3))
    
        trial_number = 0
        
        while exp_time_is_bad:
            
            trial_number += 1
            if(debug): print("   trial number is ", trial_number)
            
            # avoid infinite loops
            if trial_number > 20:
                if(debug): print('   failed to optimize exposure time')
                break
        
            img_max = get_image_max_value(image_pv, shape, binning)  
            if(debug): print('   image maximum is', round(img_max, 1))
        
            # if is saturated, divide time by 2
            if(img_max >= threshold[1]*saturation):
                exp_time = round(exp_time/2, 4)
                epics.caput(exp_time_pv, exp_time, wait=True)
                if(debug): print("   image saturated. changing exp time to ", exp_time)
                time.sleep(2*exp_time)
                
            # if exposure time is too low, estimate the ideal value
            elif(img_max <= threshold[0]*saturation):
                if(img_max >= saturation*0.05):
                    #increase_factor = np.mean(threshold)*saturation / img_max
                    increase_factor = 1.2
                else:
                    increase_factor = 2.0
                
                exp_time = round(exp_time * increase_factor, 4)
                
                ## only increase if it is less than maximum allowed time
                if(exp_time <= exp_time_max):
                
                    epics.caput(exp_time_pv, exp_time, wait=True)
                    if(debug): print("   image underexposed. changing exp time to ", exp_time)
                    time.sleep(2*exp_time)
                    
                else:
                    print('   calculated exposition time larger than allowed. Optimization Failed')
                    break
                
                
            # else, exposure time is ok
            else:
                exp_time_is_bad = False
                print("   exp time set to {0:.3f} . max value = {1:.1f}".format(exp_time, img_max) + " optimization was successful. \n")
        
        return exp_time
    
    else:
        print('   no need to optimize exposure time')
        return exp_time


def get_beam_position_arinax(px2um=0.31, pv='MNC:B:BZOOM:image1:ArrayData', plot=0):
    
    #px2um0 = 1.86 # micron/px zoom 1
    px2um0 = 0.31 # micron/px # zoom 5
    #px2um0 = 0.233 # micron/px # zoom 6
    theta = 45 # rotation angle
    shape = [1024, 1280, 3] # shape from detector
    RGB_channel = 1 # choose RGM channel
    background_value = 11 # points below this will be set to zero
    
    
    cmap = 'viridis' 
    origin = 'lower' 
    # origin = 'upper'

    # read PV from epics
    pv = epics.PV(pv)
    # data = np.array(pv.get(),np.int8)
    data = np.array(pv.get())
    #print('max count = ', np.max(data))
    max_count = np.max(data)
    # calculate pixel size for rotated image
    px2um = px2um0 # * np.cos(theta * np.pi / 180)


    # reshape array to 3-matrix (RGB)
    data2 = data.reshape(shape[0], shape[1], shape[2])

    # choose RGB channel and get x,y arrays in microns
    img = data2[:,:,RGB_channel]

    # remove background
    img[img <= background_value] = 0 
    
#     img = median_filter(img, 3)
#     img = gaussian_filter(img, 2)
    
    # rotate image
    img_rot = rotate(img, theta, cval = 0)

    shape_rot = img_rot.shape
    x = np.linspace(1, shape_rot[1], shape_rot[1]) * px2um 
    y = np.linspace(1, shape_rot[0], shape_rot[0]) * px2um

    # calculate mean values to bring beam to (0,0)
    Ix = np.sum(img_rot, axis=0)
    Iy = np.sum(img_rot, axis=1)
    x_mean = round(np.average(x, weights=Ix), 1)
    y_mean = round(np.average(y, weights=Iy), 1)

    if(plot):
    
        plt.figure(figsize=(12,8))
        plt.imshow(img_rot, origin=origin, cmap=cmap,
                  extent=[x[0], x[-1], y[0], y[-1]])
        plt.hlines(y=y_mean, xmin=x[0], xmax=x[-1], linestyle='--', color='w', alpha=0.3)
        plt.vlines(x=x_mean, ymin=y[0], ymax=y[-1], linestyle='--', color='w', alpha=0.3)
        plt.minorticks_on()
        plt.tick_params(which='both', axis='both', right=True, top=True)
        plt.xlabel('X [$\mu$m]')
        plt.ylabel('Y [$\mu$m]')
    
    return x_mean, y_mean, max_count

def get_average_position(n=5, sleep_time=0.3, px2um=0.31, pv='MNC:B:BZOOM:image1:ArrayData', plot=0):
    
    data = np.zeros((n,3))
    
    for i in range(n):
        
        x, y, counts = get_beam_position_arinax(px2um, pv, plot)
        data[i] = [x, y, counts]
        time.sleep(sleep_time)
        
    data_averaged = np.average(data, axis=0)
    data_rms = np.std(data, axis=0)
    
    return data_averaged, data_rms
    

## OK
def find_harmonics_given_energy(energy_points, min_energy=1.870, max_energy=3.600, 
                                print_points=0, initial_harmonic=3, only_max_harmonic=0):

    scan_points = []

    for i in range(len(energy_points)):

        energy = energy_points[i]

        if(only_max_harmonic):
            
            k = initial_harmonic
            while(True):
                
                if((energy / k >= min_energy) & (energy / k <= max_energy)):
                    max_harmonic = k
                    
                elif(energy / k < min_energy):
                    break
                                        
                k += 2

            scan_points.append([energy, max_harmonic])
            
            if(print_points):
                print(scan_points[-1])
            
        else:
        
            k = initial_harmonic
            while(True):
                
                if((energy / k >= min_energy) & (energy / k <= max_energy)):
                    scan_points.append([energy, k])
                    if(print_points):
                        print(scan_points[-1])
                
                elif(energy / k < min_energy):
                    break
                                        
                k += 2
                
            
    return scan_points



def acquire_image(image_pv, shape=(1024,1280), binning=(1,1), plot_image=0):

    img = np.array(epics.caget(image_pv))
    img = img.reshape(shape)
    
    if(binning != (1,1)):
        img = bin_matrix(img, binning[0], binning[1])
    
    if(plot_image):
        fig, ax = plt.subplots(figsize=(6,4))
        im = ax.imshow(img, origin='lower', cmap='jet')
        fig.colorbar(im, ax=ax)
        plt.show()
        #plt.savefig(filename, dpi=400)
        #plt.close('all')
        
    return img



    
def get_manaca_poly_coefficients():
    
    poly_coeffs = np.array([1.87693588e+00,  
                            1.16168505e-02,  
                            5.63397037e-03,  
                            9.07383600e-03,
                           -2.91843183e-03,  
                            5.54229460e-04, 
                           -5.71610126e-05,  
                            2.97752026e-06, 
                           -6.69368265e-08])
    
    return poly_coeffs

def search_phase(x, *args):
    
    energy = args[0]
    coeffs = args[1]
    
    return np.abs(poly_any_degree(x, coeffs) - energy)

def get_phase_from_energy_h1(energy, coefficients, bounds=[0,10]):
    
    from scipy.optimize import minimize_scalar
    
    args = (energy, coefficients)
    res = minimize_scalar(search_phase, args=args, bounds=[0,10], method='bounded')
    return res.x

def get_phase_from_energy(energy_value=12.000, harmonic_number=5, poly_coeffs=[0]*6, verbose=1):

    ### find possible harmonics
    harmonics = find_harmonics_given_energy(energy_points = [energy_value],
                               only_max_harmonic=False)

    harmonics = np.array(harmonics)
    harmonics = harmonics[:,1].astype(int)


    if(harmonic_number in harmonics):

        energy1 = energy_value / harmonic_number
        
        phase = get_phase_from_energy_h1(energy1, poly_coeffs, bounds=[0,10])
        phase = round(phase,3)
        
        if(verbose):
            print('phase value is:', phase)
            print('energy:', energy_value, 'keV')
            print('h =', harmonic_number)
            print('fundamental energy =', round(energy1, 3))

    else:
        if(verbose):
            print('harmonic h =', harmonic_number, 'is not valid for this energy.')
            print('please choose a valid harmonic number')
        phase = np.nan
            
    if(verbose):
        print('\n')    
        print('possible harmonics for this energy are:')
        for h in harmonics:
            print('h =', h)
        
    return phase

def define_ROI(pv_prefix='MNC:A:BASLER02', ROI_shape=(400,400), ROI_start=(0,0), ROI_binning=(1,1)):
    
    # pv_image = pv_prefix + ':image1:ArrayData'
    
    # enable ROI
    epics.caput(pv_prefix +':ROI1:EnableCallbacks', 1, wait=True)
    epics.caput(pv_prefix + ':image1:NDArrayPort', 'ROI1', wait=True)
    
    # set ROI
    epics.caput(pv_prefix +':ROI1:MinX', ROI_start[1], wait=True)    
    epics.caput(pv_prefix +':ROI1:MinY', ROI_start[0], wait=True)
    
    epics.caput(pv_prefix +':ROI1:SizeX', ROI_shape[1], wait=True)    
    epics.caput(pv_prefix +':ROI1:SizeY', ROI_shape[0], wait=True)
    
    epics.caput(pv_prefix +':ROI1:BinX', ROI_binning[1], wait=True)    
    epics.caput(pv_prefix +':ROI1:BinY', ROI_binning[0], wait=True)
    
    time.sleep(1)

def initialize_hdf5(h5_filename):
    
    with h5py.File(h5_filename, 'w') as f:
        
        f.attrs['begin time'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        f.create_group('images')
        f.create_group('vertical_cuts')
    
def end_hdf5(h5_filename):
    
    with h5py.File(h5_filename, 'a') as f:
        f.attrs['end time'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        
    
def create_group(h5_filename, group_name, group_attributes):
    
    with h5py.File(h5_filename, 'a') as f:
        group_images = f['images'].create_group(group_name)
        group_cuts = f['vertical_cuts'].create_group(group_name)
        
        for i in range(len(group_attributes)):
            group_images.attrs[group_attributes[i][0]] = group_attributes[i][1]
            group_cuts.attrs[group_attributes[i][0]] = group_attributes[i][1]
    
def append_image_to_hdf5(h5_filename, group_name, dataset_name, image, attributes):
    
    with h5py.File(h5_filename, 'a') as f:
        
        group = f['images/'+group_name]
        dset = group.create_dataset(dataset_name, data=image, compression="gzip")
        
        for i in range(len(attributes)):
            dset.attrs[attributes[i][0]] = attributes[i][1] 

def append_cut_to_hdf5(h5_filename, group_name, dataset_name, cut, attributes):
    
    with h5py.File(h5_filename, 'a') as f:
                
        group = f['vertical_cuts/'+group_name]
        dset = group.create_dataset(dataset_name, data=cut, compression="gzip")
        
        for i in range(len(attributes)):
            dset.attrs[attributes[i][0]] = attributes[i][1] 
            

    

        
        
        
        

if __name__ == '__main__':

    filename = ''        
            
    beam = np.genfromtxt(filename)
    img = beam[1:,1:]
    img_binned = bin_matrix(img, 8, 8)
        
    # adjust_exposure_time_sim(1, 2)













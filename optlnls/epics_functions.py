#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 15:29:36 2020

@author: sergio.lordano
"""

import numpy as np
from matplotlib import pyplot as plt
import epics
import time
from optlnls.math import bin_matrix

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

def bin_matrix(matrix, binning_y, binning_x):

    yn, xn = matrix.shape

    if ((xn % binning_x != 0) or (yn % binning_y != 0)):
        print('array of shape ({0} x {1}) cannot be binned by factor ({2},{3})'.format(yn, xn, binning_y, binning_x))
        return matrix
    
    else:
        #print('binning matrix of shape({0},{1}) by factors ({2},{3})'.format(yn, xn, binning_y, binning_x))
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


def get_centroid(img, x=0, y=0):
    
    shape = img.shape
    
    if(x==0):
        x = np.arange(0, shape[1])
    if(y==0):
        y = np.arange(0, shape[0])
    
    Ix = np.sum(img, axis=0)
    Iy = np.sum(img, axis=1)

    x_mean = 0
    y_mean = 0

    if(np.sum(Ix) != 0):
        x_mean = x[np.argmax(Ix)]

    if(np.sum(Iy) != 0):
        y_mean = y[np.argmax(Iy)]  
        
    return (y_mean, x_mean)

def get_vertical_cut(img, x=0, nx=1, binning=(1,1), plot_cut=0):
    
    if(binning != (1,1)):
        img = bin_matrix(img, binning[0], binning[1])
    
    if(nx == 1):
        xmin=int(x)

    elif(nx > 1):
        xmin=int(x - nx/2)

    xmax=int(xmin + nx)
        
    cut = np.sum(img[:, xmin:xmax], axis=1)/nx
    
    if(plot_cut):
        plt.figure()
        plt.plot(cut)
        plt.show()
    
    return cut
    
    
def poly_any_degree(x, coefficients):
       
    y = 0
    for i in range(len(coefficients)):
        y += coefficients[i] * x**i
    return y

def search_phase(x, *args):
    
    energy = args[0]
    coeffs = args[1]
    
    return np.abs(poly_any_degree(x, coeffs) - energy)

def get_phase_from_energy_h1(energy, coefficients, bounds=[0,10]):
    
    from scipy.optimize import minimize_scalar
    
    args = (energy, coefficients)
    res = minimize_scalar(search_phase, args=args, bounds=[0,10], method='bounded')
    return res.x

def define_ROI(pv_prefix='MNC:A:BASLER02', ROI_shape=(400,400), ROI_start=(0,0), ROI_binning=(1,1)):
    
    pv_image = pv_prefix + ':image1:ArrayData'
    
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
        
    adjust_exposure_time_sim(1, 2)













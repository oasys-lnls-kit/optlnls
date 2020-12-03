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
    
def put_undulator(value, pv_und, pv_und_start, sleep_time=0.5):
    
    epics.caput(pv_und, value)
    if(sleep_time > 0 ): time.sleep(0.5)
    epics.caput(pv_und_start, 3)

def get_current_value(current_pv='SI-Glob:AP-CurrInfo:Current-Mon'):
    
    return epics.caget(current_pv)




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
                         binning=[1,1], debug=0):
    
    import epics
    
    img_max = get_image_max_value(image_pv, shape, binning)
    
    ### check if exposure time is ok
    exp_time_is_bad = ((img_max < threshold[0]*saturation) or  
                       (img_max > threshold[1]*saturation))
    
    ### optimize exposure time
    if(exp_time_is_bad):
        
        if(debug): print("\n exposure time is bad. trying to optimize... \n")
        
        exp_time = float(epics.caget(exp_time_pv)) 
        if(debug): print("exposure time is ", round(exp_time, 5))
    
        trial_number = 0
        
        while exp_time_is_bad:
            
            trial_number += 1
            if(debug): print("trial number is ", trial_number)
            
            # avoid infinite loops
            if trial_number > 20:
                if(debug): print('failed to optimize exposure time')
                break
        
            img_max = get_image_max_value(image_pv, shape, binning)    
        
            # if is saturated, divide time by 2
            if(img_max >= threshold[1]*saturation):
                exp_time = round(exp_time/2, 5)
                epics.caput(exp_time_pv, exp_time, wait=True)
                if(debug): print("image saturated. changing exp time to ", exp_time)
                
            # if exposure time is too low, estimate the ideal value
            elif(img_max <= threshold[0]*saturation):
                increase_factor = np.mean(threshold)*saturation / img_max
                exp_time = exp_time * increase_factor
                epics.caput(exp_time_pv, exp_time, wait=True)
                if(debug): print("image underexposed. changing exp time to ", exp_time)
                
            # else, exposure time is ok
            else:
                exp_time_is_bad = False
                if(debug): print("exp time set to {0:.5f} .".format(exp_time) + " optimization was successful. \n")
        
        return 
        
def adjust_exposure_time_sim(image_pv, exp_time_pv, saturation=256, 
                             threshold=[0.75, 0.85], shape=[1024, 1280, 1], 
                             binning=[1,1]):
    
    #import epics
    
    img_max = 256
    
    ### check if exposure time is ok
    exp_time_is_bad = ((img_max < threshold[0]*saturation) or  
                       (img_max > threshold[1]*saturation))
    
    ### optimize exposure time
    if(exp_time_is_bad):
        print("exposure time is bad")
        
        exp_time = 1.2
        print("exposure time is ", exp_time )
    
        trial_number = 0
        while exp_time_is_bad:
            
            trial_number += 1
            print('optimization trial number ', trial_number)
            
            # avoid infinite loops
            if trial_number > 20:
                print('failed to optimize exposure time')
                break
        
            #img_max =  
            print('max value is ', img_max)
        
            # if is saturated, divide time by 2
            if(img_max >= threshold[1]*saturation):
                #exp_time_setpoint = exp_time/2
                #epics.caput(exp_time_pv, exp_time_setpoint, wait=True)
                img_max /= 2 
                
            # if exposure time is too low, estimate the ideal value
            elif(img_max <= threshold[0]*saturation):
                increase_factor = np.mean(threshold)*saturation / img_max
                #exp_time_setpoint = exp_time * increase_factor
                #epics.caput(exp_time_pv, exp_time_setpoint, wait=True)
                img_max *= increase_factor
                
            # else, exposure time is ok
            else:
                exp_time_is_bad = False
        
        
        
        
        

if __name__ == '__main__':

    filename = ''        
            
    beam = np.genfromtxt(filename)
    img = beam[1:,1:]
    img_binned = bin_matrix(img, 8, 8)
        
    adjust_exposure_time_sim(1, 2)













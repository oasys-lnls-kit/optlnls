#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 12:12:08 2021

@author: lordano
"""

import numpy as np
from matplotlib import pyplot as plt

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

def crop_matrix(matrix, new_idx_y, new_idx_x, plot_matrix=0):
    
        matrix_cropped = matrix[int(new_idx_y[0]) : int(new_idx_y[1]),
                                int(new_idx_x[0]) : int(new_idx_x[1])]   

        if(plot_matrix):
            fig, ax = plt.subplots(ncols=2)
            ax[0].imshow(np.log10(matrix), origin='lower')
            ax[1].imshow(np.log10(matrix_cropped), origin='lower')
            plt.show()

        return matrix_cropped
    
    
    
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
    
    
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 11:03:10 2020

@author: sergio.lordano
"""

import numpy as np
from scipy import ndimage


def read_shadow_beam(beam, x_column_index=1, y_column_index=3, nbins_x=100, nbins_y=100, nolost = 1, ref = 23, zeroPadding=0, gaussian_filter=0):
    """
    

    Parameters
    ----------
    beam : ShadowBeam()
        General Shadow beam object.
    x_column_index : int
        Shadow column number for x axis. The default is 1.
    y_column_index : int
        Shadow column number for y axis. The default is 3.
    nbins_x : int
        Number of bins for x axis. The default is 100.
    nbins_y : int
        Number of bins for y axis. The default is 100.
    nolost : int
        1 to use only good rays; 0 to use good and lost rays. The default is 1.
    ref : TYPE, optional
        Shadow column used as weights. The default is 23 (intensity). 
    zeroPadding : float
        Range factor for inserting zeros in the beam matrix. The default is 0.
    gaussian_filter : float
        A float larger than 0 to apply gaussian filter. The default is 0.

    Returns
    -------
    XY : float array
        returns a 2D numpy array where first row is x coordinates, first column
        is y coordinates, [0,0] is not used, and [1:1:] is the 2D histogram.

    """

    
    histo2D = beam.histo2(col_h = x_column_index, col_v = y_column_index, nbins_h = nbins_x, nbins_v = nbins_y, nolost = nolost, ref = ref)
    
    x_axis = histo2D['bin_h_center']
    y_axis = histo2D['bin_v_center']
    xy = histo2D['histogram']
    
    
    if(zeroPadding==0):
        XY = np.zeros((nbins_y+1,nbins_x+1))
        XY[1:,0] = y_axis
        XY[0,1:] = x_axis
        XY[1:,1:] = np.array(xy).transpose()
        
        if(gaussian_filter != 0):
            XY[1:,1:] = ndimage.gaussian_filter(np.array(xy).transpose(), gaussian_filter)
        
    else:
        x_step = x_axis[1]-x_axis[0]
        y_step = y_axis[1]-y_axis[0]
        fct = zeroPadding
        XY = np.zeros((nbins_y+15, nbins_x+15))
        XY[8:nbins_y+8,0] = y_axis
        XY[0,8:nbins_x+8] = x_axis
        XY[8:nbins_y+8,8:nbins_x+8] = np.array(xy).transpose()
        
        XY[1,0] = np.min(y_axis) - (np.max(y_axis) - np.min(y_axis))*fct
        XY[2:-1,0] = np.linspace(y_axis[0] - 6*y_step, y_axis[-1] + 6*y_step, nbins_y+12)
        XY[-1,0] = np.max(y_axis) + (np.max(y_axis) - np.min(y_axis))*fct
        
        XY[0,1] = np.min(x_axis) - (np.max(x_axis) - np.min(x_axis))*fct
        XY[0,2:-1] = np.linspace(x_axis[0] - 6*x_step, x_axis[-1] + 6*x_step, nbins_x+12)
        XY[0,-1] = np.max(x_axis) + (np.max(x_axis) - np.min(x_axis))*fct
        
        if(gaussian_filter != 0):
            XY[3:nbins_y+3,3:nbins_x+3] = ndimage.gaussian_filter(np.array(xy).transpose(), gaussian_filter)
    
    
    return XY




def read_spectra_xyz(filename):
    """
    

    Parameters
    ----------
    filename : str
        path to spectra file with xyz columns.

    Returns
    -------
    beam : float array 
          Returns a 2D numpy array where first row is x coordinates, first column
          is y coordinates, [0,0] is not used, and [1:1:] is the z axis.

    """
        
    data = np.genfromtxt(filename, skip_header=2)

    X = data[:,0]
    Y = data[:,1]
    I = data[:,2]

    for nx in range(len(X)):
        if(X[nx+1] == X[0]):
            nx += 1
            break

    ny = int(len(Y)/nx)
    print(nx, ny)

    I_mtx = I.reshape((ny,nx))

    beam = np.zeros((ny+1, nx+1))
    beam[1:,0] = Y[0::nx]
    beam[0,1:] = X[:nx]
    beam[1:,1:] = I_mtx

    return beam


def read_srw_wfr(wfr, pol_to_extract=6, int_to_extract=0, unwrap_phase=0):
    """
    

    Parameters
    ----------
    wfr : SRWLWfr()
        SRW wavefront.
    pol_to_extract : int, optional
        Polarization component to extract. The default is 6.
    int_to_extract : int, optional
        Intensity type or phase component to extract. The default is 0.

    Returns
    -------
    mtx : float array
        Returns a 2D numpy array where first row is x coordinates, first column
        is y coordinates, [0,0] is not used, and [1:1:] is the z axis.

    """
    

    from array import array
    import srwlpy as srwl 
    
    if int_to_extract == 4:
        arI = array('d', [0]*wfr.mesh.nx*wfr.mesh.ny) #"flat" 2D array to take intensity data
    else:
        arI = array('f', [0]*wfr.mesh.nx*wfr.mesh.ny) #"flat" 2D array to take intensity data

    srwl.CalcIntFromElecField(arI, wfr, pol_to_extract, int_to_extract, 3, wfr.mesh.eStart, 0, 0)
    
    int_mtx = np.array(arI)
    int_mtx = int_mtx.reshape((wfr.mesh.ny, wfr.mesh.nx))
    
    if(unwrap_phase):
        int_mtx = np.unwrap(int_mtx, axis=0, discont=np.pi)
        int_mtx = np.unwrap(int_mtx, axis=1, discont=np.pi)
    
    mtx = np.zeros((wfr.mesh.ny+1, wfr.mesh.nx+1), dtype=np.float)
    mtx[0,1:] = np.linspace(wfr.mesh.xStart, wfr.mesh.xFin, wfr.mesh.nx)*1e3
    mtx[1:,0] = np.linspace(wfr.mesh.yStart, wfr.mesh.yFin, wfr.mesh.ny)*1e3
    mtx[1:,1:] = int_mtx
    
    return mtx


def read_srw_int(filename):
    """
    

    Parameters
    ----------
    filename : str
        Path to SRW intensity file.

    Returns
    -------
    mtx : float array
        Returns a 2D numpy array where first row is x coordinates, first column
        is y coordinates, [0,0] is not used, and [1:1:] is the z axis.

    """
    
    
    with open(filename, 'r') as infile:
        data = infile.readlines()
    infile.close()
    
    ei = float(data[1].split('#')[1])
    ef = float(data[2].split('#')[1])
    en = int(data[3].split('#')[1])
    xi = float(data[4].split('#')[1])
    xf = float(data[5].split('#')[1])
    xn = int(data[6].split('#')[1])
    yi = float(data[7].split('#')[1])
    yf = float(data[8].split('#')[1])
    yn = int(data[9].split('#')[1])
    
    nheaders = 11
    if not(data[10][0]=='#'): nheaders = 10
    
    if(0):       
#       #loop method      
        intensity = np.zeros((en, yn, xn))       
        count = 0     
        for i in range(yn):
            for j in range(xn):
                for k in range(en):
                    intensity[k, i, j] = data[count + nheaders]
                    count += 1
    if(1):            
#       #Reshape method
        intensity = np.array(data[nheaders:], dtype='float').reshape((en, yn, xn))
    
    e_pts = np.linspace(ei, ef, en)    
    mtx = np.zeros((en, yn+1, xn+1))
    for i in range(en):
        mtx[i][0,0] = e_pts[i]
        mtx[i][0,1:] = np.linspace(xi, xf, xn)*1e3
        mtx[i][1:,0] = np.linspace(yi, yf, yn)*1e3
        mtx[i][1:,1:] = intensity[i]
    
    return mtx




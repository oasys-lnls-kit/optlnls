#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 15:29:25 2020

@author: sergio.lordano
"""
import numpy
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from optlnls.math import derivate


def crop_height_error_matrix(matrix, L, W, height_offset=False):
    """
    This function receives as input a 2D mirror height error matrix and crop it\
    to new length and width within the matrix range. Input and Output format is a 2D matrix in which\
    the first row is the longitudinal coordinates, the first column is sagittal coordinates,\
    elements [1:,1:] are the corresponding height errors and element [0,0] is unused.\
    Length, width and matrix must be in same units.\n
    :matrix: 2D matrix in the above mentioned format
    :L: total mirror length in which the matrix must be cropped 
    :W: total mirror width in which the matrix must be cropped
    :height_offset: if True, a height offset will be added in all points so that the median value of central line is zero.
    """
    import numpy as np    
    idx_min_L = np.abs(matrix[0,:]-(-L/2.0)).argmin() # search for the coordinate values which are the closest from desired L and W.
    idx_max_L = np.abs(matrix[0,:]-(L/2.0)).argmin()
    idx_min_W = np.abs(matrix[:,0]-(-W/2.0)).argmin()
    idx_max_W = np.abs(matrix[:,0]-(W/2.0)).argmin() 
#    print(idx_min_L, idx_max_L, idx_min_W, idx_max_W)   
    
    height2D_part = matrix[idx_min_W:idx_max_W+1,idx_min_L:idx_max_L+1]
    height2D_fmt = np.zeros((len(height2D_part[:,0])+1,len(height2D_part[0,:])+1))
    height2D_fmt[0,:][1:] = matrix[0,:][idx_min_L:idx_max_L+1]
    height2D_fmt[:,0][1:] = matrix[:,0][idx_min_W:idx_max_W+1]
    if(height_offset):
        vert_displacement = (max(height2D_part[int(len(height2D_part)/2.0)+1,:])+min(height2D_part[int(len(height2D_part)/2.0)+1,:]))/2.0
    else:
        vert_displacement = 0
    height2D_fmt[1:,1:] = height2D_part - vert_displacement

    return height2D_fmt

def write_shadow_height_error(matrix, file_name, unit_factor, delimiter=' '):
    """
    Creates a mirror height error file in shadow/dabam format from a 2D matrix in which\
    the first row is the longitudinal coordinates, the first column is sagittal coordinates,\
    elements [1:,1:] are the corresponding height errors and element [0,0] is unused.\n
    :matrix: 2D matrix in the above mentioned format
    :file_name: file name containing extension, e.g. 'mirror_error.dat'
    :unit_factor: unit factor from matrix units, that is, 1e+3 to write file in [mm] if matrix is in [m]
    :delimiter: space as standard for shadow format, or alternatively tab.
    """
    nl = len(matrix[:,0])-1
    nc = len(matrix[0,:])-1
    with open(file_name, 'w') as ofile:
        ofile.write('{0}'.format(nl)+ delimiter +'{0}\n'.format(nc))
        for line in range(nl+1):
            for column in range(nc+1):
                if not(line==0 and column==0):
                    ofile.write('{0:.3e}'.format(matrix[line, column]*unit_factor)+delimiter) 
            ofile.write('\n')
    ofile.closed
    
def from_shadow_to_matrix(file_path, unit_factor):
    """
    Imports a mirror height error file in shadow/dabam format and returns a 2D matrix (in meters) in which\
    the first row is the longitudinal coordinates, the first column is sagittal coordinates,\
    elements [1:,1:] are the corresponding height errors and element [0,0] is unused.\n
    :file_path: path to file including file name.
    :unit_factor: unit factor from the file units to meter, that is, 1e-3 if file is in [mm] for instance.
    
    """
    import numpy as np
    data = [] # array to store all elements in 1D list    
    with open(file_path, 'r') as datafile: # reads and append every element sequentially until readline() fails.
        while True:
            read_line = datafile.readline()
            if not read_line: break    
            for element in read_line.split():
                data.append(element)
    datafile.close()
    
    nl, nc = int(data[0]), int(data[1])
    matrix_data = np.zeros((nl+1, nc+1)) # allocate matrix
    matrix_data[0,:][1:] = np.array(data[2:nc+2], dtype='float')*unit_factor # [m] Longitudinal coordinates
    for line in range(nl): # associate elements in data array to the matrix lines 
        matrix_data[line+1,:] = np.array(data[1+(nc+1)*(line+1):1+(nc+1)*(line+2)], dtype='float')*unit_factor # [m]
    return matrix_data

def SRW_figure_error(file_name, unit_factor, angle_in, angle_out, orientation_x_or_y, crop=False, height_offset=False, L=5e3, W=5e3):
    """
    Returns an instance of srwlib.SRWLOptT() which simulates a mirror height error in SRW module.\
    To run this function, it is necessary to have srwlib able to import.\n
    :file_name: filename of file in shadow/dabam format
    :unit_factor: unit factor from the file units to meter, that is, 1e-3 if file is in [mm] for instance.
    :angle_in: incidence angle [rad] relative to mirror surface
    :angle_out: reflection angle [rad] relative to mirror surface
    :orientation_x_or_y: 'x' for horizontal or 'y' for vertical deflection
    :crop: (optional) if True, crops the matrix to new length L and width W and optionally offset data to make median value equal zero
    :height_offset: if True, a height offset will be added in all points so that the median value of central line is zero.
    :L: total mirror length in which the matrix must be cropped 
    :W: total mirror width in which the matrix must be cropped
    """
    from srwlib import srwl_opt_setup_surf_height_2d

    
    height2D = from_shadow_to_matrix(file_name, unit_factor)
    if(crop):
        height2D_cropped = crop_height_error_matrix(height2D, L, W, height_offset)
        print('Actual L x W: {0:.3f} m x {1:.3f} m'.format(height2D_cropped[0,-1]-height2D_cropped[0,1], height2D_cropped[-1,0]-height2D_cropped[1,0]))
        return srwl_opt_setup_surf_height_2d(height2D_cropped, orientation_x_or_y, angle_in, angle_out)
    else:
        return srwl_opt_setup_surf_height_2d(height2D, orientation_x_or_y, angle_in, angle_out)
    
    
def analyze_height_error(filelist, unit_factor, workingFolder=''):
    
    import os
    from optlnls.math import derivate, psd, linear_function
    from optlnls.plot import set_ticks_size
    from scipy.optimize import curve_fit
    
    
    # === CREATES SUBFOLDERS === #
    if(workingFolder != ''):
        os.chdir(workingFolder)
        
    if not (os.path.exists('2D')): os.mkdir('2D')
    if not (os.path.exists('meridional')): os.mkdir('meridional')
    if not (os.path.exists('sagittal')): os.mkdir('sagittal')
    if not (os.path.exists('PSD')): os.mkdir('PSD')
    if not (os.path.exists('statistics')): os.mkdir('statistics')         
    
    
    for slopefile in filelist:
        

        
        # *** Finds adequate dimensions and writes new Shadow input file *** #
        filename = slopefile.split('/')[-1]
        slope2d = from_shadow_to_matrix(slopefile, unit_factor)
        new_mtx = slope2d #mypkg.select_part_surface(slope2d, L, W)
        nW = len(new_mtx[1:,0])
        nL = len(new_mtx[0,1:])    
        realL = new_mtx[0,-1]-new_mtx[0,1]
        realW = new_mtx[-1,0]-new_mtx[1,0]    
        X = new_mtx[0, 1:]        
        
        if(0):
            write_shadow_height_error(new_mtx, filename.split('.')[0]+'_ctr.dat', unit_factor, delimiter=' ')        
        
        #==================================================#
        # MERIDIONAL
        #==================================================#    
        mer_cut = new_mtx[int(np.ceil((len(new_mtx)-1)/2)+1),1:]
        mer_cut_slope = derivate(X, mer_cut)
        PV_mer_cut = (np.max(mer_cut)-np.min(mer_cut))*1e9
        std_height = np.std(mer_cut)*1e9
        std_slope = np.std(mer_cut_slope)*1e6 
        
        PSD, frequencies = psd(X, mer_cut)
    
        mer_errors = []
        for i in [-2, -1, 0, +1, +2]:
            idx = int(np.ceil((nW-1)/2.0)+1 + ((nW-1)/5.0)*i)
            mer_errors.append(calc_errors(new_mtx[0,1:], new_mtx[idx,1:], new_mtx[idx,0]))
        mer_errors = np.array(mer_errors)   
        
        length_scan = []
        for i in [realL/8.0, realL/4.0, realL/2.0]:
            idx_i, idx_f = np.abs(X+i).argmin(), np.abs(X-i).argmin()
            length_scan.append(calc_errors(X[idx_i:idx_f+1], new_mtx[int(np.ceil((nW-1)/2.0)+1),1:][idx_i:idx_f+1], X[idx_f]-X[idx_i]))
        length_scan = np.array(length_scan)
        
        #==================================================#
        # SAGITTAL
        #==================================================#    
        sag_cut = new_mtx[1:,int(np.ceil((len(new_mtx[0,:])-1)/2)+1)]
        sag_cut_slope = derivate(new_mtx[1:,0], sag_cut)
        PV_sag_cut = (np.max(sag_cut)-np.min(sag_cut))*1e9
        std_height_sag = np.std(sag_cut)*1e9
        std_slope_sag = np.std(sag_cut_slope)*1e6
        
        sag_errors = []
        for i in [-2, -1, 0, +1, +2]:
            idx = int(np.ceil((nL-1)/2.0)+1 + ((nL-1)/5.0)*i)
            sag_errors.append(calc_errors(new_mtx[1:,0], new_mtx[1:,idx], new_mtx[0,idx]))
        sag_errors = np.array(sag_errors)
        
        

        # === WRITES STATISTICS === #
        if(1): 
            write_stats(filename, length_scan, mer_errors, sag_errors, new_mtx) 

        #==================================================#
        # PLOTS MERIDIONAL
        #==================================================#
        fig_size = 5.5
        fsize = 14
        
        plt.figure(figsize=(2.5*fig_size,fig_size)) 

        plt.subplot(121)
        plt.plot(new_mtx[0][1:]*1e3, mer_cut*1e9, '.-', markersize=2.0)
        plt.title(filename+'\nL = {0} mm ; W = {1} mm'.format(realL*1e3, realW*1e3), fontsize=fsize)   
        plt.xlabel('Mirror Length [mm]', fontsize=fsize)
        plt.ylabel('Height Error [nm]', fontsize=fsize)    
        plt.grid(True)
        plt.text(0.95,0.95,"RMS = {0:.2f} nm".format(std_height)+'\n'+'PV = {0:.2f} nm'.format(PV_mer_cut), verticalalignment="top", horizontalalignment="right", transform=plt.gca().transAxes)
        plt.ylim(plt.ylim()[0], plt.ylim()[1]*1.15)        
        set_ticks_size(fsize)    
        
        plt.subplot(122)
        plt.plot(new_mtx[0][1:]*1e3, mer_cut_slope*1e6)
        plt.title(filename+'\nL = {0} mm ; W = {1} mm'.format(realL*1e3, realW*1e3), fontsize=fsize)
        plt.xlabel('Mirror Length [mm]', fontsize=fsize)
        plt.ylabel(r'Slope Error [$\mu rad$]', fontsize=fsize)    
    #    plt.ylim(-2.0,2.0)
        plt.grid(True)
        plt.text(0.95,0.95,"RMS = {0:.3f} ".format(std_slope)+r'$\mu rad$', verticalalignment="top", horizontalalignment="right", transform=plt.gca().transAxes)
        set_ticks_size(fsize)        
        
        plt.savefig(os.path.join(os.getcwd(), 'meridional', filename.split('.')[0]+'_mer.png'), dpi=300)
        
        
        #==================================================#
        # PLOTS SAGITTAL
        #==================================================#
        plt.figure(figsize=(2.5*fig_size,fig_size)) 
        
        plt.subplot(121)
        plt.plot(new_mtx[1:,0]*1e3, sag_cut*1e9, '.-', markersize=2.0)
        plt.title(filename+'\nL = {0} mm ; W = {1} mm'.format(realL*1e3, realW*1e3), fontsize=fsize)   
        plt.xlabel('Mirror Width [mm]', fontsize=fsize)
        plt.ylabel('Height Error [nm]', fontsize=fsize)    
        plt.grid(True)
        plt.text(0.95,0.95,"RMS = {0:.2f} nm".format(std_height_sag)+'\n'+'PV = {0:.2f} nm'.format(PV_sag_cut), verticalalignment="top", horizontalalignment="right", transform=plt.gca().transAxes)
        plt.ylim(plt.ylim()[0], plt.ylim()[1]*1.15)
        set_ticks_size(fsize)
        
        
        plt.subplot(122)
        plt.plot(new_mtx[1:,0]*1e3, sag_cut_slope*1e6)
        plt.title(filename+'\nL = {0} mm ; W = {1} mm'.format(realL*1e3, realW*1e3), fontsize=fsize)
        plt.xlabel('Mirror Width [mm]', fontsize=fsize)
        plt.ylabel(r'Slope Error [$\mu rad$]', fontsize=fsize)    
    #    plt.ylim(-2.0,2.0)
        plt.grid(True)
        plt.text(0.95,0.95,"RMS = {0:.3f} ".format(std_slope_sag)+r'$\mu rad$', verticalalignment="top", horizontalalignment="right", transform=plt.gca().transAxes)
        set_ticks_size(fsize)
        
        plt.savefig(os.path.join(os.getcwd(), 'sagittal', filename.split('.')[0]+'_sag.png'), dpi=300)
        
        #==================================================#
        # PLOTS 2D SURFACE
        #==================================================#
        plt.figure(figsize=(12,3.0))
        #fig = plt.gcf()
        ax = plt.gca()
        #cbound = np.max([np.abs(np.floor(np.min(new_mtx[1:,1:]*1e9))),np.abs(np.ceil(np.max(new_mtx[1:,1:]*1e9)))])    
        plt.imshow(new_mtx[1:,1:]*1e9, vmin=np.floor(np.min(new_mtx[1:,1:]*1e9)), vmax=np.ceil(np.max(new_mtx[1:,1:]*1e9)), # plot Z in nanometers
                   extent=[-realL/2.0*1e3, realL/2.0*1e3, -realW/2.0*1e3, realW/2.0*1e3], aspect='auto',
                   interpolation='nearest', origin='lower')
        cb = plt.colorbar(orientation="horizontal", fraction = 0.3, pad=0.21, shrink=0.5)
        cb.set_label('nm')
        ax.set_position([0.125, 0.6, 0.8, 0.3])
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        plt.xlabel('Length [mm]', fontsize=fsize)
        plt.ylabel('Width [mm]', fontsize=fsize)
        set_ticks_size(fsize)
        plt.savefig(os.path.join(os.getcwd(), '2D', filename.split('.')[0]+'_2D.png'), dpi = 200)
        
        #==================================================#
        # PLOTS PSD 
        #==================================================#        
        popt, pcov = curve_fit(linear_function, np.log10(frequencies[int(len(frequencies)*0.05): int(len(frequencies)*0.9)]), np.log10(PSD[int(len(frequencies)*0.05): int(len(frequencies)*0.9)]), maxfev=10000)        
        x = frequencies[int(len(frequencies)*0.05): int(len(frequencies)*0.9)]
        y = linear_function(np.log10(x), popt[0], popt[1])
        plt.figure()
        plt.loglog(frequencies, PSD)
        plt.loglog(x, 10**y, '-k')
        plt.title(filename+'\nPower Spectrum Density', fontsize=fsize)   
        plt.xlabel('Spatial Frequency [$m^{-1}$]', fontsize=fsize)
        plt.ylabel('PSD [$m^{2}$]', fontsize=fsize)    
        plt.grid(True)
        plt.text(0.95,0.95, r'$y = \alpha\ + \beta*x$'+'\n'+r'$\alpha= $'+'{0:.3f}'.format(popt[1])+'\n'+r'$\beta= $'+'{0:.3f}'.format(popt[0]), verticalalignment="top", horizontalalignment="right", transform=plt.gca().transAxes)
        set_ticks_size(fsize)
        plt.savefig(os.path.join(os.getcwd(), 'PSD', filename.split('.')[0]+'_PSD.png'), dpi=200)

        plt.close('all')            
        # plt.show()

def calc_errors(axis, heights, coordinate_value=0):
    
    from optlnls.math import derivate
    
    PV_cut = (np.max(heights)-np.min(heights))
    std_cut = np.std(heights)
    mean_heights = (np.max(heights)+np.min(heights))/2.0
    cut_slope = derivate(axis, heights)
    std_slope = np.std(cut_slope)
    return [coordinate_value, PV_cut, std_cut, mean_heights, std_slope]

def write_stats(filename, length_scan, mer_errors, sag_errors, new_mtx):
    
    import os
    
    with open('statistics/'+filename.split('.')[0]+'_stats.txt', 'w') as stats:
        stats.write('Figure Error Analysis'+ '\n' + 'File: ' + os.path.join(os.getcwd(), filename) +'\n\n' + '# *** Meridional Center Line - Different Lengths *** #\n' + 'L [mm]\tHeight-PV[nm]\tHeight-rms[nm]\tHeight-mean[nm]\tSlope-rms[urad]\n')
        for i in range(len(length_scan)):
            stats.write('{0:.1f}\t{1:.2f}\t{2:.2f}\t{3:.2f}\t{4:.3f}\n'.format(length_scan[i,0]*1e3, length_scan[i,1]*1e9, length_scan[i,2]*1e9, length_scan[i,3]*1e9, length_scan[i,4]*1e6))
        stats.write('\n'+'# *** Meridional Cuts Statistics *** # \n' + 'W [mm]\tHeight-PV[nm]\tHeight-rms[nm]\tHeight-mean[nm]\tSlope-rms[urad]\n')
        for i in range(len(mer_errors)):
            stats.write('{0:.1f}\t{1:.2f}\t{2:.2f}\t{3:.2f}\t{4:.3f}\n'.format(mer_errors[i,0]*1e3, mer_errors[i,1]*1e9, mer_errors[i,2]*1e9, mer_errors[i,3]*1e9, mer_errors[i,4]*1e6))
        stats.write('\n'+'# *** Sagittal Cuts Statistics *** # \n' + 'L [mm]\tHeight-PV[nm]\tHeight-rms[nm]\tHeight-mean[nm]\tSlope-rms[urad]\n')
        for i in range(len(sag_errors)):
            stats.write('{0:.1f}\t{1:.2f}\t{2:.2f}\t{3:.2f}\t{4:.3f}\n'.format(sag_errors[i,0]*1e3, sag_errors[i,1]*1e9, sag_errors[i,2]*1e9, sag_errors[i,3]*1e9, sag_errors[i,4]*1e6))
        stats.write('\n# *** Over all surface *** # \nPeak-to-Valley[nm]\tMinimum-Height[nm]\tMaximum-Height[nm]\n'+'{0:.3f}\t{1:.3f}\t{2:.3f}\n'.format(np.max(new_mtx[1:,1:]*1e9) - np.min(new_mtx[1:,1:]*1e9), np.min(new_mtx[1:,1:]*1e9), np.max(new_mtx[1:,1:]*1e9)))
    stats.close()
    
    
    
def gen_figure_error(L=400e-3, W=40e-3, stepL=1e-3, stepW=1e-3, 
                     rmsL=1e-9, rmsW=1e-9, betaL=2.0, betaW=2.0, 
                     typeL='h', typeW='h', seedL=8787, seedW=8454,
                     filename='', plot=False, print_out=False):
    
    from srxraylib.metrology import profiles_simulation
    
    # ==== Main Parameters ============================== #
    
    if(typeL == 'h'):
        typeL = 0 # (0) Normalize by Height RMS, (1) Normalize by Slope RMS
    elif(typeL == 's'):
        typeL = 1

    if(typeW == 'h'):
        typeW = 0 # (0) Normalize by Height RMS, (1) Normalize by Slope RMS
    elif(typeW == 's'):
        typeW = 1
        
    if(seedL == 0):
        fct = 1e4 if np.random.random() <= 0.5 else 1e3
        seedL = int(np.random.random()*fct)

    if(seedW == 0):
        fct = 1e4 if np.random.random() <= 0.5 else 1e3
        seedW = int(np.random.random()*fct)
        
    
    # ==== Creates 2D matrix ============================ #
    x, y, z  = profiles_simulation.simulate_profile_2D(combination='FF',
                         mirror_length=L, step_l=stepL, random_seed_l=seedL, 
                         error_type_l=typeL, rms_l=rmsL,
                         power_law_exponent_beta_l=betaL,
                         mirror_width=W, step_w=stepW, random_seed_w=seedW, 
                         error_type_w=typeW, rms_w=rmsW,
                         power_law_exponent_beta_w=betaW)
    
    # ==== Formats matrix to the required by function === #                      
    mtx_fmt = np.zeros((len(x)+1, len(y)+1))
    mtx_fmt[0, 1:] = y
    mtx_fmt[1:, 0] = x
    mtx_fmt[1:,1:] = np.transpose(z)
    
    if(print_out):
        print(filename)    
        print('RMS = {0:.3f} nm'.format(np.std(z[:, 0]*1e9)))
        print('PV = {0:.3f} nm\n'.format(np.max(z[:, 0]*1e9)-np.min(z[:, 0]*1e9)))
    
                         
    # # ==== Writes files for shadow input ================ #                     
    if(filename != ''):
        write_shadow_height_error(mtx_fmt, filename, 1e3, ' ') # [mm]
    
    # # ==== Preview Plots ================================ #
    if(plot):
        
        plt.figure(figsize=(6,1))
        plt.pcolormesh(y, x, np.transpose(z))
        plt.autoscale(enable=True, tight=True)
        
        plt.figure()
        plt.plot(y, z[:, 0]*1e9)
        plt.xlabel('L')
        plt.ylabel('height error')
        
        plt.show()     
        
    return mtx_fmt

def minimize_slope_and_height(x, *args):
    
    from optlnls.math import derivate
    
    betaL = x
    
    inputs = args[0]

    mtx = gen_figure_error(L=inputs['L'], W=inputs['W'], 
                           stepL=inputs['stepL'], stepW=inputs['stepW'], 
                           rmsL=inputs['h_rms'], rmsW=inputs['rmsW'], 
                           betaL=betaL, betaW=inputs['betaW'], 
                           typeL='h', typeW=inputs['typeW'], 
                           seedL=inputs['seedL'], seedW=inputs['seedW'])    
  
    l = mtx[0, 1:]
    h_cut = mtx[1, 1:]
    s_cut = derivate(l, h_cut)
    s_rms = np.std(s_cut)
    #print('evaluating... ', s_rms*1e9, inputs['s_rms']*1e9)
    
    return np.abs(s_rms - inputs['s_rms'])*1e9
    
    
    
        
def gen_figure_error_multi(L=400e-3, W=40e-3, stepL=0.5e-3, stepW=0.5e-3, 
                           betaW=2.0, rmsW=1e-9, typeW='h',
                           list_height_rms=[1.0e-9], list_slope_rms=[1.0e-9], 
                           seedL=8787, seedW=8454, prefix='test', tolL=0.1):
    
    inputs = {
            "L":L,
            "W":W,
            "stepL":stepL,
            "stepW":stepW,
            "seedL":seedL,
            "seedW":seedW,
            "betaW":betaW,
            "rmsW":rmsW,
            "typeW":typeW            
            }
    
    counter = 1
    
    for h_rms in list_height_rms:
        for s_rms in list_slope_rms:
            inputs["h_rms"] = h_rms
            inputs['s_rms'] = s_rms
            args = (inputs)
            res = minimize(minimize_slope_and_height, x0=np.array([2.0]), args=args, 
                           method='L-BFGS-B', bounds=[(0.05, 15.0)])
            
            print('Minimization succesful: ', res.success)
            betaL = res.x
            inputs["betaL"] = betaL
            
            gen_figure_error(L=L, W=W, stepL=stepL, stepW=stepW, 
                             rmsL=h_rms, rmsW=rmsW, betaL=betaL, betaW=betaW, 
                             typeL='h', typeW=typeW, seedL=seedL, seedW=seedW,
                             filename=prefix+'_{0:d}.dat'.format(int(counter)))
            
            counter += 1          
    

def spherical(x, R, x0, y0):
    return -numpy.sqrt(R**2 - (x-x0)**2) + y0


def func_ellipse_slopes(x, p, q, theta):
    #
    # returns y'(x), the slopes of an ellipse defined by p,q, and theta
    #


    a = (p + q) / 2
    b = numpy.sqrt( p * q) * numpy.sin(theta)
    c = numpy.sqrt(a*a - b*b)

    epsilon = c / a

    # (x0,y0) are the coordinates of the center of the mirror
    # x0 = (p*p - q*q) / 4 / c
    x0 = (p - q) / 2 / epsilon
    y0 = -b * numpy.sqrt(1 - ((x0/a)**2))

    # the versor normal to the surface at the mirror center is -grad(ellipse)
    xnor = -2 * x0 / a**2
    ynor = -2 * y0 / b**2
    modnor = numpy.sqrt(xnor**2 + ynor**2)
    xnor /= modnor
    ynor /= modnor
    # tangent  versor is perpendicular to normal versor
    xtan =  ynor
    ytan = -xnor


    A = 1/b**2
    B = 1/a**2
    C = A

    CCC = numpy.zeros(11)
    #CCC[1] = A
    CCC[2] = B*xtan**2 + C*ytan**2
    CCC[3] = B*xnor**2 + C*ynor**2
    #CCC[4] = .0
    CCC[5] = 2*(B*xnor*xtan+C*ynor*ytan)
    #CCC[6] = .0
    #CCC[7] = .0
    CCC[8] = .0
    CCC[9] = 2*(B*x0*xnor+C*y0*ynor)
    CCC[10]= .0

    # ellipse implicit eq is c2 x^2 + c3 y^2 + c5 x y + c8 x + c9 y + c10 = 0
    # AA y^2 + BB y + CC = 0
    AA = CCC[3]
    BB = CCC[5]*x + CCC[9]
    CC = CCC[2]*x*x + CCC[8]*x + CCC[10]
    DD = BB*BB-4*AA*CC
    #yell = (-BB - numpy.sqrt(DD) )/(2*AA)
    #yellp = numpy.gradient(yell,(x[1]-x[0]))

    #calculate derivatives (primes P)
    BBP = CCC[5]
    CCP = 2*CCC[2]*x+CCC[8]
    DDP = 2*BB*BBP -4*AA*CCP
    ells = (-1/2/AA) * (BBP + DDP/2/numpy.sqrt(DD))

    return ells


def func_ellipse(x, p, q, theta):
    #
    # returns y(x), the heights of an ellipse defined by p,q, and theta
    #

    a = (p + q) / 2
    b = numpy.sqrt( p * q) * numpy.sin(theta)
    c = numpy.sqrt(a*a - b*b)

    epsilon = c / a

    # (x0,y0) are the coordinates of the center of the mirror
    # x0 = (p*p - q*q) / 4 / c
    x0 = (p - q) / 2 / epsilon
    y0 = -b * numpy.sqrt(1 - ((x0/a)**2))

    # print(">>>> func_ellipse: a=%f, b=%f, c=%f"%(a,b,c))
    # print(">>>> func_ellipse: x0=%f, y0=%f"%(x0,y0))

    # the versor normal to the surface at the mirror center is -grad(ellipse)
    xnor = -2 * x0 / a**2
    ynor = -2 * y0 / b**2
    modnor = numpy.sqrt(xnor**2 + ynor**2)
    xnor /= modnor
    ynor /= modnor
    # tangent  versor is perpendicular to normal versor
    xtan =  ynor
    ytan = -xnor


    A = 1/b**2
    B = 1/a**2
    C = A

    CCC = numpy.zeros(11)
    #CCC[1] = A
    CCC[2] = B*xtan**2 + C*ytan**2
    CCC[3] = B*xnor**2 + C*ynor**2
    #CCC[4] = .0
    CCC[5] = 2*(B*xnor*xtan+C*ynor*ytan)
    #CCC[6] = .0
    #CCC[7] = .0
    CCC[8] = .0
    CCC[9] = 2*(B*x0*xnor+C*y0*ynor)
    CCC[10]= .0

    # ellipse implicit eq is c2 x^2 + c3 y^2 + c5 x y + c8 x + c9 y + c10 = 0
    # AA y^2 + BB y + CC = 0
    AA = CCC[3]
    BB = CCC[5]*x + CCC[9]
    CC = CCC[2]*x*x + CCC[8]*x + CCC[10]

    #yell1 = (-BB + numpy.sqrt(BB*BB-4*AA*CC) )/(2*AA)
    yell2 = (-BB - numpy.sqrt(BB*BB-4*AA*CC) )/(2*AA)

    return yell2
    


def calc_radius_of_curvature(x, y):
    """

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.

    Returns
    -------
    x2 : array_like
        new coordinates x.
    R : array_like
        Radius of curvature in x2.

    """

    x1, d1y = derivate(x, y)
    x2, d2y = derivate(x1, d1y)

    d1y_interp = interp1d(x1, d1y)
    d1y_i = d1y_interp(x2)

    R = (1 + d1y_i ** 2) ** (3 / 2) / np.abs(d2y)

    return x2, R

    
    
def radius_difference(x, *args):
    
    # for LTP analysis
    
    # get x_ell and radius_ell from args
    x_ell = args[0]
    radius_ell = args[1]
    R0 = args[2]
    
    radius_ell_interp = interp1d(x_ell, radius_ell)
    
    # manually chack if x is inside x_ell bounds
    if((x <= x_ell.min()) or (x >= x_ell.max())):
        return np.inf # return large value to fail minimization
    else:
        return radius_ell_interp(x) - R0
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
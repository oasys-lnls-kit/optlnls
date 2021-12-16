#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 17:09:15 2021

@author: humberto.junior
"""


# Lib:

import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d


################################################### Basic Functions ###################################################


def rotate(theta, x, y):
    
    if (len(x)%2):
    
        index = int((len(x)-1)/2)
        
        desloc_x = x[index]
        desloc_y = y[index]
    
    else:
    
        desloc_x = (x[int(len(x)/2+0.5)] + x[int(len(x)/2-0.5)])/2
        desloc_y = (y[int(len(y)/2+0.5)] + y[int(len(y)/2-0.5)])/2
    
    
    x = x - desloc_x
    y = y - desloc_y
    
    x_rot = x*np.cos(theta)-y*np.sin(theta)
    y_rot = x*np.sin(theta)+y*np.cos(theta)
    
    return x_rot+desloc_x, y_rot+desloc_y


def ButterworthFilter(x, y, Spatial_Cutoff=1):
    
    import scipy.signal as signal
    
    Nyquist_Freq = 2 * (x[1]-x[0])

    Wn = Nyquist_Freq / Spatial_Cutoff

    if Spatial_Cutoff < Nyquist_Freq:
        Wn = 1
    b, a = signal.butter(1, Wn, 'low')
    output_signal = signal.filtfilt(b, a, y)

    return output_signal

    
################################################# Shift between curves ################################################
    

# Shift in x direction:

def calc_diff_rms_shift_x(shift, args):
    
    """
    Given two curves and a "shift" value in "x" direction, calculates the Root Mean Square of the difference between the second curve shifted and the first curve.
    
    Parameters are:
        
        -> shift: value that will be used to shift the second curve in "x" direction.
        -> args: tuple of the form [x1, x2, f1, f2] containing the position x and f(x) values of the two curves:  
            x1: position array of the first curve.
            x2: position array of the second curve.
            f1: array containing f1(x1) values.
            f2: array containing f2(x2) values.           
            
    """
    
    x1, x2, f1, f2 = args # Unpack

    x2 = x2 - shift
    
    x_bar = np.linspace(np.max([x1[0], x2[0]]), np.min([x1[-1], x2[-1]]), int(np.max([len(x1), len(x2)])))
    
    f1_interp = interp1d(x1, f1)
    f2_interp = interp1d(x2, f2) 
    
    f1_i = f1_interp(x_bar)
    f2_i = f2_interp(x_bar)
    
    f_diff = f2_i - f1_i
    
    rms = np.std(f_diff)
    
    return rms


def find_shift_x(x1, x2, f1, f2, shift0=1):
    
    """
    Calculates the shift in "x" direction between two curves by minimizing the Root Mean Square of their difference. Parameters are:
        
        -> x1: position array of the first curve.
        -> x2: position array of the second curve.
        -> f1: array containing f1(x1) values.
        -> f2: array containing f2(x2) values.
        -> shift0: shift initial guess.
    """
    
    args = [x1, x2, f1, f2]
    
    res = minimize(fun=calc_diff_rms_shift_x, x0=shift0, args=args, options={'disp':True})
    
    shift = res.x[0]
    
    return shift


# Shift in y direction:
    
def calc_rms_y(shift, args):
    
    """
    Given two curves and a "shift" value in "y" direction, calculates the Root Mean Square of the difference between the second curve shifted and the first curve.
    
    Parameters are:
        
        -> shift: value that will be used to shift the second curve in "y" direction.
        -> args: tuple of the form [x1, x2, f1, f2] containing the position x and f(x) values of the two curves:  
            x1: position array of the first curve.
            x2: position array of the second curve.
            f1: array containing f1(x1) values.
            f2: array containing f2(x2) values.           
            
    """    
    
    x1, x2, f1, f2 = args # Unpack

    f2 = f2 - shift
    
    x_bar = np.linspace(np.max([x1[0], x2[0]]), np.min([x1[-1], x2[-1]]), int(np.max([len(x1), len(x2)])))
    
    f1_interp = interp1d(x1, f1)
    f2_interp = interp1d(x2, f2) 
    
    f1_i = f1_interp(x_bar)
    f2_i = f2_interp(x_bar)
    
    f_diff = f2_i - f1_i
    
    rms = np.sqrt(np.mean(f_diff**2))
    
    return rms


def calc_shift_y(x1, x2, f1, f2, shift0=40):
    
    """
    Calculates the shift in "y" direction between two curves by minimizing the Root Mean Square of their difference. Parameters are:
        
        -> x1: position array of the first curve.
        -> x2: position array of the second curve.
        -> f1: array containing f1(x1) values.
        -> f2: array containing f2(x2) values.
        -> shift0: shift initial guess.
    """
    
    args = [x1, x2, f1, f2]
    
    res = minimize(fun=calc_rms_y, x0=shift0, args=args, options={'disp':True})
    
    shift = res.x[0]
    
    return shift 


# Shifts in x and y directions:
    
def calc_diff_rms_shift_xy(shift, args):
    
    """
    Given two curves and a shifts values, calculates the Root Mean Square of the difference between the second curve shifted and the first curve.
    
    Parameters are:
        
        -> shift: vector of the form [shift_x, shift_y] containing the shift values in each direction for the second curve:
            - shift_x: shift value in "x" direction.
            - shift_y: shift value in "y" direction.
            
        -> args: tuple of the form [x1, x2, f1, f2] containing the position x and f(x) values of the two curves:  
            - x1: position array of the first curve.
            - x2: position array of the second curve.
            - f1: array containing f1(x1) values.
            - f2: array containing f2(x2) values.           
            
    """
    
    x1, x2, f1, f2 = args # Unpack
    
    shift_x, shift_y = shift # Unpack

    x2 = x2 - shift_x
    
    f2 = f2 - shift_y
    
    x_bar = np.linspace(np.max([x1[0], x2[0]]), np.min([x1[-1], x2[-1]]), int(np.max([len(x1), len(x2)])))
    
    f1_interp = interp1d(x1, f1)
    f2_interp = interp1d(x2, f2) 
    
    f1_i = f1_interp(x_bar)
    f2_i = f2_interp(x_bar)
    
    f_diff = f2_i - f1_i
    
    rms = np.sqrt(np.mean(f_diff**2))
    
    return rms        
            
            
def find_shift_xy(x1, x2, f1, f2, shift0=[1, 1]):
    
    """
    Calculates the shift in "x" and "y" directions between two curves by minimizing the Root Mean Square of their difference. Parameters are:
        
        -> x1: position vector of the first curve.
        -> x2: position vector of the second curve.
        -> f1: vector containing f1(x1) values.
        -> f2: vector containing f2(x2) values.
        -> shift0: vector of the form [shift_x0, shift_y0] containing shift initial guess in each direction:
            - shift_x0: shift initial guess in "x" direction.
            - shift_y0: shift initial guess in "y" direction.
    """
    
    args = [x1, x2, f1, f2]
    
    res = minimize(fun=calc_diff_rms_shift_xy, x0=shift0, args=args, options={'disp':True})
    
    shift_x = res.x[0]
    
    shift_y = res.x[1]
    
    return shift_x, shift_y 


###################################################### Stitching ######################################################
    

def calc_diff_rms_theta_shift(theta_shift, args):
    
    '''
    args = [[x0, y0], [x1, y1], ..., [xn-1, yn-1]]
    theta_shift = [theta0, shift_x0, shift_y0, theta1, shift_x1, shift_y1, ..., theta(n-1), shift_x(n-1), shift_y(n-1)]
    
    '''
       
#    Rotation and Shift:        
        
    theta_shift = theta_shift.reshape([len(args), 3]) # Converts the 1-D array to a matrix of shape: [[theta1, shift_x1, shift_y1], [theta2, shift_x2, shift_y2], ..., [theta(n-1), shift_x(n-1), shift_y(n-1)]]
    
    sum_rms = 0
    
    new_curves = []
    rot_curves = []
    
    for i in range(len(args)):
        
        rot_curves.append(rotate(theta_shift[i][0], args[i][0], args[i][1]))
        
    rot_curves = np.array(rot_curves)
    
    for i in range(len(args)):
        
        new_curves.append([rot_curves[i][0]-theta_shift[i][1], rot_curves[i][1]-theta_shift[i][2]])
        
#   Calculating the RMS of the difference:
        
    for i in range(len(new_curves)-1):
                            
        for j in range(i+1, len(new_curves), 1):
            
            try:
                
                x1, f1, x2, f2 = new_curves[i][0],  new_curves[i][1], new_curves[j][0], new_curves[j][1] # Unpack
                
                x_bar = np.linspace(np.max([x1[0], x2[0]]), np.min([x1[-1], x2[-1]]), int(0.7*np.max([len(x1), len(x2)])))
                
                f1_interp = interp1d(x1, f1)
                f2_interp = interp1d(x2, f2) 
                
                f1_i = f1_interp(x_bar)
                f2_i = f2_interp(x_bar)
                
                f_diff = f2_i - f1_i
                
                rms = np.sqrt(np.mean(f_diff**2))
                
                sum_rms = sum_rms + rms
                
            except(ValueError):
                
                break
    
    return sum_rms        


def find_theta_shift(args, theta_shift_0, bounds, maxiter=5000, UseMinimize=True):
    
    '''
    args = [[x0, y0], [x1, y1], ..., [xn-1, yn-1]]
    theta_shift_0 = [theta0, shift_x0, shift_y0, theta1, shift_x1, shift_y1, ..., theta(n-1), shift_x(n-1), shift_y(n-1)]
    
    '''
    
    from scipy.optimize import differential_evolution

    if (bounds==0):
        
        res = minimize(fun=calc_diff_rms_theta_shift, x0=theta_shift_0, args=args, options={'disp':True, 'maxiter':maxiter})
        
    else:
        
        if(UseMinimize):
            
            #bnds = int(len(theta_shift_0)/3)*((None,None), (None,None), (None,None))
            
            res = minimize(fun=calc_diff_rms_theta_shift, x0=theta_shift_0, args=args, bounds=bounds, options={'disp':True, 'maxiter':maxiter})
            
        else:
            
            res = differential_evolution(func=calc_diff_rms_theta_shift, args=(args,), bounds=bounds, maxiter=maxiter)
            
            
    print('\n', res, '\n')
    
    return res.x


def stitching(curves_list, theta_shift_0, bounds=0, maxiter=5000, UseMinimize=True, UseButterworthFilter=False, Spatial_Cutoff=0.01, PlotCurves2by2=False, SaveFigs=False, Save_txt=False, filename_prefix=''):
    
    '''
    Stitches a set of curves with an overlap between each other by minimizing the sum of the RMS of the difference between the overlap regions.
    
    Parameters are:
        
        -> curves_list: Array or tuple of shape [[X0, Y0], [X1, Y1], ..., [Xk, Yk],..., [Xn-1, Yn-1]] containing the n curves to be stitched. 
            - Xk and Yk are 1D arrays containing, respectively, the position and f(Xk) values of the k-th curve.
            
        -> theta_shift_0: 1D array or tuple of shape [theta(0), shift_x(0), shift_y(0), ..., theta(k), shift_x(k), shift_y(k), ..., theta(n-1), shift_x(n-1), shift_y(n-1)] containing the initial guess values of rotation and shift for each curve.
            - theta(k), shift_x(k), shift_y(k) are, respectively, the rotation angle, the shift in the horizontal direction and the shift in the vertical direction for the k-th curve.
        
        -> bounds (optional): Bounds for variables. Sequence containing a (lower limit, upper limit) pair for each element in theta_shift_0. It is required to have len(bounds) == len(theta_shift_0). If 0, it does not consider bounds. Default is 0.
        
        -> maxiter (optional): int. Maximum number of iterations. Default is 5000.
        
        -> UseMinimize (optional): If "True" it uses scipy.optimize.minimize for the optimization. If "False" it uses scipy.optimize.differential_evolution. Notice that if bounds=0, scipy.optimize.minimize is used obligatorily. Default is True.
        
        -> UseButterworthFilter (optional): If "True", filters each curve in curves_list with a Butterworth filter. Default is False.
        
        -> Spatial_Cutoff (optional): Cut frequency for the Butterworth filter. Only used if UseButterworthFilter=True. Default is 0.01.
        
        -> PlotCurves2by2 (optional): If "True", it plots graphics showing the stitched curves in pairs. Default is False. Warning: If the number of curves to be stitched is large, many figures will be created, which may consume a lot of memory. Using PlotCurves2by2=False is recommended in this case.
        
        -> SaveFigs (optional): If "True", it saves the generated figures. Default is False.  
        
        -> Save_txt (optional): If "True", it saves each stitched curve in a .txt file. Default is False.
        
        -> filename_prefix (optional): file name prefix to save data.
            
    Returns:
        
        -> cx: 1D array with the position values of the stitched curves average. 
        
        -> cy: 1D array with the stitched curves average.
        
        -> final_curves: Array of the same shape as "curves_list" containing the stitched curves.
    
    '''
    
    # Libraries:

    import matplotlib.pyplot as plt
    from optlnls.math import common_region_average
    
    
    # Aplying filter:
    
    if(UseButterworthFilter):
        
        for i in range(len(curves_list)):
            
            curves_list[i][1] = 1e+9*(ButterworthFilter(1e-3*curves_list[i][0], 1e-9*curves_list[i][1], Spatial_Cutoff=Spatial_Cutoff))
            
            
    # Stitching:
    
    final_curves = []
    
    curves_rot = []      
    
        
    thetas_shift = find_theta_shift(curves_list, theta_shift_0, bounds, maxiter, UseMinimize)            
    
    theta = thetas_shift[0::3]
    
    shift_x = thetas_shift[1::3]
    
    shift_y = thetas_shift[2::3]
    
    for j in range(len(curves_list)):
        
        curves_rot.append(rotate(theta[j], curves_list[j][0], curves_list[j][1]))
    
    for j in range(len(curves_list)):
        
        final_curves.append([(curves_rot[j][0]-shift_x[j]), (curves_rot[j][1]-shift_y[j])])
                                   
    
    final_curves = np.array(final_curves)
    
    cx, cy = common_region_average(final_curves)
    
    cy = cy - cy.mean()
            

    # Graphics:
        
    plt.figure()   
    
    for i in range(len(curves_list)):
        plt.plot(curves_list[i][0], curves_list[i][1])  
        
    plt.ylabel('Height [nm]')
    plt.xlabel('Position [mm]')
    plt.title('Initial Data')
    plt.minorticks_on()
    plt.tick_params(which='both', axis='both', direction='in', right=True, top=True)
    plt.grid(which='both', alpha=0.2)
    
    if(SaveFigs):
        plt.savefig(filename_prefix+'_initial_data.png', dpi=600)
    
        
    plt.figure()
    
    for i in range(len(final_curves)):
        plt.plot(final_curves[i][0], final_curves[i][1])

    plt.ylabel('Height [nm]')
    plt.xlabel('Position [mm]')
    plt.title('Stitched curves')
    plt.minorticks_on()
    plt.tick_params(which='both', axis='both', direction='in', right=True, top=True)
    plt.grid(which='both', alpha=0.2)
    
    if(SaveFigs):
        plt.savefig(filename_prefix+'_after_stitching.png', dpi=600)

    
    plt.figure()
            
    plt.plot(cx, cy)
    plt.ylabel('Height [nm]')
    plt.xlabel('Position [mm]')
    plt.title('Stitched curves average')
    plt.minorticks_on()
    plt.tick_params(which='both', axis='both', direction='in', right=True, top=True)
    plt.grid(which='both', alpha=0.2)
    
    if(SaveFigs):
        plt.savefig(filename_prefix+'_curves_average.png', dpi=600)
        
    
    if(PlotCurves2by2):
        
    #    for i in range(1, len(final_curves), 1):
    #        plt.figure() 
    #        for j in range(i+1):
    #            plt.plot(final_curves[j][0], final_curves[j][1])
        
        
        for i in range(len(final_curves)-1):
            plt.figure()
            plt.plot(final_curves[i][0], final_curves[i][1])
            plt.plot(final_curves[i+1][0], final_curves[i+1][1])
            plt.ylabel('Height [nm]')
            plt.xlabel('Position [mm]')
            plt.title('Pair %d' %(i+1))
            plt.minorticks_on()
            plt.tick_params(which='both', axis='both', direction='in', right=True, top=True)
            plt.grid(which='both', alpha=0.2)
            
            if(SaveFigs):
                plt.savefig(filename_prefix+'_pair_%d.png' %(i+1), dpi=600)
  

    # Saving .txt files:
    
    filename = filename_prefix+'_stitched_curves_average.txt'

    with open(filename, 'w') as f:
        
        f.write('#Position[mm]\tHeight[nm]\n')
        
        for i in range(len(cx)):
            
            f.write('%.10f\t%.10f\n' %(cx[i], cy[i]))
    
    
    if(Save_txt):
        
        for i in range(len(final_curves)):
            
            filename = filename_prefix+'_stitched_curve_%d.txt'%(i)
            
            x_stitched, y_stitched = final_curves[i] # Unpack
    
            with open(filename, 'w') as f:
                
                f.write('#Position[mm]\tHeight[nm]\n')
                
                for j in range(len(x_stitched)):
                    
                    f.write('%.10f\t%.10f\n' %(x_stitched[j], y_stitched[j]))
      
        
    return cx, cy, final_curves


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
########################################## DISCLAIMER ##########################################
#                                                                                              #
# Python functions for hybrid screen calculations for optics simulations developed in LNLS,    #
# CNPEM, based on OASYS1 and ShadowOui and licensed under GNU GENERAL PUBLIC LICENSE.          #
# OASYS1    : (https://github.com/oasys-kit/OASYS1)                                            #
# ShadowOui : (https://github.com/oasys-kit/ShadowOui)                                         #
#                                                                                              #
# The original HYBRID code works inside OASYS environment, therefore the modifications made    #
# were purely to automatize the functions for use in python scripts, removing widget classes   #
# dependences.                                                                                 #
#                                                                                              #
################################################################################################

Created on Fri May  6 10:59:06 2022

@author: joao.astolfo
"""
import oasys.util.oasys_util as OU
import numpy as np
import xraylib
import copy

from orangecontrib.shadow.util.shadow_objects import ShadowBeam, ShadowOpticalElement
from orangecontrib.shadow.util.shadow_util import ShadowPhysics, ShadowPreProcessor

from srxraylib.util.data_structures import ScaledArray, ScaledMatrix

from scipy.interpolate import RectBivariateSpline

from oasys.widgets import congruence

def sh_read_gfile(gfilename):
    return ShadowOpticalElement.create_oe_from_file(congruence.checkFile(gfilename))


def get_delta(input_parameters, calculation_parameters):
    density = xraylib.ElementDensity(xraylib.SymbolToAtomicNumber(input_parameters.crl_material))

    energy_in_KeV = ShadowPhysics.getEnergyFromWavelength(calculation_parameters.gwavelength*input_parameters.widget.workspace_units_to_m*1e10)/1000
    delta = 1-xraylib.Refractive_Index_Re(input_parameters.crl_material, energy_in_KeV, density)

    return delta

#########################################################

def read_shadow_beam(shadow_beam, lost=False):
    cursor_go = np.where(shadow_beam._beam.rays[:, 9] == 1)

    image_beam_rays = copy.deepcopy(shadow_beam._beam.rays[cursor_go])
    image_beam_rays[:, 11] = np.arange(1, len(image_beam_rays) + 1, 1)

    out_beam_go = ShadowBeam()
    out_beam_go._beam.rays = image_beam_rays

    if lost:
        cursor_lo = np.where(shadow_beam._beam.rays[:, 9] != 1)

        lost_rays = copy.deepcopy(shadow_beam._beam.rays[cursor_lo])
        lost_rays[:, 11] = np.arange(1, len(lost_rays) + 1, 1)

        out_beam_lo = ShadowBeam()
        out_beam_lo._beam.rays = lost_rays

        return out_beam_go, out_beam_lo
    else:
        return out_beam_go

#########################################################

def sh_readsh(shfilename):
    image_beam = ShadowBeam()
    image_beam.loadFromFile(congruence.checkFile(shfilename))

    return read_shadow_beam(image_beam)

#########################################################

def sh_readangle(filename, mirror_beam=None):
    values = np.loadtxt(congruence.checkFile(filename))
    dimension = len(mirror_beam._beam.rays)

    angle_inc = np.zeros(dimension)
    angle_ref = np.zeros(dimension)

    ray_index = 0
    for index in range(0, len(values)):
        if values[index, 3] == 1:
            angle_inc[ray_index] = values[index, 1]
            angle_ref[ray_index] = values[index, 2]

            ray_index += 1

    return angle_inc, angle_ref

#########################################################

def sh_readsurface(filename, dimension):
    if dimension == 1:
        values = np.loadtxt(congruence.checkFile(filename))

        return ScaledArray(values[:, 1], values[:, 0])
    elif dimension == 2:
        x_coords, y_coords, z_values = ShadowPreProcessor.read_surface_error_file(filename)

        return ScaledMatrix(x_coords, y_coords, z_values)

def h5_readsurface(filename):
    x_coords, y_coords, z_values = OU.read_surface_file(filename)

    return ScaledMatrix(x_coords, y_coords, z_values.T)


#########################################################

def calculate_function_average_value(function, x_min, x_max, sampling=100):
    sampled_function = ScaledArray.initialize_from_range(np.zeros(sampling), x_min, x_max)
    sampled_function.np_array = function(sampled_function.scale)

    return np.average(sampled_function.np_array)

#########################################################

def hy_findrmsslopefromheight(wmirror_l):
    array_first_derivative = np.gradient(wmirror_l.np_array, wmirror_l.delta())

    return hy_findrmserror(ScaledArray(array_first_derivative, wmirror_l.scale))

#########################################################

def hy_findrmserror(data):
    wfftcol = np.absolute(np.fft.fft(data.np_array))

    waPSD = (2 * data.delta() * wfftcol[0:int(len(wfftcol)/2)]**2)/data.size() # uniformed with IGOR, FFT is not simmetric around 0
    waPSD[0] /= 2
    waPSD[len(waPSD)-1] /= 2

    fft_scale = np.fft.fftfreq(data.size())/data.delta()

    waRMS = np.trapz(waPSD, fft_scale[0:int(len(wfftcol)/2)]) # uniformed with IGOR: Same kind of integration, with automatic range assignement

    return np.sqrt(waRMS)

#########################################################

# 1D
def calculate_focal_length_ff(min_value, max_value, n_peaks, wavelength):
#    return (min(abs(max_value), abs(min_value))*2)**2/n_peaks/2/0.88/wavelength  #xshi used for now, but will have problem when the aperture is off center
    return (max_value - min_value)**2/n_peaks/2/0.88/wavelength  #xshi suggested, but need to first fix the problem of getting the fake solution of mirror aperture by SHADOW.


def calculate_focal_length_ff_2D(min_x_value, max_x_value, min_z_value, max_z_value, n_peaks, wavelength):
    return (min((max_z_value - min_z_value), (max_x_value - min_x_value)))**2/n_peaks/2/0.88/wavelength


def calculate_fft_size(min_value, max_value, wavelength, propagation_distance, fft_npts, factor=100):
    return int(min(factor * (max_value - min_value) ** 2 / wavelength / propagation_distance / 0.88, fft_npts))


def get_mirror_phase_shift(abscissas,
                           wavelength,
                           w_angle_function,
                           w_l_function,
                           mirror_profile):
    return (-1.0) * 4 * np.pi / wavelength * np.sin(w_angle_function(abscissas)/1e3) * mirror_profile.interpolate_values(w_l_function(abscissas))


def get_grating_phase_shift(abscissas,
                            wavelength,
                            w_angle_function,
                            w_angle_ref_function,
                            w_l_function,
                            grating_profile):
    return (-1.0) * 2 * np.pi / wavelength * (np.sin(w_angle_function(abscissas)/1e3) + np.sin(w_angle_ref_function(abscissas)/1e3)) * grating_profile.interpolate_values(w_l_function(abscissas))


def get_crl_phase_shift(thickness_error_profile, input_parameters, calculation_parameters, coordinates):
    coord_x = thickness_error_profile.x_coord
    coord_y = thickness_error_profile.y_coord
    thickness_error = thickness_error_profile.z_values

    interpolator = RectBivariateSpline(coord_x, coord_y, thickness_error, bbox=[None, None, None, None], kx=1, ky=1, s=0)

    wavefront_coord_x = coordinates[0]
    wavefront_coord_y = coordinates[1]

    thickness_error = interpolator(wavefront_coord_x, wavefront_coord_y)
    thickness_error[np.where(np.isnan(thickness_error))] = 0.0
    thickness_error *= input_parameters.crl_scaling_factor

    return -2*np.pi*calculation_parameters.crl_delta*thickness_error/calculation_parameters.gwavelength
    
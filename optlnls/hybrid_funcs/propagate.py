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

Created on Fri May  6 10:27:57 2022

@author: joao.astolfo
"""
import numpy

#from utility import hy_findrmsslopefromheight, calculate_focal_length_ff, calculate_focal_length_ff_2D
#from utility import calculate_fft_size, get_mirror_phase_shift, get_grating_phase_shift, get_crl_phase_shift

from optlnls.hybrid_funcs.utility import *

from srxraylib.util.data_structures import ScaledArray, ScaledMatrix
from srxraylib.waveoptics.wavefront import Wavefront1D
from srxraylib.waveoptics.wavefront2D import Wavefront2D
from srxraylib.waveoptics import propagator
from srxraylib.waveoptics import propagator2D

def propagate_1D_x_direction(calculation_parameters, input_parameters):

    scale_factor = 1.0

    shadow_oe = calculation_parameters.shadow_oe_end

    global_phase_shift_profile = None

    if shadow_oe._oe.F_MOVE == 1 and shadow_oe._oe.Y_ROT != 0.0:
        if input_parameters.ghy_calcType == 3 or input_parameters.ghy_calcType == 4:
            global_phase_shift_profile = calculation_parameters.w_mirror_lx
        elif input_parameters.ghy_calcType == 2:
            global_phase_shift_profile = ScaledArray.initialize_from_range(numpy.zeros(3), shadow_oe._oe.RWIDX2, shadow_oe._oe.RWIDX1)

        global_phase_shift_profile.set_values(global_phase_shift_profile.get_values() +
                                              global_phase_shift_profile.get_abscissas()*numpy.sin(numpy.radians(-shadow_oe._oe.Y_ROT)))
    elif input_parameters.ghy_calcType == 3 or input_parameters.ghy_calcType == 4:
        global_phase_shift_profile = calculation_parameters.w_mirror_lx

    if input_parameters.ghy_calcType == 3 or input_parameters.ghy_calcType == 4:
        rms_slope = hy_findrmsslopefromheight(global_phase_shift_profile)

        print("Using RMS slope = " + str(rms_slope))

        average_incident_angle = numpy.radians(90-calculation_parameters.shadow_oe_end._oe.T_INCIDENCE)*1e3
        average_reflection_angle = numpy.radians(90-calculation_parameters.shadow_oe_end._oe.T_REFLECTION)*1e3

        if calculation_parameters.beam_not_cut_in_x:
            dp_image = numpy.std(calculation_parameters.xx_focal_ray)/input_parameters.ghy_focallength
            dp_se = 2 * rms_slope * numpy.sin(average_incident_angle/1e3)	# different in x and z
            dp_error = calculation_parameters.gwavelength/2/(calculation_parameters.ghy_x_max-calculation_parameters.ghy_x_min)

            scale_factor = max(1, 5*min(dp_error/dp_image, dp_error/dp_se))

    # ------------------------------------------
    # far field calculation
    # ------------------------------------------
    if calculation_parameters.do_ff_x:
        focallength_ff = calculate_focal_length_ff(calculation_parameters.ghy_x_min,
                                                   calculation_parameters.ghy_x_max,
                                                   input_parameters.ghy_npeak,
                                                   calculation_parameters.gwavelength)

        if input_parameters.ghy_calcType == 3:
            if not (rms_slope == 0.0 or average_incident_angle == 0.0):
                focallength_ff = min(focallength_ff,(calculation_parameters.ghy_x_max-calculation_parameters.ghy_x_min) / 16 / rms_slope / numpy.sin(average_incident_angle / 1e3))#xshi changed
        elif input_parameters.ghy_calcType == 4:
            if not (rms_slope == 0.0 or average_incident_angle == 0.0):
                focallength_ff = min(focallength_ff,(calculation_parameters.ghy_x_max-calculation_parameters.ghy_x_min) / 8 / rms_slope / (numpy.sin(average_incident_angle / 1e3) + numpy.sin(average_reflection_angle / 1e3)))#xshi changed
        elif input_parameters.ghy_calcType == 2 and not global_phase_shift_profile is None:
            focallength_ff = min(focallength_ff, input_parameters.ghy_distance*4) #TODO: PATCH to be found with a formula

        fftsize = int(scale_factor * calculate_fft_size(calculation_parameters.ghy_x_min,
                                                        calculation_parameters.ghy_x_max,
                                                        calculation_parameters.gwavelength,
                                                        focallength_ff,
                                                        input_parameters.ghy_fftnpts))
        
        print("FF: creating plane wave begin, fftsize = " +  str(fftsize))

        wavefront = Wavefront1D.initialize_wavefront_from_range(wavelength=calculation_parameters.gwavelength,
                                                                number_of_points=fftsize,
                                                                x_min=scale_factor * calculation_parameters.ghy_x_min,
                                                                x_max=scale_factor * calculation_parameters.ghy_x_max)

        if scale_factor == 1.0:
            try:
                wavefront.set_plane_wave_from_complex_amplitude(numpy.sqrt(calculation_parameters.wIray_x.interpolate_values(wavefront.get_abscissas())))
            except IndexError:
                raise Exception("Unexpected Error during interpolation: try reduce Number of bins for I(Sagittal) histogram")

        wavefront.apply_ideal_lens(focallength_ff)

        if input_parameters.ghy_calcType == 3 or \
                (input_parameters.ghy_calcType == 2 and not global_phase_shift_profile is None):
           wavefront.add_phase_shifts(get_mirror_phase_shift(wavefront.get_abscissas(),
                                                             calculation_parameters.gwavelength,
                                                             calculation_parameters.wangle_x,
                                                             calculation_parameters.wl_x,
                                                             global_phase_shift_profile))
        elif input_parameters.ghy_calcType == 4:
           wavefront.add_phase_shifts(get_grating_phase_shift(wavefront.get_abscissas(),
                                                              calculation_parameters.gwavelength,
                                                              calculation_parameters.wangle_x,
                                                              calculation_parameters.wangle_ref_x,
                                                              calculation_parameters.wl_x,
                                                              global_phase_shift_profile))

        print("calculated plane wave: begin FF propagation (distance = " +  str(focallength_ff) + ")")

        propagated_wavefront = propagator.propagate_1D_fresnel(wavefront, focallength_ff)

        print("dif_xp: begin calculation")

        shadow_oe = calculation_parameters.shadow_oe_end

        imagesize = min(abs(calculation_parameters.ghy_x_max), abs(calculation_parameters.ghy_x_min)) * 2
        # 2017-01 Luca Rebuffi
        imagesize = min(imagesize,
                        input_parameters.ghy_npeak*2*0.88*calculation_parameters.gwavelength*focallength_ff/abs(calculation_parameters.ghy_x_max-calculation_parameters.ghy_x_min))

        # TODO: this is a patch: to be rewritten
        if shadow_oe._oe.F_MOVE==1 and not shadow_oe._oe.Y_ROT==0:
            imagesize = max(imagesize, 8*(focallength_ff*numpy.tan(numpy.radians(numpy.abs(shadow_oe._oe.Y_ROT))) + numpy.abs(shadow_oe._oe.OFFX)))

        imagenpts = int(round(imagesize / propagated_wavefront.delta() / 2) * 2 + 1)

        dif_xp = ScaledArray.initialize_from_range(numpy.ones(propagated_wavefront.size()),
                                                   -(imagenpts - 1) / 2 * propagated_wavefront.delta(),
                                                   (imagenpts - 1) / 2 * propagated_wavefront.delta())


        dif_xp.np_array = numpy.absolute(propagated_wavefront.get_interpolated_complex_amplitudes(dif_xp.scale))**2

        dif_xp.set_scale_from_range(-(imagenpts - 1) / 2 * propagated_wavefront.delta() / focallength_ff,
                                    (imagenpts - 1) / 2 * propagated_wavefront.delta() / focallength_ff)

        calculation_parameters.dif_xp = dif_xp

    # ------------------------------------------
    # near field calculation
    # ------------------------------------------
    if input_parameters.ghy_nf >= 1 and input_parameters.ghy_calcType > 1:  # near field calculation
        focallength_nf = input_parameters.ghy_focallength

        fftsize = int(scale_factor * calculate_fft_size(calculation_parameters.ghy_x_min,
                                                        calculation_parameters.ghy_x_max,
                                                        calculation_parameters.gwavelength,
                                                        numpy.abs(focallength_nf),
                                                        input_parameters.ghy_fftnpts))

        print("NF: creating plane wave begin, fftsize = " +  str(fftsize))

        wavefront = Wavefront1D.initialize_wavefront_from_range(wavelength=calculation_parameters.gwavelength,
                                                                number_of_points=fftsize,
                                                                x_min=scale_factor*calculation_parameters.ghy_x_min,
                                                                x_max=scale_factor*calculation_parameters.ghy_x_max)

        if scale_factor == 1.0:
            try:
                wavefront.set_plane_wave_from_complex_amplitude(numpy.sqrt(calculation_parameters.wIray_x.interpolate_values(wavefront.get_abscissas())))
            except IndexError:
                raise Exception("Unexpected Error during interpolation: try reduce Number of bins for I(Sagittal) histogram")

        wavefront.apply_ideal_lens(focallength_nf)

        if input_parameters.ghy_calcType == 3 or \
                (input_parameters.ghy_calcType == 2 and not global_phase_shift_profile is None):
           wavefront.add_phase_shifts(get_mirror_phase_shift(wavefront.get_abscissas(),
                                                             calculation_parameters.gwavelength,
                                                             calculation_parameters.wangle_x,
                                                             calculation_parameters.wl_x,
                                                             global_phase_shift_profile))
        elif input_parameters.ghy_calcType == 4:
           wavefront.add_phase_shifts(get_grating_phase_shift(wavefront.get_abscissas(),
                                                              calculation_parameters.gwavelength,
                                                              calculation_parameters.wangle_x,
                                                              calculation_parameters.wangle_ref_x,
                                                              calculation_parameters.wl_x,
                                                              global_phase_shift_profile))

        print("calculated plane wave: begin NF propagation (distance = " + str(input_parameters.ghy_distance) + ")")

        propagated_wavefront = propagator.propagate_1D_fresnel(wavefront, input_parameters.ghy_distance)

        # ghy_npeak in the wavefront propagation image
        imagesize = (input_parameters.ghy_npeak * 2 * 0.88 * calculation_parameters.gwavelength * numpy.abs(focallength_nf) / abs(calculation_parameters.ghy_x_max - calculation_parameters.ghy_x_min))
        imagesize = max(imagesize,
                        2 * abs((calculation_parameters.ghy_x_max - calculation_parameters.ghy_x_min) * (input_parameters.ghy_distance - numpy.abs(focallength_nf))) / numpy.abs(focallength_nf))

        if input_parameters.ghy_calcType == 3:
            imagesize = max(imagesize,
                            16 * rms_slope * numpy.abs(focallength_nf) * numpy.sin(average_incident_angle / 1e3))
        elif input_parameters.ghy_calcType == 4:
            imagesize = max(imagesize,
                            8 * rms_slope * numpy.abs(focallength_nf) * (numpy.sin(average_incident_angle / 1e3) + numpy.sin(average_reflection_angle / 1e3)))

        # TODO: this is a patch: to be rewritten
        if shadow_oe._oe.F_MOVE==1 and not shadow_oe._oe.Y_ROT==0:
            imagesize = max(imagesize, 8*(input_parameters.ghy_distance*numpy.tan(numpy.radians(numpy.abs(shadow_oe._oe.Y_ROT))) + numpy.abs(shadow_oe._oe.OFFX)))

        imagenpts = int(round(imagesize / propagated_wavefront.delta() / 2) * 2 + 1)

        print("dif_x: begin calculation")

        dif_x = ScaledArray.initialize_from_range(numpy.ones(imagenpts),
                                                  -(imagenpts - 1) / 2 * propagated_wavefront.delta(),
                                                  (imagenpts - 1) / 2 * propagated_wavefront.delta())

        dif_x.np_array *= numpy.absolute(propagated_wavefront.get_interpolated_complex_amplitudes(dif_x.scale))**2

        calculation_parameters.dif_x = dif_x

##########################################################################
# 1D PROPAGATION ALGORITHM - Z DIRECTION
##########################################################################

def propagate_1D_z_direction(calculation_parameters, input_parameters):

    scale_factor = 1.0

    shadow_oe = calculation_parameters.shadow_oe_end

    global_phase_shift_profile = None

    if shadow_oe._oe.F_MOVE == 1 and shadow_oe._oe.X_ROT != 0.0:
        if input_parameters.ghy_calcType == 3 or input_parameters.ghy_calcType == 4:
            global_phase_shift_profile = calculation_parameters.w_mirror_lz
        elif input_parameters.ghy_calcType == 2:
            global_phase_shift_profile = ScaledArray.initialize_from_range(numpy.zeros(3), shadow_oe._oe.RLEN2, shadow_oe._oe.RLEN1)

        global_phase_shift_profile.set_values(global_phase_shift_profile.get_values() +
                                              global_phase_shift_profile.get_abscissas()*numpy.sin(numpy.radians(-shadow_oe._oe.X_ROT)))
    elif input_parameters.ghy_calcType == 3 or input_parameters.ghy_calcType == 4:
        global_phase_shift_profile = calculation_parameters.w_mirror_lz

    if input_parameters.ghy_calcType == 3 or input_parameters.ghy_calcType == 4:
        rms_slope = hy_findrmsslopefromheight(global_phase_shift_profile)

        print("Using RMS slope = " + str(rms_slope))

        if calculation_parameters.beam_not_cut_in_z:
            dp_image = numpy.std(calculation_parameters.zz_focal_ray)/input_parameters.ghy_focallength
            dp_se = 2 * rms_slope # different in x and z
            dp_error = calculation_parameters.gwavelength/2/(calculation_parameters.ghy_z_max-calculation_parameters.ghy_z_min)

            scale_factor = max(1, 5*min(dp_error/dp_image, dp_error/dp_se))

    # ------------------------------------------
    # far field calculation
    # ------------------------------------------
    if calculation_parameters.do_ff_z:
        focallength_ff = calculate_focal_length_ff(calculation_parameters.ghy_z_min,
                                                   calculation_parameters.ghy_z_max,
                                                   input_parameters.ghy_npeak,
                                                   calculation_parameters.gwavelength)
        
        if (input_parameters.ghy_calcType == 3 or input_parameters.ghy_calcType == 4) and rms_slope != 0:
            focallength_ff = min(focallength_ff, (calculation_parameters.ghy_z_max-calculation_parameters.ghy_z_min) / 16 / rms_slope ) #xshi changed
        elif input_parameters.ghy_calcType == 2 and not global_phase_shift_profile is None:
            focallength_ff = min(focallength_ff, input_parameters.ghy_distance*4) #TODO: PATCH to be found with a formula
        
        # focallength_ff = 25000
        fftsize = int(scale_factor * calculate_fft_size(calculation_parameters.ghy_z_min,
                                                        calculation_parameters.ghy_z_max,
                                                        calculation_parameters.gwavelength,
                                                        focallength_ff,
                                                        input_parameters.ghy_fftnpts))

        print("FF: creating plane wave begin, fftsize = " +  str(fftsize))

        wavefront = Wavefront1D.initialize_wavefront_from_range(wavelength=calculation_parameters.gwavelength,
                                                                number_of_points=fftsize,
                                                                x_min=scale_factor * calculation_parameters.ghy_z_min,
                                                                x_max=scale_factor * calculation_parameters.ghy_z_max)

        if scale_factor == 1.0:
            try:
                wavefront.set_plane_wave_from_complex_amplitude(numpy.sqrt(calculation_parameters.wIray_z.interpolate_values(wavefront.get_abscissas())))
            except IndexError:
                raise Exception("Unexpected Error during interpolation: try reduce Number of bins for I(Tangential) histogram")

        wavefront.apply_ideal_lens(focallength_ff)

        if input_parameters.ghy_calcType == 3 or \
                (input_parameters.ghy_calcType == 2 and not global_phase_shift_profile is None):
           wavefront.add_phase_shifts(get_mirror_phase_shift(wavefront.get_abscissas(),
                                                             calculation_parameters.gwavelength,
                                                             calculation_parameters.wangle_z,
                                                             calculation_parameters.wl_z,
                                                             global_phase_shift_profile))
        elif input_parameters.ghy_calcType == 4:
           wavefront.add_phase_shifts(get_grating_phase_shift(wavefront.get_abscissas(),
                                                              calculation_parameters.gwavelength,
                                                              calculation_parameters.wangle_z,
                                                              calculation_parameters.wangle_ref_z,
                                                              calculation_parameters.wl_z,
                                                              global_phase_shift_profile))

        print("calculated plane wave: begin FF propagation (distance = " +  str(focallength_ff) + ")")

        propagated_wavefront = propagator.propagate_1D_fresnel(wavefront, focallength_ff)

        print("dif_zp: begin calculation")

        imagesize = min(abs(calculation_parameters.ghy_z_max), abs(calculation_parameters.ghy_z_min)) * 2
        # 2017-01 Luca Rebuffi
        imagesize = min(imagesize,
                        input_parameters.ghy_npeak*2*0.88*calculation_parameters.gwavelength*focallength_ff/abs(calculation_parameters.ghy_z_max-calculation_parameters.ghy_z_min))

        # TODO: this is a patch: to be rewritten
        if shadow_oe._oe.F_MOVE==1 and not shadow_oe._oe.X_ROT==0:
            imagesize = max(imagesize, 8*(focallength_ff*numpy.tan(numpy.radians(numpy.abs(shadow_oe._oe.X_ROT))) + numpy.abs(shadow_oe._oe.OFFZ)))

        imagenpts = int(round(imagesize / propagated_wavefront.delta() / 2) * 2 + 1)

        dif_zp = ScaledArray.initialize_from_range(numpy.ones(propagated_wavefront.size()),
                                                   -(imagenpts - 1) / 2 * propagated_wavefront.delta(),
                                                   (imagenpts - 1) / 2 * propagated_wavefront.delta())

        dif_zp.np_array *= numpy.absolute(propagated_wavefront.get_interpolated_complex_amplitudes(dif_zp.scale))**2

        dif_zp.set_scale_from_range(-(imagenpts - 1) / 2 * propagated_wavefront.delta() / focallength_ff,
                                    (imagenpts - 1) / 2 * propagated_wavefront.delta() / focallength_ff)

        calculation_parameters.dif_zp = dif_zp


    # ------------------------------------------
    # near field calculation
    # ------------------------------------------
    if input_parameters.ghy_nf >= 1 and input_parameters.ghy_calcType > 1:
        focallength_nf = input_parameters.ghy_focallength

        fftsize = int(scale_factor * calculate_fft_size(calculation_parameters.ghy_z_min,
                                                        calculation_parameters.ghy_z_max,
                                                        calculation_parameters.gwavelength,
                                                        numpy.abs(focallength_nf),
                                                        input_parameters.ghy_fftnpts))

        print("NF: creating plane wave begin, fftsize = " +  str(fftsize))

        wavefront = Wavefront1D.initialize_wavefront_from_range(wavelength=calculation_parameters.gwavelength,
                                                                number_of_points=fftsize,
                                                                x_min=scale_factor*calculation_parameters.ghy_z_min,
                                                                x_max=scale_factor*calculation_parameters.ghy_z_max)

        if scale_factor == 1.0:
            try:
                wavefront.set_plane_wave_from_complex_amplitude(numpy.sqrt(calculation_parameters.wIray_z.interpolate_values(wavefront.get_abscissas())))
            except IndexError:
                raise Exception("Unexpected Error during interpolation: try reduce Number of bins for I(Tangential) histogram")

        wavefront.apply_ideal_lens(focallength_nf)

        if input_parameters.ghy_calcType == 3 or \
                (input_parameters.ghy_calcType == 2 and not global_phase_shift_profile is None):
           wavefront.add_phase_shifts(get_mirror_phase_shift(wavefront.get_abscissas(),
                                                             calculation_parameters.gwavelength,
                                                             calculation_parameters.wangle_z,
                                                             calculation_parameters.wl_z,
                                                             global_phase_shift_profile))
        elif input_parameters.ghy_calcType == 4:
           wavefront.add_phase_shifts(get_grating_phase_shift(wavefront.get_abscissas(),
                                                              calculation_parameters.gwavelength,
                                                              calculation_parameters.wangle_z,
                                                              calculation_parameters.wangle_ref_z,
                                                              calculation_parameters.wl_z,
                                                              global_phase_shift_profile))

        print("calculated plane wave: begin NF propagation (distance = " + str(input_parameters.ghy_distance) + ")")

        propagated_wavefront = propagator.propagate_1D_fresnel(wavefront, input_parameters.ghy_distance)

        # ghy_npeak in the wavefront propagation image
        imagesize = (input_parameters.ghy_npeak * 2 * 0.88 * calculation_parameters.gwavelength * numpy.abs(focallength_nf) / abs(calculation_parameters.ghy_z_max - calculation_parameters.ghy_z_min))
        imagesize = max(imagesize,
                        2 * abs((calculation_parameters.ghy_z_max - calculation_parameters.ghy_z_min) * (input_parameters.ghy_distance - numpy.abs(focallength_nf))) / numpy.abs(focallength_nf))

        if input_parameters.ghy_calcType == 3 or input_parameters.ghy_calcType == 4:
            imagesize = max(imagesize, 16 * rms_slope * numpy.abs(focallength_nf))

        # TODO: this is a patch: to be rewritten
        if shadow_oe._oe.F_MOVE==1 and not shadow_oe._oe.X_ROT==0:
            imagesize = max(imagesize, 8*(input_parameters.ghy_distance*numpy.tan(numpy.radians(numpy.abs(shadow_oe._oe.X_ROT))) + numpy.abs(shadow_oe._oe.OFFZ)))

        imagenpts = int(round(imagesize / propagated_wavefront.delta() / 2) * 2 + 1)

        print("dif_z: begin calculation")

        dif_z = ScaledArray.initialize_from_range(numpy.ones(imagenpts),
                                                  -(imagenpts - 1) / 2 * propagated_wavefront.delta(),
                                                   (imagenpts - 1) / 2 * propagated_wavefront.delta())

        dif_z.np_array *= numpy.absolute(propagated_wavefront.get_interpolated_complex_amplitudes(dif_z.scale)**2)

        calculation_parameters.dif_z = dif_z


##########################################################################
# 2D PROPAGATION ALGORITHM
##########################################################################

def propagate_2D(calculation_parameters, input_parameters):
    shadow_oe = calculation_parameters.shadow_oe_end

    if calculation_parameters.do_ff_z and calculation_parameters.do_ff_x:
        global_phase_shift_profile = None

        if shadow_oe._oe.F_MOVE == 1 and shadow_oe._oe.X_ROT != 0.0:
            if input_parameters.ghy_calcType == 3 or input_parameters.ghy_calcType == 4:
                global_phase_shift_profile = calculation_parameters.w_mirr_2D_values
            elif input_parameters.ghy_calcType == 2:
                global_phase_shift_profile = ScaledMatrix.initialize_from_range(numpy.zeros((3, 3)),
                                                                                shadow_oe._oe.RWIDX2, shadow_oe._oe.RWIDX1,
                                                                                shadow_oe._oe.RLEN2,  shadow_oe._oe.RLEN1)

            for x_index in range(global_phase_shift_profile.size_x()):
                global_phase_shift_profile.z_values[x_index, :] += global_phase_shift_profile.get_y_values()*numpy.sin(numpy.radians(-shadow_oe._oe.X_ROT))
        elif input_parameters.ghy_calcType == 3 or input_parameters.ghy_calcType == 4:
            global_phase_shift_profile = calculation_parameters.w_mirr_2D_values

        # only tangential slopes
        if input_parameters.ghy_calcType == 3 or input_parameters.ghy_calcType == 4:
            rms_slope = hy_findrmsslopefromheight(ScaledArray(np_array=global_phase_shift_profile.z_values[int(global_phase_shift_profile.size_x()/2), :],
                                                              scale=global_phase_shift_profile.get_y_values()))

            print("Using RMS slope = " + str(rms_slope))

        # ------------------------------------------
        # far field calculation
        # ------------------------------------------
        focallength_ff = calculate_focal_length_ff_2D(calculation_parameters.ghy_x_min,
                                                      calculation_parameters.ghy_x_max,
                                                      calculation_parameters.ghy_z_min,
                                                      calculation_parameters.ghy_z_max,
                                                      input_parameters.ghy_npeak,
                                                      calculation_parameters.gwavelength)

        if (input_parameters.ghy_calcType == 3 or input_parameters.ghy_calcType == 4) and rms_slope != 0:
            focallength_ff = min(focallength_ff,(calculation_parameters.ghy_z_max-calculation_parameters.ghy_z_min) / 16 / rms_slope ) #xshi changed
        elif input_parameters.ghy_calcType == 2 and not global_phase_shift_profile is None:
            focallength_ff = min(focallength_ff, input_parameters.ghy_distance*4) #TODO: PATCH to be found with a formula

        print("FF: calculated focal length: " + str(focallength_ff))

        fftsize_x = int(calculate_fft_size(calculation_parameters.ghy_x_min,
                                           calculation_parameters.ghy_x_max,
                                           calculation_parameters.gwavelength,
                                           focallength_ff,
                                           input_parameters.ghy_fftnpts,
                                           factor=20))

        fftsize_z = int(calculate_fft_size(calculation_parameters.ghy_z_min,
                                        calculation_parameters.ghy_z_max,
                                        calculation_parameters.gwavelength,
                                        focallength_ff,
                                        input_parameters.ghy_fftnpts,
                                        factor=20))

        print("FF: creating plane wave begin, fftsize_x = " +  str(fftsize_x) + ", fftsize_z = " +  str(fftsize_z))

        wavefront = Wavefront2D.initialize_wavefront_from_range(wavelength=calculation_parameters.gwavelength,
                                                                number_of_points=(fftsize_x, fftsize_z),
                                                                x_min=calculation_parameters.ghy_x_min,
                                                                x_max=calculation_parameters.ghy_x_max,
                                                                y_min=calculation_parameters.ghy_z_min,
                                                                y_max=calculation_parameters.ghy_z_max)

        try:
            for i in range(0, len(wavefront.electric_field_array.x_coord)):
                for j in range(0, len(wavefront.electric_field_array.y_coord)):
                    interpolated = calculation_parameters.wIray_2d.interpolate_value(wavefront.electric_field_array.x_coord[i],
                                                                                     wavefront.electric_field_array.y_coord[j])
                    wavefront.electric_field_array.set_z_value(i, j, numpy.sqrt(0.0 if interpolated < 0 else interpolated))
        except IndexError:
            raise Exception("Unexpected Error during interpolation: try reduce Number of bins for I(Tangential) histogram")

        wavefront.apply_ideal_lens(focallength_ff, focallength_ff)

        shadow_oe = calculation_parameters.shadow_oe_end

        if input_parameters.ghy_calcType == 3 or \
                (input_parameters.ghy_calcType == 2 and not global_phase_shift_profile is None):
            print("FF: calculating phase shift due to Height Error Profile")

            phase_shifts = numpy.zeros(wavefront.size())

            for index in range(0, phase_shifts.shape[0]):
                np_array = numpy.zeros(global_phase_shift_profile.shape()[1])
                for j in range(0, len(np_array)):
                    np_array[j] = global_phase_shift_profile.interpolate_value(wavefront.get_coordinate_x()[index], calculation_parameters.w_mirr_2D_values.get_y_value(j))

                global_phase_shift_profile_z = ScaledArray.initialize_from_steps(np_array,
                                                                                 global_phase_shift_profile.y_coord[0],
                                                                                 global_phase_shift_profile.y_coord[1] - global_phase_shift_profile.y_coord[0])

                phase_shifts[index, :] = get_mirror_phase_shift(wavefront.get_coordinate_y(),
                                                                calculation_parameters.gwavelength,
                                                                calculation_parameters.wangle_z,
                                                                calculation_parameters.wl_z,
                                                                global_phase_shift_profile_z)
            wavefront.add_phase_shifts(phase_shifts)
        elif input_parameters.ghy_calcType == 4:
            print("FF: calculating phase shift due to Height Error Profile")

            phase_shifts = numpy.zeros(wavefront.size())

            for index in range(0, phase_shifts.shape[0]):
                global_phase_shift_profile_z = ScaledArray.initialize_from_steps(global_phase_shift_profile.z_values[index, :],
                                                                                 global_phase_shift_profile.y_coord[0],
                                                                                 global_phase_shift_profile.y_coord[1] - global_phase_shift_profile.y_coord[0])

                phase_shifts[index, :] = get_grating_phase_shift(wavefront.get_coordinate_y(),
                                                                 calculation_parameters.gwavelength,
                                                                 calculation_parameters.wangle_z,
                                                                 calculation_parameters.wangle_ref_z,
                                                                 calculation_parameters.wl_z,
                                                                 global_phase_shift_profile_z)
            wavefront.add_phase_shifts(phase_shifts)
        elif input_parameters.ghy_calcType == 6:
            for w_mirr_2D_values in calculation_parameters.w_mirr_2D_values:
                phase_shift = get_crl_phase_shift(w_mirr_2D_values,
                                                  input_parameters,
                                                  calculation_parameters,
                                                  [wavefront.get_coordinate_x(), wavefront.get_coordinate_y()])

                wavefront.add_phase_shift(phase_shift)

        print("calculated plane wave: begin FF propagation (distance = " +  str(focallength_ff) + ")")

        propagated_wavefront = propagator2D.propagate_2D_fresnel(wavefront, focallength_ff)

        print("dif_zp: begin calculation")

        imagesize_x = min(abs(calculation_parameters.ghy_x_max), abs(calculation_parameters.ghy_x_min)) * 2
        imagesize_x = min(imagesize_x,
                          input_parameters.ghy_npeak*2*0.88*calculation_parameters.gwavelength*focallength_ff/abs(calculation_parameters.ghy_x_max-calculation_parameters.ghy_x_min))

        # TODO: this is a patch: to be rewritten
        if shadow_oe._oe.F_MOVE==1 and not shadow_oe._oe.Y_ROT==0:
            imagesize_x = max(imagesize_x, 8*(focallength_ff*numpy.tan(numpy.radians(numpy.abs(shadow_oe._oe.Y_ROT))) + numpy.abs(shadow_oe._oe.OFFX)))

        delta_x = propagated_wavefront.delta()[0]
        imagenpts_x = int(round(imagesize_x/delta_x/2) * 2 + 1)

        imagesize_z = min(abs(calculation_parameters.ghy_z_max), abs(calculation_parameters.ghy_z_min)) * 2
        imagesize_z = min(imagesize_z,
                          input_parameters.ghy_npeak*2*0.88*calculation_parameters.gwavelength*focallength_ff/abs(calculation_parameters.ghy_z_max-calculation_parameters.ghy_z_min))

        # TODO: this is a patch: to be rewritten
        if shadow_oe._oe.F_MOVE==1 and not shadow_oe._oe.X_ROT==0:
            imagesize_z = max(imagesize_z, 8*(focallength_ff*numpy.tan(numpy.radians(numpy.abs(shadow_oe._oe.X_ROT))) + numpy.abs(shadow_oe._oe.OFFZ)))

        delta_z = propagated_wavefront.delta()[1]
        imagenpts_z = int(round(imagesize_z/delta_z/2) * 2 + 1)

        dif_xpzp = ScaledMatrix.initialize_from_range(numpy.ones((imagenpts_x, imagenpts_z)),
                                                      min_scale_value_x = -(imagenpts_x - 1) / 2 * delta_x,
                                                      max_scale_value_x =(imagenpts_x - 1) / 2 * delta_x,
                                                      min_scale_value_y = -(imagenpts_z - 1) / 2 * delta_z,
                                                      max_scale_value_y =(imagenpts_z - 1) / 2 * delta_z)

        for i in range(0, dif_xpzp.shape()[0]):
            for j in range(0, dif_xpzp.shape()[1]):
                dif_xpzp.set_z_value(i, j, numpy.absolute(propagated_wavefront.get_interpolated_complex_amplitude(
                                                               dif_xpzp.x_coord[i],
                                                               dif_xpzp.y_coord[j]))**2
                                                           )

        dif_xpzp.set_scale_from_range(0,
                                      -(imagenpts_x - 1) / 2 * delta_x / focallength_ff,
                                      (imagenpts_x - 1) / 2 * delta_x / focallength_ff)

        dif_xpzp.set_scale_from_range(1,
                                      -(imagenpts_z - 1) / 2 * delta_z / focallength_ff,
                                      (imagenpts_z - 1) / 2 * delta_z / focallength_ff)

        calculation_parameters.dif_xpzp = dif_xpzp

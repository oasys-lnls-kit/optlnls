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

Created on Fri May  6 09:20:01 2022

@author: joao.astolfo
"""
import numpy as np
import copy
import os

#from hybrid_funcs.propagate import propagate_1D_x_direction, propagate_1D_z_direction, propagate_2D
#from hybrid_funcs.utility import get_delta, read_shadow_beam, sh_readsh, sh_readangle, sh_readsurface, h5_readsurface

from optlnls.hybrid_funcs.propagate import *
from optlnls.hybrid_funcs.utility import *

from orangecontrib.shadow.widgets.special_elements.bl.hybrid_control import HybridCalculationParameters, HybridNotNecessaryWarning
from orangecontrib.shadow.util.shadow_util import ShadowCongruence, ShadowPhysics
from orangecontrib.shadow.util.shadow_objects import ShadowBeam, ShadowOpticalElement

from srxraylib.util.data_structures import ScaledArray, ScaledMatrix
from srxraylib.util.inverse_method_sampler import Sampler2D, Sampler1D

class HybridInputParameters(object):
    shadow_beam = None
    original_shadow_beam = None

    ghy_n_oe = -1
    ghy_n_screen = -1

    ghy_diff_plane = 2
    ghy_calcType = 1

    ghy_focallength = -1
    ghy_distance = -1

    ghy_mirrorfile = "mirror.dat"

    ghy_nf = 1

    ghy_nbins_x = 200
    ghy_nbins_z = 200
    ghy_npeak = 20
    ghy_fftnpts = 1e6
    ghy_lengthunit = 1

    file_to_write_out = 0

    ghy_automatic = 1

    crl_error_profiles = None
    crl_material = None
    crl_delta = None
    crl_scaling_factor = 1.0

    random_seed = None

    def __init__(self):
        super().__init__()

    def dump(self):
        return self.__dict__


def run_hybrid(beam, units=2, diff_plane=1, calcType=2, dist_to_img_calc=0, distance=25000,
               focal_length_calc=0, focallength=25000, nf=0, nbins_x=100, nbins_z=100, npeak=10, fftnpts=1e5,
               write_file=0, automatic=1, send_original_beam=False):
    """

    Parameters
    ----------
    beam : ShadowBeam()
        Input ShadowBeam.
    units : int, optional
        OASYS units.\n
        0: m\n
        1: cm\n
        2: mm\n
        The default is 2.
    diff_plane : int, optional
        Diffraction plane.\n
        0: Sagittal\n
        1: Tangential\n
        2: Both (2D)\n
        3: Both (1D + 1D)\n
        The default is 1.
    calcType : int, optional
        Calculation type.\n
        0: Diffraction by Simple Aperture\n
        1: Diffraction by Mirror or Grating Size\n
        2: Diffraction by Mirror Size + Figure Errors\n
        3: Diffraction by Grating Size + Figure Errors\n
        4: Diffraction by Lens/C.R.L./Transf. Size\n
        5: Diffraction by Lens/C.R.L./Transf. Size + Thickness Errors\n
        The default is 2.
    dist_to_img_calc : int, optional
        Distance to image calculation method.\n
        0: Use O.E. Image Plane Distance\n
        1: Specify Value\n
        The default is 0.
    distance : float, optional
        Distance to image value [mm]. The default is 25000.0.
    focal_length_calc : int, optional
        Focal length calculation method.\n
        0: Use O.E. Focal Distance\n
        1: Specify Value\n
        The default is 0.
    focallength : float, optional
        Focal length used on calculation. The default is 25000.0.
    nf : int, optional
        Near field calculation. The default is 0.
    nbins_x : int, optional
        Number of bins on x axis. The default is 100.
    nbins_z : int, optional
        Number of bins on z axis. The default is 100.
    npeak : int, optional
        Number of diffraction peaks. The default is 10.
    fftnpts : int, optional
        Number of points for FFT. The default is 1e5.
    write_file : int, optional
        Files to write out.\n
        0: None\n
        1: Debug (star.xx)\n
        The default is 0.
    automatic : int, optional
        Analize geometry to avoid unuseful calculation. The default is 1.
    send_original_beam : bool, optional
        If hybrid fails, return original beam. The default is False.

    Returns
    -------
    None
        Just calls other functions for hybrid screen calculations.

    """
    try:
        if ShadowCongruence.checkEmptyBeam(beam):
            if ShadowCongruence.checkGoodBeam(beam):
                
                input_parameters = HybridInputParameters()
                
                input_parameters.ghy_lengthunit = units
                input_parameters.shadow_beam = beam
                input_parameters.ghy_diff_plane = diff_plane + 1
                input_parameters.ghy_calcType = calcType + 1
            
                if dist_to_img_calc == 0:
                    input_parameters.ghy_distance = -1
                else:
                    input_parameters.ghy_distance = distance
            
                if focal_length_calc == 0:
                    input_parameters.ghy_focallength = -1
                else:
                    input_parameters.ghy_focallength = focallength
            
                if calcType != 0:
                    input_parameters.ghy_nf = nf
                else:
                    input_parameters.ghy_nf = 0
            
                input_parameters.ghy_nbins_x = int(nbins_x)
                input_parameters.ghy_nbins_z = int(nbins_z)
                input_parameters.ghy_npeak = int(npeak)
                input_parameters.ghy_fftnpts = int(fftnpts)
                input_parameters.file_to_write_out = write_file
            
                input_parameters.ghy_automatic = automatic
            
                try:
                    calculation_parameters = hy_run(input_parameters,write_file)
                    
                    distance = input_parameters.ghy_distance
                    nbins_x = int(input_parameters.ghy_nbins_x)
                    nbins_z = int(input_parameters.ghy_nbins_z)
                    npeak   = int(input_parameters.ghy_npeak)
                    fftnpts = int(input_parameters.ghy_fftnpts)

                    if not calculation_parameters.ff_beam is None:
                        calculation_parameters.ff_beam.setScanningData(beam.scanned_variable_data)

                    return calculation_parameters.ff_beam

                    do_nf = input_parameters.ghy_nf == 1 and input_parameters.ghy_calcType > 1

                    if do_nf and not calculation_parameters.nf_beam is None:
                        calculation_parameters.nf_beam.setScanningData(beam.scanned_variable_data)

                        return calculation_parameters.nf_beam
                    
                except Exception as e:
                    if send_original_beam==1:
                        return beam.duplicate(history=True)
                    
                    else:
                        raise e
            else:
                raise Exception("Input Beam with no good rays")
        else:
            raise Exception("Empty Input Beam")
    except:
        raise Exception('Hybrid Screen calculation failed')


def hy_run(input_parameters=HybridInputParameters(),write_file=0):
    calculation_parameters = HybridCalculationParameters()

    try:
        input_parameters.original_shadow_beam = input_parameters.shadow_beam.duplicate(history=True)
        
        hy_check_congruence(input_parameters,calculation_parameters,write_file)
        
        hy_readfiles(input_parameters, calculation_parameters,write_file)	#Read shadow output files needed by HYBRID
        
        if input_parameters.ghy_diff_plane == 4:
            # FIRST: X DIRECTION
            input_parameters.ghy_diff_plane = 1

            hy_init(input_parameters, calculation_parameters)		#Calculate functions needed to construct exit pupil function

            hy_prop(input_parameters, calculation_parameters)	    #Perform wavefront propagation

            hy_conv(input_parameters, calculation_parameters)	    #Perform ray resampling

            # SECOND: Z DIRECTION
            input_parameters.ghy_diff_plane = 2

            hy_init(input_parameters, calculation_parameters)		#Calculate functions needed to construct exit pupil function

            hy_prop(input_parameters, calculation_parameters)	    #Perform wavefront propagation

            hy_conv(input_parameters, calculation_parameters)	    #Perform ray resampling

            input_parameters.ghy_diff_plane = 3

            hy_create_shadow_beam(input_parameters, calculation_parameters)

        else:
            hy_init(input_parameters, calculation_parameters)		#Calculate functions needed to construct exit pupil function

            hy_prop(input_parameters, calculation_parameters)	    #Perform wavefront propagation

            hy_conv(input_parameters, calculation_parameters)	    #Perform ray resampling

            hy_create_shadow_beam(input_parameters, calculation_parameters)

    except HybridNotNecessaryWarning as warning:
        try:
            print("Error")
        except:
            print(str(warning))

    except Exception as exception:
        raise exception

    return calculation_parameters


def hy_check_congruence(input_parameters=HybridInputParameters(),
                        calculation_parameters=HybridCalculationParameters(),
                        write_file=0):
    if input_parameters.ghy_n_oe < 0 and input_parameters.shadow_beam._oe_number == 0: # TODO!!!!!
        raise Exception("Source calculation not yet supported")

    beam_after = input_parameters.shadow_beam
    history_entry =  beam_after.getOEHistory(beam_after._oe_number)

    widget_class_name = history_entry._widget_class_name

    if input_parameters.ghy_calcType == 1 and not "ScreenSlits" in widget_class_name:
        raise Exception("Simple Aperture calculation runs for Screen-Slits widgets only")

    if input_parameters.ghy_calcType == 2:
        if not ("Mirror" in widget_class_name or "Grating" in widget_class_name):
            raise Exception("Mirror/Grating calculation runs for Mirror/Grating widgets only")

    if input_parameters.ghy_calcType == 3:
        if not ("Mirror" in widget_class_name):
            raise Exception("Mirror calculation runs for Mirror widgets only")

    if input_parameters.ghy_calcType == 4:
        if not ("Grating" in widget_class_name):
            raise Exception("Grating calculation runs for Gratings widgets only")

    if input_parameters.ghy_calcType in [5, 6]:
        if not ("Lens" in widget_class_name or "CRL" in widget_class_name or "Transfocator"):
            raise Exception("CRL calculation runs for Lens, CRLs or Transfocators widgets only")

    if input_parameters.ghy_n_oe < 0:
        beam_before = history_entry._input_beam.duplicate()
        oe_before = history_entry._shadow_oe_start.duplicate()

        number_of_good_rays_before =  len(beam_before._beam.rays[np.where(beam_before._beam.rays[:, 9] == 1)])
        number_of_good_rays_after = len(beam_after._beam.rays[np.where(beam_after._beam.rays[:, 9] == 1)])

        if number_of_good_rays_before == number_of_good_rays_after:
            calculation_parameters.beam_not_cut_in_x = True
            calculation_parameters.beam_not_cut_in_z = True

            if (not input_parameters.ghy_calcType in [3, 4, 6]) and input_parameters.ghy_automatic == 1:
                calculation_parameters.ff_beam = input_parameters.shadow_beam

                raise HybridNotNecessaryWarning("O.E. contains the whole beam, diffraction effects are not expected:\nCalculation aborted, beam remains unaltered")
        else:
            # displacements analysis
            if input_parameters.ghy_calcType < 5 and oe_before._oe.F_MOVE==1:
                if input_parameters.ghy_calcType == 2 or input_parameters.ghy_calcType == 3 or input_parameters.ghy_calcType == 4:
                    if input_parameters.ghy_diff_plane == 1: #X
                        if oe_before._oe.X_ROT != 0.0 or oe_before._oe.Z_ROT != 0.0:
                            raise Exception("Only rotations around the Y axis are supported for sagittal diffraction plane")
                    elif input_parameters.ghy_diff_plane == 2 or input_parameters.ghy_diff_plane == 3: #Z
                        if oe_before._oe.Y_ROT != 0.0 or oe_before._oe.Z_ROT != 0.0:
                            raise Exception("Only rotations around the X axis are supported for tangential or Both (2D) diffraction planes")
                    elif input_parameters.ghy_diff_plane == 4: #Z
                        if oe_before._oe.Z_ROT != 0.0:
                            raise Exception("Only rotations around the X and Y axis are supported for Both (1D+1D) diffraction planes")
                else:
                    raise Exception("O.E. Movements are not supported for this kind of calculation")

            ticket_tangential = None
            ticket_sagittal = None
            max_tangential = np.Inf
            min_tangential = -np.Inf
            max_sagittal = np.Inf
            min_sagittal = -np.Inf
            is_infinite = False

            # CASE SIMPLE APERTURE:
            if input_parameters.ghy_calcType == 1:
                if oe_before._oe.I_SLIT[0] == 0: # NOT APERTURE
                    is_infinite = True
                else:
                    if oe_before._oe.I_STOP[0] == 1: # OBSTRUCTION
                        raise Exception("Simple Aperture calculation runs for apertures only")

                    beam_at_the_slit = beam_before.duplicate(history=False)
                    beam_at_the_slit._beam.retrace(oe_before._oe.T_SOURCE) # TRACE INCIDENT BEAM UP TO THE SLIT

                    # TODO: MANAGE CASE OF ROTATED SLITS (OE MOVEMENT OR SOURCE MOVEMENT)
                    max_tangential = oe_before._oe.CZ_SLIT[0] + oe_before._oe.RZ_SLIT[0]/2
                    min_tangential = oe_before._oe.CZ_SLIT[0] - oe_before._oe.RZ_SLIT[0]/2
                    max_sagittal = oe_before._oe.CX_SLIT[0] + oe_before._oe.RX_SLIT[0]/2
                    min_sagittal = oe_before._oe.CX_SLIT[0] - oe_before._oe.RX_SLIT[0]/2

                    ticket_tangential = beam_at_the_slit._beam.histo1(3, nbins=500, nolost=1, ref=23)
                    ticket_sagittal = beam_at_the_slit._beam.histo1(1, nbins=500, nolost=1, ref=23)

            elif input_parameters.ghy_calcType in [2, 3, 4]: # MIRRORS/GRATINGS
                if oe_before._oe.FHIT_C == 0: #infinite
                    is_infinite = True
                else:
                    str_n_oe = str(input_parameters.shadow_beam._oe_number)
                    if input_parameters.shadow_beam._oe_number < 10:
                        str_n_oe = "0" + str_n_oe

                    beam_before._beam.rays = beam_before._beam.rays[np.where(beam_before._beam.rays[:, 9] == 1)] # GOOD ONLY BEFORE THE BEAM

                    oe_before._oe.FWRITE = 1
                    mirror_beam = ShadowBeam.traceFromOE(beam_before, oe_before, history=False)
                    mirror_beam.loadFromFile("mirr." + str_n_oe)
                    if not write_file:
                        os.remove("mirr."+str_n_oe)
                        os.remove("effic."+str_n_oe)
                        os.remove("optax.0"+str(int((str_n_oe))-1))
                        os.remove("optax."+str_n_oe)
                        os.remove("rmir."+str_n_oe)

                    max_tangential = oe_before._oe.RLEN1
                    min_tangential = oe_before._oe.RLEN2
                    max_sagittal = oe_before._oe.RWIDX1
                    min_sagittal = oe_before._oe.RWIDX2
                    ticket_tangential = mirror_beam._beam.histo1(2, nbins=500, nolost=0, ref=23) # ALL THE RAYS FOR ANALYSIS
                    ticket_sagittal = mirror_beam._beam.histo1(1, nbins=500, nolost=0, ref=23) # ALL THE RAYS  FOR ANALYSIS

            elif input_parameters.ghy_calcType in [5, 6]: # CRL/LENS/TRANSFOCATOR
                oes_list = history_entry._shadow_oe_end._oe.list

                beam_at_the_slit = beam_before.duplicate(history=False)
                beam_at_the_slit._beam.retrace(oes_list[0].T_SOURCE) # TRACE INCIDENT BEAM UP TO THE SLIT

                is_infinite = True
                max_tangential_list = []
                min_tangential_list = []
                max_sagittal_list = []
                min_sagittal_list = []
                for oe in oes_list:
                    if oe.FHIT_C == 1:
                        is_infinite = False

                        max_tangential_list.append(np.abs(oe.RLEN2))
                        min_tangential_list.append(-np.abs(oe.RLEN2))
                        max_sagittal_list.append(np.abs(oe.RWIDX2))
                        min_sagittal_list.append(-np.abs(oe.RWIDX2))

                if not is_infinite:
                    max_tangential = np.min(max_tangential_list)
                    min_tangential = np.max(min_tangential_list)
                    max_sagittal = np.min(max_sagittal_list)
                    min_sagittal = np.max(min_sagittal_list)

                ticket_tangential = beam_at_the_slit._beam.histo1(3, nbins=500, nolost=1, ref=23)
                ticket_sagittal = beam_at_the_slit._beam.histo1(1, nbins=500, nolost=1, ref=23)

            ############################################################################

            if is_infinite:
                calculation_parameters.beam_not_cut_in_x = True
                calculation_parameters.beam_not_cut_in_z = True
            else: # ANALYSIS OF THE HISTOGRAMS
                # SAGITTAL
                intensity_sagittal = ticket_sagittal['histogram']
                total_intensity_sagittal = np.sum(intensity_sagittal) # should be identical to total_intensity_tangential
                coordinate_sagittal = ticket_sagittal['bin_center']

                cursor_up = np.where(coordinate_sagittal < min_sagittal)
                cursor_down = np.where(coordinate_sagittal > max_sagittal)
                intensity_sagittal_cut = (np.sum(intensity_sagittal[cursor_up]) + np.sum(intensity_sagittal[cursor_down]))/total_intensity_sagittal

                # TANGENTIAL
                intensity_tangential = ticket_tangential['histogram']
                total_intensity_tangential = np.sum(intensity_tangential)
                coordinate_tangential = ticket_tangential['bin_center']

                cursor_up = np.where(coordinate_tangential < min_tangential)
                cursor_down = np.where(coordinate_tangential > max_tangential)
                intensity_tangential_cut = (np.sum(intensity_tangential[cursor_up]) + np.sum(intensity_tangential[cursor_down]))/total_intensity_tangential

                calculation_parameters.beam_not_cut_in_x = intensity_sagittal_cut < 0.05
                calculation_parameters.beam_not_cut_in_z = intensity_tangential_cut < 0.05

            # REQUEST FILTERING OR REFUSING

            if  not input_parameters.ghy_calcType in [3, 4, 6]: # no figure/thickness errors
                if input_parameters.ghy_automatic == 1:
                    if input_parameters.ghy_diff_plane == 1 and calculation_parameters.beam_not_cut_in_x :
                        calculation_parameters.ff_beam = input_parameters.original_shadow_beam

                        raise HybridNotNecessaryWarning("O.E. contains almost the whole beam, diffraction effects are not expected:\nCalculation aborted, beam remains unaltered")

                    if input_parameters.ghy_diff_plane == 2 and calculation_parameters.beam_not_cut_in_z:
                        calculation_parameters.ff_beam = input_parameters.original_shadow_beam

                        raise HybridNotNecessaryWarning("O.E. contains almost the whole beam, diffraction effects are not expected:\nCalculation aborted, beam remains unaltered")

                    if input_parameters.ghy_diff_plane == 3 or input_parameters.ghy_diff_plane == 4: # BOTH
                        if calculation_parameters.beam_not_cut_in_x and calculation_parameters.beam_not_cut_in_z:
                            calculation_parameters.ff_beam = input_parameters.original_shadow_beam

                            raise HybridNotNecessaryWarning("O.E. contains almost the whole beam, diffraction effects are not expected:\nCalculation aborted, beam remains unaltered")

                        if calculation_parameters.beam_not_cut_in_x:
                            input_parameters.ghy_diff_plane = 2

                            try:
                                print("O.E. does not cut the beam in the Sagittal plane:\nCalculation is done in Tangential plane only")
                            except:
                                print("O.E. does not cut the beam in the Sagittal plane:\nCalculation is done in Tangential plane only")

                        elif calculation_parameters.beam_not_cut_in_z:
                            input_parameters.ghy_diff_plane = 1

                            try:
                                print("O.E. does not cut the beam in the Tangential plane:\nCalculation is done in Sagittal plane only")
                            except:
                                print("O.E. does not cut the beam in the Tangential plane:\nCalculation is done in Sagittal plane only")


def hy_readfiles(input_parameters=HybridInputParameters(),
                 calculation_parameters=HybridCalculationParameters(),
                 write_file=0):
    if input_parameters.ghy_calcType in [5, 6]: #CRL OR LENS
        history_entry =  input_parameters.shadow_beam.getOEHistory(input_parameters.shadow_beam._oe_number)
        compound_oe = history_entry._shadow_oe_end

        for oe in compound_oe._oe.list:
            if oe.FHIT_C == 0: #infinite
                raise Exception("Calculation not possible: at least one lens have infinite diameter")

        last_oe = compound_oe._oe.list[-1]

        image_plane_distance = last_oe.T_IMAGE

        screen_slit = ShadowOpticalElement.create_screen_slit()

        screen_slit._oe.DUMMY = input_parameters.widget.workspace_units_to_cm # Issue #3 : Global User's Unit
        screen_slit._oe.T_SOURCE     = -image_plane_distance
        screen_slit._oe.T_IMAGE      = image_plane_distance
        screen_slit._oe.T_INCIDENCE  = 0.0
        screen_slit._oe.T_REFLECTION = 180.0
        screen_slit._oe.ALPHA        = 0.0

        n_screen = 1
        i_screen = np.zeros(10)  # after
        i_abs = np.zeros(10)
        i_slit = np.zeros(10)
        i_stop = np.zeros(10)
        k_slit = np.zeros(10)
        thick = np.zeros(10)
        file_abs = np.array(['', '', '', '', '', '', '', '', '', ''])
        rx_slit = np.zeros(10)
        rz_slit = np.zeros(10)
        sl_dis = np.zeros(10)
        file_scr_ext = np.array(['', '', '', '', '', '', '', '', '', ''])
        cx_slit = np.zeros(10)
        cz_slit = np.zeros(10)

        i_slit[0] = 1
        k_slit[0] = 1

        rx_slit[0] = np.abs(2*last_oe.RWIDX2)
        rz_slit[0] = np.abs(2*last_oe.RLEN2)

        screen_slit._oe.set_screens(n_screen,
                                  i_screen,
                                  i_abs,
                                  sl_dis,
                                  i_slit,
                                  i_stop,
                                  k_slit,
                                  thick,
                                  file_abs,
                                  rx_slit,
                                  rz_slit,
                                  cx_slit,
                                  cz_slit,
                                  file_scr_ext)

        input_parameters.shadow_beam = ShadowBeam.traceFromOE(input_parameters.shadow_beam, screen_slit)

    str_n_oe = str(input_parameters.shadow_beam._oe_number)
    if input_parameters.shadow_beam._oe_number < 10: str_n_oe = "0" + str_n_oe

    # Before ray-tracing save the original history:
    calculation_parameters.original_beam_history = input_parameters.shadow_beam.getOEHistory()

    history_entry =  input_parameters.shadow_beam.getOEHistory(input_parameters.shadow_beam._oe_number)

    shadow_oe = history_entry._shadow_oe_start.duplicate() # no changes to the original object!
    shadow_oe_input_beam = history_entry._input_beam.duplicate(history=False)
    
    if shadow_oe._oe.F_SCREEN == 1:
        if shadow_oe._oe.N_SCREEN == 10: raise Exception("Hybrid Screen has not been created: O.E. has already 10 screens")

        n_screen     = shadow_oe._oe.N_SCREEN + 1
        i_screen     = shadow_oe._oe.I_SCREEN
        sl_dis       = shadow_oe._oe.I_ABS
        i_abs        = shadow_oe._oe.SL_DIS
        i_slit       = shadow_oe._oe.I_SLIT
        i_stop       = shadow_oe._oe.I_STOP      
        k_slit       = shadow_oe._oe.K_SLIT      
        thick        = shadow_oe._oe.THICK       
        file_abs     = np.copy(shadow_oe._oe.FILE_ABS)
        rx_slit      = shadow_oe._oe.RX_SLIT     
        rz_slit      = shadow_oe._oe.RZ_SLIT     
        cx_slit      = shadow_oe._oe.CX_SLIT     
        cz_slit      = shadow_oe._oe.CZ_SLIT     
        file_scr_ext = np.copy(shadow_oe._oe.FILE_SCR_EXT)
        
        index = n_screen - 1
        
        i_screen[index] = 0 
        sl_dis[index] = 0       
        i_abs[index] = 0        
        i_slit[index] = 0       
        i_stop[index] = 0       
        k_slit[index] = 0       
        thick[index] = 0        
        rx_slit[index] = 0
        rz_slit[index] = 0      
        cx_slit[index] = 0      
        cz_slit[index] = 0
    else:
        n_screen = 1
        i_screen = np.zeros(10)  # after
        i_abs = np.zeros(10)
        i_slit = np.zeros(10)
        i_stop = np.zeros(10)
        k_slit = np.zeros(10)
        thick = np.zeros(10)
        file_abs = np.array(['', '', '', '', '', '', '', '', '', ''])
        rx_slit = np.zeros(10)
        rz_slit = np.zeros(10)
        sl_dis = np.zeros(10)
        file_scr_ext = np.array(['', '', '', '', '', '', '', '', '', ''])
        cx_slit = np.zeros(10)
        cz_slit = np.zeros(10)
        
        index = 0

    fileShadowScreen = "screen." + str_n_oe + ("0" + str(n_screen)) if n_screen < 10 else "10"

    if input_parameters.ghy_calcType in [1, 5, 6]: # simple aperture or CRLs
        if (shadow_oe._oe.FMIRR == 5 and \
            shadow_oe._oe.F_CRYSTAL == 0 and \
            shadow_oe._oe.F_REFRAC == 2 and \
            shadow_oe._oe.F_SCREEN==1 and \
            shadow_oe._oe.N_SCREEN==1):

            i_abs[index] = shadow_oe._oe.I_ABS[index]
            i_slit[index] = shadow_oe._oe.I_SLIT[index]

            if shadow_oe._oe.I_SLIT[index] == 1:
                i_stop[index] = shadow_oe._oe.I_STOP[index]
                k_slit[index] = shadow_oe._oe.K_SLIT[index]

                if shadow_oe._oe.K_SLIT[index] == 2:
                    file_scr_ext[index] = shadow_oe._oe.FILE_SCR_EXT[index]
                else:
                    rx_slit[index] = shadow_oe._oe.RX_SLIT[index]
                    rz_slit[index] = shadow_oe._oe.RZ_SLIT[index]
                    cx_slit[index] = shadow_oe._oe.CX_SLIT[index]
                    cz_slit[index] = shadow_oe._oe.CZ_SLIT[index]

            if shadow_oe._oe.I_ABS[index] == 1:
                thick[index] = shadow_oe._oe.THICK[index]
                file_abs[index] = shadow_oe._oe.FILE_ABS[index]
        else:
            raise Exception("Connected O.E. is not a Screen-Slit or CRL widget!")
    elif input_parameters.ghy_calcType == 2: # ADDED BY XIANBO SHI
        shadow_oe._oe.F_RIPPLE = 0
    elif input_parameters.ghy_calcType == 3 or input_parameters.ghy_calcType == 4: # mirror/grating + figure error
        if shadow_oe._oe.F_RIPPLE == 1 and shadow_oe._oe.F_G_S == 2:
            input_parameters.ghy_mirrorfile = shadow_oe._oe.FILE_RIP

            # disable slope error calculation for OE, must be done by HYBRID!
            shadow_oe._oe.F_RIPPLE = 0
        else:
            raise Exception("O.E. has not Surface Error file (setup Advanced Option->Modified Surface:\n\nModification Type = Surface Error\nType of Defect: external spline)")

    #TODO: check compatibility between hybrid calcualtion and angle of rotations (es. tangential -> rot X, sagittal -> rot Y)

    # tracing must be done without o.e. movements: hybrid is going to take care of that
    x_rot  = shadow_oe._oe.X_ROT
    y_rot  = shadow_oe._oe.Y_ROT
    z_rot  = shadow_oe._oe.Z_ROT

    shadow_oe._oe.X_ROT  = 0.0
    shadow_oe._oe.Y_ROT  = 0.0
    shadow_oe._oe.Z_ROT  = 0.0

    shadow_oe._oe.set_screens(n_screen,
                              i_screen,
                              i_abs,
                              sl_dis,
                              i_slit,
                              i_stop,
                              k_slit,
                              thick,
                              file_abs,
                              rx_slit,
                              rz_slit,
                              cx_slit,
                              cz_slit,
                              file_scr_ext)

    if input_parameters.ghy_calcType > 0: # THIS WAS RESPONSIBLE OF THE SERIOUS BUG AT SOS WORKSHOP!!!!!
        if shadow_oe._oe.FWRITE > 1 or shadow_oe._oe.F_ANGLE == 0:
            shadow_oe._oe.FWRITE = 0 # all
            shadow_oe._oe.F_ANGLE = 1 # angles

    # need to rerun simulation

    shadow_beam_at_image_plane = ShadowBeam.traceFromOE(shadow_oe_input_beam, shadow_oe, history=False)

    # restore o.e. setting for further calculations
    shadow_oe._oe.X_ROT  = x_rot
    shadow_oe._oe.Y_ROT  = y_rot
    shadow_oe._oe.Z_ROT  = z_rot

    input_parameters.shadow_beam = shadow_beam_at_image_plane

    image_beam, image_beam_lo = read_shadow_beam(shadow_beam_at_image_plane, lost=True) #xshi change from 0 to 1

    calculation_parameters.shadow_oe_end = shadow_oe

    if input_parameters.file_to_write_out == 1:
        image_beam.writeToFile("hybrid_beam_at_image_plane." + str_n_oe)

    calculation_parameters.image_plane_beam = image_beam
    calculation_parameters.image_plane_beam.set_initial_flux(input_parameters.original_shadow_beam.get_initial_flux())
    calculation_parameters.image_plane_beam_lost = image_beam_lo

    # read shadow screen file
    screen_beam= sh_readsh(fileShadowScreen)    #xshi change from 0 to 1
    if not write_file: os.remove(fileShadowScreen)

    if input_parameters.file_to_write_out == 1:
        screen_beam.writeToFile("hybrid_beam_at_oe_hybrid_screen." + str_n_oe)

    calculation_parameters.screen_plane_beam = screen_beam

    calculation_parameters.wenergy     = ShadowPhysics.getEnergyFromShadowK(screen_beam._beam.rays[:, 10])
    calculation_parameters.wwavelength = ShadowPhysics.getWavelengthFromShadowK(screen_beam._beam.rays[:, 10])
    calculation_parameters.xp_screen   = screen_beam._beam.rays[:, 3]
    calculation_parameters.yp_screen   = screen_beam._beam.rays[:, 4]
    calculation_parameters.zp_screen   = screen_beam._beam.rays[:, 5]

    #genergy = np.average(calculation_parameters.wenergy) # average photon energy in eV

    calculation_parameters.xx_screen = screen_beam._beam.rays[:, 0]
    calculation_parameters.ghy_x_min = np.min(calculation_parameters.xx_screen)
    calculation_parameters.ghy_x_max = np.max(calculation_parameters.xx_screen)

    calculation_parameters.zz_screen = screen_beam._beam.rays[:, 2]
    calculation_parameters.ghy_z_min = np.min(calculation_parameters.zz_screen)
    calculation_parameters.ghy_z_max = np.max(calculation_parameters.zz_screen)

    calculation_parameters.dx_ray = np.arctan(calculation_parameters.xp_screen/calculation_parameters.yp_screen) # calculate divergence from direction cosines from SHADOW file  dx = atan(v_x/v_y)
    calculation_parameters.dz_ray = np.arctan(calculation_parameters.zp_screen/calculation_parameters.yp_screen) # calculate divergence from direction cosines from SHADOW file  dz = atan(v_z/v_y)

    # Process mirror/grating
	# reads file with mirror height mesh
 	# calculates the function of the "incident angle" and the "mirror height" versus the Z coordinate in the screen.

    if input_parameters.ghy_calcType in [2, 3, 4]:
        mirror_beam = sh_readsh("mirr." + str_n_oe)  #xshi change from 0 to 1
        if not write_file:
            os.remove("mirr."+str_n_oe)
            os.remove("effic."+str_n_oe)
            os.remove("optax.0"+str(int((str_n_oe))-1))
            os.remove("optax."+str_n_oe)
            os.remove("rmir."+str_n_oe)
            os.remove("star."+str_n_oe)

        if input_parameters.file_to_write_out == 1:
            mirror_beam.writeToFile("hybrid_footprint_on_oe." + str_n_oe)

        calculation_parameters.xx_mirr = mirror_beam._beam.rays[:, 0]
        calculation_parameters.yy_mirr = mirror_beam._beam.rays[:, 1]
        calculation_parameters.zz_mirr = mirror_beam._beam.rays[:, 2]

        # read in angle files

        angle_inc, angle_ref = sh_readangle("angle." + str_n_oe, mirror_beam)   #xshi change from 0 to 1
        if not write_file: os.remove("angle."+str_n_oe)

        calculation_parameters.angle_inc = (90.0 - angle_inc)/180.0*1e3*np.pi
        calculation_parameters.angle_ref = (90.0 - angle_ref)/180.0*1e3*np.pi

        if not input_parameters.ghy_calcType == 2:
            calculation_parameters.w_mirr_2D_values = sh_readsurface(input_parameters.ghy_mirrorfile, dimension=2)

        # generate theta(z) and l(z) curve over a continuous grid

        hy_npoly_angle = 3
        hy_npoly_l = 6

        if np.amax(calculation_parameters.xx_screen) == np.amin(calculation_parameters.xx_screen):
            if input_parameters.ghy_diff_plane == 1 or input_parameters.ghy_diff_plane == 3: raise Exception("Unconsistend calculation: Diffraction plane is set on X, but the beam has no extention in that direction")
        else:
            calculation_parameters.wangle_x     = np.poly1d(np.polyfit(calculation_parameters.xx_screen, calculation_parameters.angle_inc, hy_npoly_angle))
            calculation_parameters.wl_x         = np.poly1d(np.polyfit(calculation_parameters.xx_screen, calculation_parameters.xx_mirr, hy_npoly_l))
            if input_parameters.ghy_calcType == 4: calculation_parameters.wangle_ref_x = np.poly1d(np.polyfit(calculation_parameters.xx_screen, calculation_parameters.angle_ref, hy_npoly_angle))

        if np.amax(calculation_parameters.zz_screen) == np.amin(calculation_parameters.zz_screen):
            if input_parameters.ghy_diff_plane == 2 or input_parameters.ghy_diff_plane == 3: raise Exception("Unconsistend calculation: Diffraction plane is set on Z, but the beam has no extention in that direction")
        else:
            calculation_parameters.wangle_z     = np.poly1d(np.polyfit(calculation_parameters.zz_screen, calculation_parameters.angle_inc, hy_npoly_angle))
            calculation_parameters.wl_z         = np.poly1d(np.polyfit(calculation_parameters.zz_screen, calculation_parameters.yy_mirr, hy_npoly_l))
            if input_parameters.ghy_calcType == 4: calculation_parameters.wangle_ref_z = np.poly1d(np.polyfit(calculation_parameters.zz_screen, calculation_parameters.angle_ref, hy_npoly_angle))
    elif input_parameters.ghy_calcType == 6:
        calculation_parameters.w_mirr_2D_values = [h5_readsurface(thickness_error_file) for thickness_error_file in input_parameters.crl_error_profiles]


def hy_init(input_parameters=HybridInputParameters(), calculation_parameters=HybridCalculationParameters()):
    oe_number = input_parameters.shadow_beam._oe_number

    if input_parameters.ghy_calcType > 1:
        simag = calculation_parameters.shadow_oe_end._oe.SIMAG

        if input_parameters.ghy_focallength < 0:
            input_parameters.ghy_focallength = simag
            print("Focal length not set (<-1), take from SIMAG" + str(input_parameters.ghy_focallength))

        if input_parameters.ghy_focallength != simag:
            print("Defined focal length is different from SIMAG, used the defined focal length = " + str(input_parameters.ghy_focallength))
        else:
            print("Focal length = " + str(input_parameters.ghy_focallength))

    if input_parameters.ghy_diff_plane == 1 or input_parameters.ghy_diff_plane == 3:
        calculation_parameters.xx_focal_ray = copy.deepcopy(calculation_parameters.xx_screen) + \
                                              input_parameters.ghy_focallength * np.tan(calculation_parameters.dx_ray)


    if input_parameters.ghy_diff_plane == 2 or input_parameters.ghy_diff_plane == 3:
        calculation_parameters.zz_focal_ray = copy.deepcopy(calculation_parameters.zz_screen) + \
                                              input_parameters.ghy_focallength * np.tan(calculation_parameters.dz_ray)


    t_image = calculation_parameters.shadow_oe_end._oe.T_IMAGE

    if input_parameters.ghy_distance < 0:
        if oe_number != 0:
            input_parameters.ghy_distance = t_image
            print("Distance not set (<-1), take from T_IMAGE " + str(input_parameters.ghy_distance))

    if oe_number != 0:
        if (input_parameters.ghy_distance == t_image):
            print("Defined OE star plane distance is different from T_IMAGE, used the defined distance = " + str(input_parameters.ghy_distance))
        else:
            print("Propagation distance = " + str(input_parameters.ghy_distance))

    if input_parameters.ghy_calcType == 3 or input_parameters.ghy_calcType == 4: # mirror/grating with figure error
        shadow_oe = calculation_parameters.shadow_oe_end

        if input_parameters.ghy_diff_plane == 1 or input_parameters.ghy_diff_plane == 3: #X
            offset_y_index =  0.0 if shadow_oe._oe.F_MOVE == 0 else shadow_oe._oe.OFFY/calculation_parameters.w_mirr_2D_values.delta_y()

            np_array = calculation_parameters.w_mirr_2D_values.z_values[:, int(len(calculation_parameters.w_mirr_2D_values.y_coord)/2 - offset_y_index)]

            calculation_parameters.w_mirror_lx = ScaledArray.initialize_from_steps(np_array,
                                                                                   calculation_parameters.w_mirr_2D_values.x_coord[0],
                                                                                   calculation_parameters.w_mirr_2D_values.x_coord[1] - calculation_parameters.w_mirr_2D_values.x_coord[0])

        if input_parameters.ghy_diff_plane == 2 or input_parameters.ghy_diff_plane == 3: #Z
            offset_x_index =  0.0 if shadow_oe._oe.F_MOVE == 0 else shadow_oe._oe.OFFX/calculation_parameters.w_mirr_2D_values.delta_x()

            np_array = calculation_parameters.w_mirr_2D_values.z_values[int(len(calculation_parameters.w_mirr_2D_values.x_coord)/2 - offset_x_index), :]

            calculation_parameters.w_mirror_lz = ScaledArray.initialize_from_steps(np_array,
                                                                                   calculation_parameters.w_mirr_2D_values.y_coord[0],
                                                                                   calculation_parameters.w_mirr_2D_values.y_coord[1] - calculation_parameters.w_mirr_2D_values.y_coord[0])

    # generate intensity profile (histogram): I_ray(z) curve

    if input_parameters.ghy_diff_plane == 1: # 1d in X
        if (input_parameters.ghy_nbins_x < 0):
            input_parameters.ghy_nbins_x = 200

        input_parameters.ghy_nbins_x = min(input_parameters.ghy_nbins_x, round(len(calculation_parameters.xx_screen) / 20)) #xshi change from 100 to 20
        input_parameters.ghy_nbins_x = max(input_parameters.ghy_nbins_x, 10)

        ticket = calculation_parameters.screen_plane_beam._beam.histo1(1,
                                                                       nbins=int(input_parameters.ghy_nbins_x),
                                                                       xrange=[np.min(calculation_parameters.xx_screen), np.max(calculation_parameters.xx_screen)],
                                                                       nolost=1,
                                                                       ref=23)

        bins = ticket['bins']

        calculation_parameters.wIray_x = ScaledArray.initialize_from_range(ticket['histogram'], bins[0], bins[len(bins)-1])
    elif input_parameters.ghy_diff_plane == 2: # 1d in Z
        if (input_parameters.ghy_nbins_z < 0):
            input_parameters.ghy_nbins_z = 200

        input_parameters.ghy_nbins_z = min(input_parameters.ghy_nbins_z, round(len(calculation_parameters.zz_screen) / 20)) #xshi change from 100 to 20
        input_parameters.ghy_nbins_z = max(input_parameters.ghy_nbins_z, 10)

        ticket = calculation_parameters.screen_plane_beam._beam.histo1(3,
                                                                       nbins=int(input_parameters.ghy_nbins_z),
                                                                       xrange=[np.min(calculation_parameters.zz_screen), np.max(calculation_parameters.zz_screen)],
                                                                       nolost=1,
                                                                       ref=23)

        bins = ticket['bins']

        calculation_parameters.wIray_z = ScaledArray.initialize_from_range(ticket['histogram'], bins[0], bins[len(bins)-1])
    elif input_parameters.ghy_diff_plane == 3: # 2D
        if (input_parameters.ghy_nbins_x < 0):
            input_parameters.ghy_nbins_x = 50

        if (input_parameters.ghy_nbins_z < 0):
            input_parameters.ghy_nbins_z = 50

        input_parameters.ghy_nbins_x = min(input_parameters.ghy_nbins_x, round(np.sqrt(len(calculation_parameters.xx_screen) / 10)))
        input_parameters.ghy_nbins_z = min(input_parameters.ghy_nbins_z, round(np.sqrt(len(calculation_parameters.zz_screen) / 10)))

        input_parameters.ghy_nbins_x = max(input_parameters.ghy_nbins_x, 10)
        input_parameters.ghy_nbins_z = max(input_parameters.ghy_nbins_z, 10)

        ticket = calculation_parameters.screen_plane_beam._beam.histo2(col_h=1,
                                                                       col_v=3,
                                                                       nbins_h=int(input_parameters.ghy_nbins_x),
                                                                       nbins_v=int(input_parameters.ghy_nbins_z),
                                                                       xrange=[np.min(calculation_parameters.xx_screen), np.max(calculation_parameters.xx_screen)],
                                                                       yrange=[np.min(calculation_parameters.zz_screen), np.max(calculation_parameters.zz_screen)],
                                                                       nolost=1,
                                                                       ref=23)

        bins_h = ticket['bin_h_edges']
        bins_v = ticket['bin_v_edges']
        calculation_parameters.wIray_x = ScaledArray.initialize_from_range(ticket['histogram_h'], bins_h[0], bins_h[len(bins_h)-1])
        calculation_parameters.wIray_z = ScaledArray.initialize_from_range(ticket['histogram_v'], bins_v[0], bins_v[len(bins_v)-1])
        calculation_parameters.wIray_2d = ScaledMatrix.initialize_from_range(ticket['histogram'], bins_h[0], bins_h[len(bins_h)-1], bins_v[0], bins_v[len(bins_v)-1])

    calculation_parameters.gwavelength = np.average(calculation_parameters.wwavelength)

    if input_parameters.ghy_lengthunit == 0:
        um = "m"
        calculation_parameters.gwavelength *= 1e-10
    if input_parameters.ghy_lengthunit == 1:
        um = "cm"
        calculation_parameters.gwavelength *= 1e-8
    elif input_parameters.ghy_lengthunit == 2:
        um = "mm"
        calculation_parameters.gwavelength *= 1e-7

    print("Using MEAN photon wavelength (" + um + "): " + str(calculation_parameters.gwavelength))

    calculation_parameters.gknum = 2.0*np.pi/calculation_parameters.gwavelength #in [user-unit]^-1, wavenumber

    if input_parameters.ghy_calcType == 6:
        if input_parameters.crl_delta is None:
            calculation_parameters.crl_delta = get_delta(input_parameters, calculation_parameters)
        else:
            calculation_parameters.crl_delta = input_parameters.crl_delta

##########################################################################

def hy_prop(input_parameters=HybridInputParameters(), calculation_parameters=HybridCalculationParameters()):

    # set distance and focal length for the aperture propagation.
    if input_parameters.ghy_calcType in [1, 5, 6]: # simple aperture
        if input_parameters.ghy_diff_plane == 1: # X
            calculation_parameters.ghy_focallength = (calculation_parameters.ghy_x_max-calculation_parameters.ghy_x_min)**2/calculation_parameters.gwavelength/input_parameters.ghy_npeak
        elif input_parameters.ghy_diff_plane == 2: # Z
            calculation_parameters.ghy_focallength = (calculation_parameters.ghy_z_max-calculation_parameters.ghy_z_min)**2/calculation_parameters.gwavelength/input_parameters.ghy_npeak
        elif input_parameters.ghy_diff_plane == 3: # 2D
            calculation_parameters.ghy_focallength = (max(np.abs(calculation_parameters.ghy_x_max-calculation_parameters.ghy_x_min),
                                                          np.abs(calculation_parameters.ghy_z_max-calculation_parameters.ghy_z_min)))**2/calculation_parameters.gwavelength/input_parameters.ghy_npeak

        print("Focal length set to: " + str(calculation_parameters.ghy_focallength))

    # automatic control of number of peaks to avoid numerical overflow
    if input_parameters.ghy_npeak < 0: # number of bins control
        if input_parameters.ghy_diff_plane == 3:
            input_parameters.ghy_npeak = 10
        else:
            input_parameters.ghy_npeak = 50

    input_parameters.ghy_npeak = max(input_parameters.ghy_npeak, 5)

    if input_parameters.ghy_fftnpts < 0:
        input_parameters.ghy_fftnpts = 4e6
    input_parameters.ghy_fftnpts = min(input_parameters.ghy_fftnpts, 4e6)

    if input_parameters.ghy_diff_plane == 1: #1d calculation in x direction
        propagate_1D_x_direction(calculation_parameters, input_parameters)
    elif input_parameters.ghy_diff_plane == 2: #1d calculation in z direction
        propagate_1D_z_direction(calculation_parameters, input_parameters)
    elif input_parameters.ghy_diff_plane == 3: #2D
        propagate_2D(calculation_parameters, input_parameters)
        

def hy_conv(input_parameters=HybridInputParameters(), calculation_parameters=HybridCalculationParameters()):
    if input_parameters.ghy_diff_plane == 1: #1d calculation in x direction
        if calculation_parameters.do_ff_x:
            s1d = Sampler1D(calculation_parameters.dif_xp.get_values(), calculation_parameters.dif_xp.get_abscissas())
            pos_dif = s1d.get_n_sampled_points(len(calculation_parameters.xp_screen), seed=None if input_parameters.random_seed is None else (input_parameters.random_seed + 1))

            dx_wave = np.arctan(pos_dif) # calculate dx from tan(dx)
            dx_conv = dx_wave + calculation_parameters.dx_ray # add the ray divergence kicks

            calculation_parameters.xx_image_ff = calculation_parameters.xx_screen + input_parameters.ghy_distance*np.tan(dx_conv) # ray tracing to the image plane
            calculation_parameters.dx_conv = dx_conv

        if input_parameters.ghy_nf >= 1 and input_parameters.ghy_calcType > 1:
            s1d = Sampler1D(calculation_parameters.dif_x.get_values(), calculation_parameters.dif_x.get_abscissas())
            pos_dif = s1d.get_n_sampled_points(len(calculation_parameters.xx_focal_ray), seed=None if input_parameters.random_seed is None else (input_parameters.random_seed + 2))

            calculation_parameters.xx_image_nf = pos_dif + calculation_parameters.xx_focal_ray
    elif input_parameters.ghy_diff_plane == 2: #1d calculation in z direction
        if calculation_parameters.do_ff_z:
            s1d = Sampler1D(calculation_parameters.dif_zp.get_values(), calculation_parameters.dif_zp.get_abscissas())
            pos_dif = s1d.get_n_sampled_points(len(calculation_parameters.zp_screen), seed=None if input_parameters.random_seed is None else (input_parameters.random_seed + 3))

            dz_wave = np.arctan(pos_dif) # calculate dz from tan(dz)
            dz_conv = dz_wave + calculation_parameters.dz_ray # add the ray divergence kicks

            calculation_parameters.zz_image_ff = calculation_parameters.zz_screen + input_parameters.ghy_distance*np.tan(dz_conv) # ray tracing to the image plane
            calculation_parameters.dz_conv = dz_conv

        if input_parameters.ghy_nf >= 1 and input_parameters.ghy_calcType > 1:
            s1d = Sampler1D(calculation_parameters.dif_z.get_values(), calculation_parameters.dif_z.get_abscissas())
            pos_dif = s1d.get_n_sampled_points(len(calculation_parameters.zz_focal_ray), seed=None if input_parameters.random_seed is None else (input_parameters.random_seed + 4))

            calculation_parameters.zz_image_nf = pos_dif + calculation_parameters.zz_focal_ray
    elif input_parameters.ghy_diff_plane == 3: #2D
        if calculation_parameters.do_ff_x and calculation_parameters.do_ff_z:
            s2d = Sampler2D(calculation_parameters.dif_xpzp.z_values,
                            calculation_parameters.dif_xpzp.x_coord,
                            calculation_parameters.dif_xpzp.y_coord)
            pos_dif_x, pos_dif_z = s2d.get_n_sampled_points(len(calculation_parameters.zp_screen), seed=None if input_parameters.random_seed is None else (input_parameters.random_seed + 5))

            dx_wave = np.arctan(pos_dif_x) # calculate dx from tan(dx)
            dx_conv = dx_wave + calculation_parameters.dx_ray # add the ray divergence kicks

            calculation_parameters.xx_image_ff = calculation_parameters.xx_screen + input_parameters.ghy_distance*np.tan(dx_conv) # ray tracing to the image plane
            calculation_parameters.dx_conv = dx_conv

            dz_wave = np.arctan(pos_dif_z) # calculate dz from tan(dz)
            dz_conv = dz_wave + calculation_parameters.dz_ray # add the ray divergence kicks

            calculation_parameters.zz_image_ff = calculation_parameters.zz_screen + input_parameters.ghy_distance*np.tan(dz_conv) # ray tracing to the image plane
            calculation_parameters.dz_conv = dz_conv
    elif input_parameters.ghy_diff_plane == 4: #2D - x then Z
        pass


def hy_create_shadow_beam(input_parameters=HybridInputParameters(), calculation_parameters=HybridCalculationParameters()):

    do_nf = input_parameters.ghy_nf >= 1 and input_parameters.ghy_calcType > 1

    if do_nf:
        calculation_parameters.nf_beam = calculation_parameters.image_plane_beam.duplicate(history=False)
        calculation_parameters.nf_beam._oe_number = input_parameters.shadow_beam._oe_number

    if input_parameters.ghy_diff_plane == 1: #1d calculation in x direction
        if calculation_parameters.do_ff_x:
            calculation_parameters.ff_beam = calculation_parameters.image_plane_beam.duplicate(history=False)
            calculation_parameters.ff_beam._oe_number = input_parameters.shadow_beam._oe_number

            angle_perpen = np.arctan(calculation_parameters.zp_screen/calculation_parameters.yp_screen)
            angle_num = np.sqrt(1+(np.tan(angle_perpen))**2+(np.tan(calculation_parameters.dx_conv))**2)

            calculation_parameters.ff_beam._beam.rays[:, 0] = copy.deepcopy(calculation_parameters.xx_image_ff)
            calculation_parameters.ff_beam._beam.rays[:, 3] = np.tan(calculation_parameters.dx_conv)/angle_num
            calculation_parameters.ff_beam._beam.rays[:, 4] = 1/angle_num
            calculation_parameters.ff_beam._beam.rays[:, 5] = np.tan(angle_perpen)/angle_num

            if calculation_parameters.image_plane_beam_lost.get_number_of_rays() > 0:
                calculation_parameters.ff_beam = ShadowBeam.mergeBeams(calculation_parameters.ff_beam, calculation_parameters.image_plane_beam_lost, which_flux=1, merge_history=0)
        else:
            calculation_parameters.ff_beam = input_parameters.original_shadow_beam

        if do_nf:
            calculation_parameters.nf_beam._beam.rays[:, 0] = copy.deepcopy(calculation_parameters.xx_image_nf)

    elif input_parameters.ghy_diff_plane == 2: #1d calculation in z direction
        if calculation_parameters.do_ff_z:
            calculation_parameters.ff_beam = calculation_parameters.image_plane_beam.duplicate(history=False)
            calculation_parameters.ff_beam._oe_number = input_parameters.shadow_beam._oe_number

            angle_perpen = np.arctan(calculation_parameters.xp_screen/calculation_parameters.yp_screen)
            angle_num = np.sqrt(1+(np.tan(angle_perpen))**2+(np.tan(calculation_parameters.dz_conv))**2)

            calculation_parameters.ff_beam._beam.rays[:, 2] = copy.deepcopy(calculation_parameters.zz_image_ff)
            calculation_parameters.ff_beam._beam.rays[:, 3] = np.tan(angle_perpen)/angle_num
            calculation_parameters.ff_beam._beam.rays[:, 4] = 1/angle_num
            calculation_parameters.ff_beam._beam.rays[:, 5] = np.tan(calculation_parameters.dz_conv)/angle_num

            if calculation_parameters.image_plane_beam_lost.get_number_of_rays() > 0:
                calculation_parameters.ff_beam = ShadowBeam.mergeBeams(calculation_parameters.ff_beam, calculation_parameters.image_plane_beam_lost, which_flux=1, merge_history=0)
        else:
            calculation_parameters.ff_beam = input_parameters.original_shadow_beam

        if do_nf:
             calculation_parameters.nf_beam._beam.rays[:, 2] = copy.deepcopy(calculation_parameters.zz_image_nf)

    elif input_parameters.ghy_diff_plane == 3: # 2d calculation
        if calculation_parameters.do_ff_x or calculation_parameters.do_ff_z:
            calculation_parameters.ff_beam = calculation_parameters.image_plane_beam.duplicate(history=False)
            calculation_parameters.ff_beam._oe_number = input_parameters.shadow_beam._oe_number

            angle_num = np.sqrt(1+(np.tan(calculation_parameters.dz_conv))**2+(np.tan(calculation_parameters.dx_conv))**2)

            calculation_parameters.ff_beam._beam.rays[:, 0] = copy.deepcopy(calculation_parameters.xx_image_ff)
            calculation_parameters.ff_beam._beam.rays[:, 2] = copy.deepcopy(calculation_parameters.zz_image_ff)
            calculation_parameters.ff_beam._beam.rays[:, 3] = np.tan(calculation_parameters.dx_conv)/angle_num
            calculation_parameters.ff_beam._beam.rays[:, 4] = 1/angle_num
            calculation_parameters.ff_beam._beam.rays[:, 5] = np.tan(calculation_parameters.dz_conv)/angle_num

            if calculation_parameters.image_plane_beam_lost.get_number_of_rays() > 0:
                calculation_parameters.ff_beam = ShadowBeam.mergeBeams(calculation_parameters.ff_beam, calculation_parameters.image_plane_beam_lost, which_flux=1, merge_history=0)
        else:
            calculation_parameters.ff_beam = input_parameters.original_shadow_beam

    if do_nf and calculation_parameters.image_plane_beam_lost.get_number_of_rays() > 0:
        calculation_parameters.nf_beam = ShadowBeam.mergeBeams(calculation_parameters.nf_beam, calculation_parameters.image_plane_beam_lost, which_flux=1, merge_history=0)

    if input_parameters.file_to_write_out == 1:

        if input_parameters.ghy_n_oe < 0:
            str_n_oe = str(input_parameters.shadow_beam._oe_number)

            if input_parameters.shadow_beam._oe_number < 10:
                str_n_oe = "0" + str_n_oe
        else: # compatibility with old verion
            str_n_oe = str(input_parameters.ghy_n_oe)

            if input_parameters.ghy_n_oe < 10:
                str_n_oe = "0" + str_n_oe

        calculation_parameters.ff_beam.writeToFile("hybrid_ff_beam." + str_n_oe)
        if do_nf: calculation_parameters.nf_beam.writeToFile("hybrid_nf_beam." + str_n_oe)

    calculation_parameters.ff_beam.history = calculation_parameters.original_beam_history
    if do_nf: calculation_parameters.nf_beam.history = calculation_parameters.original_beam_history

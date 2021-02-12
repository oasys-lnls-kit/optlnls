#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 16:55:15 2021

@author: lordano
"""


import os
import numpy as np
from matplotlib import pyplot as plt
import subprocess
import time


def write_batch_file(batch_file_name, files_array):
    with open(batch_file_name, 'w') as ofile:
        for filename in files_array:
            ofile.write("type "+filename+" | Reflec.exe\n")
    ofile.closed
    
def write_reflec_grating_input(input_filename, grating_dict, energy_grid=[3, 10, 8], 
                               polarization="s", fourier_coeff=15, nem_filename='test.nem'):
    
    with open(input_filename, 'w') as f:
        
        # Reflec initialization
        f.write("0\nNO\nNO\n1\n1\nGR\n")
        
        # energy start and end
        f.write("{0:.3f}\n{1:.3f}\n".format(energy_grid[0], energy_grid[1])) 
        
        # check if the grating is coated
        if(grating_dict["coating"] == ""):
            f.write("0\n")
        else:
            f.write("1\n")
        
        # substrate information
        f.write(grating_dict["material"]+'\n')
        f.write('{0:6f}\n'.format(grating_dict["roughness"]))            
        f.write('{0:6f}\n'.format(grating_dict["material density"]))

        if(grating_dict["coating"] != ""):
            f.write(grating_dict["coating"]+'\n')
            f.write('{0:6f}\n'.format(grating_dict["coating thickness"]))            
            f.write('{0:6f}\n'.format(grating_dict["coating density"]))
    
        # incidence angle
        if(grating_dict["2theta constant"] != 0):
            f.write("{0:.9f}\n".format(-1 * grating_dict["2theta constant"])) # the negative sign comes here
        elif(grating_dict["cff"] != 0):
            f.write("700\n{0:.6f}\n".format(grating_dict["cff"]))
        else:
            f.write("{0:.9f}\n".format(grating_dict["alpha"])) # fixed incidence angle
            
        # grating parameters
        f.write("{0:.6f}\n".format(grating_dict["k0"]))
        f.write("{0:d}\n".format(grating_dict["order"]))
        
        if(grating_dict["type"] == "laminar"):
            f.write("3\n{0:.6f}\n".format(grating_dict["apex"]))
            f.write("{0:.6f}\n".format(grating_dict["groove depth"]))
            f.write("{0:.6f}\n".format(grating_dict["groove ratio"]))  
        elif(grating_dict["type"] == "blazed"):              
            f.write("1\n{0:.6f}\n".format(grating_dict["blaze"]))
            f.write("{0:.6f}\n".format(grating_dict["apex"]))            
        elif(grating_dict["type"] == "sinusoidal"):      
            f.write("2\n{0:.6f}\n".format(grating_dict["groove depth"]))
        else:
            print("INVALID GRATING TYPE! USE ONE BETWEEN: 'laminar' / 'blazed' / 'sinusoidal' ")
            
        # other parameters (fourier coefficients and polarization)
        f.write("{0:d}\nYES\n".format(fourier_coeff))
        
        if(polarization == 's'):
            f.write('51\n')
        elif(polarization == 'p'):
            f.write('52\n')
        elif(polarization == 'u'):
            f.write('53\n')
        else:
            print("INVALID POLARIZATION! USE ONE BETWEEN: 's' / 'p' / 'u' ")
        
        f.write('1\n')
        f.write('{0:d}\n'.format(energy_grid[2]))
        f.write('YES\nNO\nYES\n3\n')
        f.write(nem_filename[:-4]+'\n9\n0\n')
           
def delete_nem_file(nem_filename, reflec_path):
    path = os.path.join(reflec_path, 'Files', nem_filename)
    try:
        os.remove(path)
    except:
        print(".nem file not found: " + path)
    
def run_reflec(reflec_path, cmd='wineconsole', wait_execution=0):
    bashCommand = cmd + ' ' + 'Reflec.exe' 
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE, cwd=reflec_path)
    output, error = process.communicate()
    if(wait_execution):
        process.wait()

def run_reflec_from_bat_file(reflec_path, bat_filename, cmd='wineconsole', wait_execution=0):
    bashCommand = cmd + ' ' + bat_filename
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE, cwd=reflec_path)
    output, error = process.communicate()
    if(wait_execution):
        process.wait()


def test_reflec_from_python():

    reflec_path = '/home/lordano/Software/RAY_Bessy/PROGRAM1/'
        
    #####################
    ## define grating parameters
   
    
    grating = {"type": "laminar",
                 "k0": 75,
                 "groove depth": 230,
                 "groove ratio": 0.55,
                 "apex": 90,
                 "order": 1,
                  
                 "2theta constant": 162.0,
                 "cff": 0,
                 "alpha": 0,
                  
                 "material": "Si",
                 "material density": 2.32,
                 "coating": "Pt",
                 "coating density": 21.45*0.95,
                 "coating thickness": 20.0,
                 "roughness": 0.0,
                }
    
    
    # grating = {"type": "blazed",
    #            "k0": 1100,
    #            "blaze": 1.0,
    #            "apex": 170,
    #            "order": 1,
                     
    #            "2theta constant": 0,
    #            "cff": 2.25,
    #            "alpha": 0,
                     
    #            "material": "Au",
    #            "material density": 18.3,
    #            "roughness": 0.1,
    #            "coating": "",
    #            "coating density": 2.0,
    #            "coating thickness": 20.0,
    #            }
    

    ##################
    ## write input
    
    input_prefix = 'test_input'
    bat_filename = input_prefix + '.bat'
    nem_filename = 'test_blazed.nem'
    
    write_reflec_grating_input(input_filename=reflec_path + input_prefix+'.txt',
                               grating_dict=grating,
                               energy_grid=[2, 15, 51],
                               polarization='s', fourier_coeff=15,
                               nem_filename=nem_filename)
    
    write_batch_file(batch_file_name=reflec_path + bat_filename,
                     files_array=[input_prefix+'.txt'])
    
    
    
    ##################
    ## run reflec
    
    if(1):
        
        delete_nem_file(nem_filename, reflec_path)
        run_reflec_from_bat_file(reflec_path, bat_filename)
    
        # wait to run reflec and write output
        if(1):
            time.sleep(5)
    
    
    ###############
    ## plot
    
    if(1):
                  
        nem_path = reflec_path + 'Files/' + nem_filename
        # nem_path = reflec_path + 'Files/' + 'testb2.nem'
        energy, efficiency = np.genfromtxt(nem_path, unpack=True)
        
        plt.figure()
        plt.plot(energy, efficiency)



if(__name__ == '__main__'):
    
    test_reflec_from_python()





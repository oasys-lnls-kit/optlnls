
import numpy as np
from matplotlib import pyplot as plt

def test_plot_beam():
    
    from optlnls.importing import read_srw_int
    from optlnls.plot import plot_beam
        			
    filename = '/media/sergio.lordano/DATA/Oasys/SAPE/SRW/cfg5/SAPE_cfg5b_grat_focus_mE.dat'
    beam = read_srw_int(filename)
    beam = beam[0]

    prefix = 'test'
    
    plot_beam(beam2D=beam)


def test_SRW_figure_error():

    import sys
    sys.path.insert(0, '/media/sergio.lordano/DATA/SRW_Dev/env/work/srw_python')
    
    from optlnls.surface import SRW_figure_error
    
    filename = '/media/sergio.lordano/DATA/Oasys/EMA/MagnetHutch/KP-16-0087-HFM_sp_mm.dat'
    
    Tr = SRW_figure_error(filename, unit_factor=1e-3, angle_in=2.3e-3, angle_out=2.3e-3, orientation_x_or_y='x')

def test_reflectivity_xrays():
    
    from optlnls.mirror import reflectivity_xrays
    
    e = np.linspace(30, 20000, 200)    
    
    thetaM0 = 0.005*180/np.pi # [degree]
    angleM0 = 90 - thetaM0
    
    R = reflectivity_xrays('Rh', 11.78, 102.9, e, angleM0)
      
    # np.savetxt('Refl_Rh_5mrad.txt', np.transpose(np.array([e, R])), '%.6e')
    
    plt.figure()
    plt.loglog(e, R)
    plt.show()


def test_srw_undulator_spectrum():
    
    import sys
    sys.path.insert(0, '/media/sergio.lordano/DATA/SRW_Dev/env/work/srw_python')
          
    from optlnls.source import srw_undulator_spectrum
    
    mag_field=[19.0e-3, 2.4, 0, 1.29445, 0, 0, -1, +1]
    electron_beam=[19.1e-6, 2.0e-6, 13.0e-6, 1.3e-6, 3.0, 0.085e-2, 0.350]
    energy_grid=[100.0, 20000.0, 2000]
    sampling_mesh=[10.0, 1.44e-3, 1.44e-3]    
    precision = [1, 30, 1.0, 1.0]   
    
    spec = srw_undulator_spectrum(mag_field, electron_beam, energy_grid, sampling_mesh, precision)    

    e_pts = np.linspace(energy_grid[0], energy_grid[1], energy_grid[2])   
    
    fsize=14
    plt.figure()
    plt.axes([0.2, 0.15, 0.75, 0.8])
    plt.semilogy(e_pts/1000.0, spec, 'g-', label='Total Flux')
    plt.grid()
    plt.legend(loc='best', fontsize=fsize)
    plt.minorticks_on()
    plt.ylabel('Flux [ph/s/0.1%bw/100mA]', fontsize=fsize)
    plt.xlabel('Photon Energy [keV]', fontsize=fsize)    
    
    plt.show()


def test_height_error_analysis():
    
    from optlnls.surface import analyze_height_error
    
    filelist = []
    filelist.append('/media/sergio.lordano/DATA/Oasys/EMA/MagnetHutch/KP-16-0087-VFM_sp_mm.dat')
    filelist.append('/media/sergio.lordano/DATA/Oasys/EMA/MagnetHutch/KP-16-0087-HFM_sp_mm.dat')
    
    analyze_height_error(filelist, 1e-3)
    

if __name__ == '__main__':
    
    print('running tests...')
    
    # test_SRW_figure_error()
    # test_plot_beam()
    # test_reflectivity_xrays()
    # test_srw_undulator_spectrum()
    # test_height_error_analysis()

















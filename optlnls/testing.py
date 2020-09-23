
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
    import glob
    
    # filelist = []
    # filelist.append('/media/sergio.lordano/DATA/Oasys/EMA/MagnetHutch/KP-16-0087-VFM_sp_mm.dat')
    # filelist.append('/media/sergio.lordano/DATA/Oasys/EMA/MagnetHutch/KP-16-0087-HFM_sp_mm.dat')
    # filelist.append('/media/sergio.lordano/DATA/optlnls/optlnls/outputs/test_1.dat')
    
    filelist = sorted(glob.glob('/media/sergio.lordano/DATA/optlnls/optlnls/outputs/*.dat'))
    
    analyze_height_error(filelist, 1e-3, workingFolder='outputs')
    
def test_figure_error_generation():
    
    from optlnls.surface import gen_figure_error, analyze_height_error
    
    gen_figure_error(plot=True, filename='outputs/test_he.dat')

    filelist = []
    filelist.append('test_he.dat')
    
    analyze_height_error(filelist, 1e-3, workingFolder='outputs')

def test_figure_error_generation_multi():
    
    from optlnls.surface import gen_figure_error_multi
    
    list_height_rms = np.array([5, 10, 15, 20, 25, 30])*1.2e-9
    list_slope_rms = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])*1e-6
    
    idx = 7
    
    gen_figure_error_multi(L=380e-3, W=30e-3, stepL=0.5e-3, stepW=1.0e-3, 
                           betaW=2.0, rmsW=5e-9, typeW='h',
                           list_height_rms=list_height_rms[:idx], 
                           list_slope_rms=list_slope_rms[:idx], 
                           seedL=1205, seedW=982, prefix='outputs/SAP_M1_f3', tolL=0.001)
    
    
def test_cedro_grating_alpha():
    
    from optlnls.constants import hc
    from optlnls.grating import calc_constant_included_angle, calc_grating_beta    

    
    energy = np.linspace(2, 10, 101)
    # energy = np.linspace(3, 10, 101)
    wavelength = hc / energy

    k0 = np.linspace(100, 1000, 51)
    
    wavelength_x, k0_y = np.meshgrid(wavelength, k0)
    energy_x = np.array([energy for i in range(len(k0))])

    alpha, beta = calc_constant_included_angle(wavelength_x, k0_y, 1, 90)
    


    fig, (ax1, ax2) = plt.subplots(figsize=(10,4), ncols=2, sharey=True)
    fig.subplots_adjust(wspace=0.15)
    xticks = np.arange(int(energy.min()), int(energy.max())+1)
    
    im1 = ax1.contourf(energy_x, k0_y, alpha, cmap='tab20', levels=20)
    cb1 = fig.colorbar(im1, ax=ax1, label='alpha [degrees]')
    ax1.set_ylabel('k0 [lines/mm]')
    ax1.set_xlabel('energy [eV]')
    ax1.grid()
    ax1.set_xticks(xticks)
    
    im2 = ax2.contourf(energy_x, k0_y, beta, cmap='tab20', levels=20)
    cb2 = fig.colorbar(im2, ax=ax2, label='beta [degrees]')
    ax2.set_xlabel('energy [eV]')
    ax2.grid()
    ax2.set_xticks(xticks)
    
    # plt.colorbar()
    
    # plt.savefig('CED_angles.png', dpi=400)
    
    
    plt.show()
    
    # print()
    


if __name__ == '__main__':
    
    print('running tests...')
    
    # test_SRW_figure_error()
    # test_plot_beam()
    # test_reflectivity_xrays()
    # test_srw_undulator_spectrum()
    # test_height_error_analysis()
    # test_figure_error_generation_multi()
    # test_height_error_analysis()
    # test_cedro_grating_alpha()




    from optlnls.constants import hc
    from optlnls.grating import calc_constant_included_angle, calc_grating_beta    

    
    energy = 6.0
    # energy = np.linspace(3, 10, 101)
    wavelength = hc / energy

    k0 = 300
    
    alpha, beta = calc_constant_included_angle(wavelength, k0, 1, 90)


    m = -1
    energy = 6.0
    wavelength = 1239.8419843320028*1e-9 / energy # meters
    k0 = 300 # lines/mm
    # alpha = 60
    beta2 = calc_grating_beta(wavelength, alpha, k0, m)
    
    
    
    p = 4.0
    q = 4.0
    f = 1/(1/p+1/q)

    cos_alpha = np.cos(alpha*np.pi/180)
    cos_beta = np.cos(beta*np.pi/180)
    
    Rs = f * (cos_alpha + cos_beta) 
    Rm = (cos_alpha + cos_beta) / (cos_alpha**2 / p + cos_beta**2 / q)    

    # Rs_coddington = 2 * f * np.sin(alpha*np.pi/180)
    # Rm_coddington = 2 * f / np.sin(alpha*np.pi/180)

    print("alph = ", round(alpha, 7), ' deg')
    print("beta = ", round(beta, 7), ' deg')
    print("beta2 = ", round(beta2, 7), ' deg')
    print("sagittal radius = ", round(Rs, 6), ' m')
    print("meridional radius = ", round(Rm, 6), ' m')


    print(alpha)
    print(beta)









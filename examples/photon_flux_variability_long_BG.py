# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 16:37:13 2022

@author: Kate
#"""
#from astropy.convolution import AiryDisk2DKernel
#from astropy.io import fits
#from astropy.utils.data import get_pkg_data_filename
#from astropy.convolution import Gaussian2DKernel
#from scipy.signal import convolve as scipy_convolve
#from astropy.convolution import convolve
#
#from scipy.linalg import hadamard
import numpy as np
import sys
#from scipy.constants import h
#from scipy.constants import c
sys.path.append('C:/Users/Kate/Documents/hadamard/sandbox')

from hadamard_class_v2 import *
HTSI = HTSI_Models2()

from data_sim_class import *
SIM = data_sim()
#%
# Observation properties
folder = 'C:/Users/Kate/Documents/hadamard/'

#mat_type = 'S'
#order=127

def run_flux_simulations(mat_type, order, BG_mag):

    lam0, lam = 400.0,800.0
    nm_sample = 2
    slit_width = 1
    t= 30 # exposure time
    DMD_contrast = 2000
    n = 1024 # 2d img array size (nxn)
    ########
    bitdepth = 16
    max_adu = np.int(2**bitdepth -1)
    #max_adu  = 9000000
    gain = 2.1 #ADU/e-
    dt = 0.0008 # 0.0002  # The thermal dark signal in e-/pix/sec at the operating temp of -90C
    rn = 3.8 # e-, readout noise    
    
    bg_mags = []
    src_mags = []    
        
    Xp = order*slit_width*1.2
    Yp = order*slit_width*1.2 # 4112 # number of pixels in y dimension
    num_pix = Xp * Yp # total number of pixels 
    CCD = [t, dt, num_pix, rn, gain, max_adu]

    htsi_mse_obj = []
    mos_mse_obj = []

    Qs = []
    fs = []

    Q_rois = []

    c = order/2
    x1, x2, y1, y2 = np.int(c-15),np.int(c+15),np.int(c-15),np.int(c+15)
    f = [1,0.75, 0.5, 0.4, 0.3, 0.25, 0.2, 0.19, 0.18, 0.17, 0.16, 0.15, 0.12, 0.11, 0.105, 0.1, 0.09, 0.08, 0.07, 0.05, 0.03, 0.01]
    f = np.asarray(f)

    for i in range (0, len(f)):
        #x = exponents[i]
        fi = f[i]
        sky1, wls, SB_V, S_SB_V = SIM.sim_data_flux_basic_BG(lam0, lam, nm_sample, n, radius, BG_mag,fi)

        if mat_type == 'H':
            htsi_data, sky_ADU, htsi_mse, ccd_noise, photon_noise = HTSI.htsi(sky1, order, slit_width, DMD_contrast, CCD)
        if mat_type == 'S':
            htsi_data, sky_ADU, htsi_mse, ccd_noise, photon_noise = HTSI.S_htsi(sky1, order, slit_width, DMD_contrast, CCD)
        mos_data, sky_ADU_mos, mos_mse, M_ccd_noise, M_photon_noise = HTSI.MOS(sky1, order, slit_width, DMD_contrast, CCD)
       
        Q_whole = np.divide((np.sqrt(np.median(mos_mse))),(np.sqrt(np.median(htsi_mse))))
        htsi_S, mos_S, h_spec, m_spec, s_spec, Q, h_mse, m_mse = HTSI.SNR_Q_ROI(x1,x2,y1,y2, htsi_data, mos_data, sky_ADU)
        
        Qs = np.append(Qs, Q_whole)
                
        htsi_mse_obj = np.append(htsi_mse_obj,h_mse) 
        mos_mse_obj = np.append(mos_mse_obj,m_mse)
       
        bg_mags =np.append(bg_mags, SB_V)
        src_mags = np.append(bg_mags, S_SB_V)
        
        Q_rois = np.append(Q_rois,Q)
        
        print ('iteration '+str(i+1)+' of '+str(len(f)))
        fs = np.append(fs, fi)

    return  Qs, Q_rois, bg_mags, src_mags


#%%%
       # BG_mag = 0.14, 0.1, 0.08, 0.0615, 0.055, 0.03, 0.022

################################################### SAMOS low-Resolution #############################################################
mat_type = 'S'
order = 127
radius = 30
BG_mag = 0.055
Qs, Q_rois, bg_mags, src_mags = run_flux_simulations(mat_type, order, BG_mag)
data = np.vstack((Qs, Q_rois, bg_mags, src_mags))
np.savetxt(folder+'Flux_SAMOS_with_skyBG'+str(radius)+'radius_Vmag_'+str(BG_mag)+str(mat_type)+str(order)+'.txt', data, delimiter='\t')    
#%%
mat_type = 'H'
order = 128
radius = 30
BG_mag = 0.055
Qs, Q_rois, bg_mags, src_mags = run_flux_simulations(mat_type, order, BG_mag)
data = np.vstack((Qs, Q_rois, bg_mags, src_mags))
np.savetxt(folder+'Flux_SAMOS_with_skyBG'+str(radius)+'radius_Vmag_'+str(BG_mag)+str(mat_type)+str(order)+'.txt', data, delimiter='\t')    
#%%
mat_type = 'S'
order = 127
radius = 30
BG_mag = 0.022
Qs, Q_rois, bg_mags, src_mags = run_flux_simulations(mat_type, order, BG_mag)
data = np.vstack((Qs, Q_rois, bg_mags, src_mags))
np.savetxt(folder+'Flux_SAMOS_with_skyBG'+str(radius)+'radius_Vmag_'+str(BG_mag)+str(mat_type)+str(order)+'.txt', data, delimiter='\t')    
#%%
mat_type = 'S'
order = 127
radius = 30
BG_mag = 0.08
Qs, Q_rois, bg_mags, src_mags = run_flux_simulations(mat_type, order, BG_mag)
data = np.vstack((Qs, Q_rois, bg_mags, src_mags))
np.savetxt(folder+'Flux_SAMOS_with_skyBG'+str(radius)+'radius_Vmag_'+str(BG_mag)+str(mat_type)+str(order)+'.txt', data, delimiter='\t')    
#%%
mat_type = 'S'
order = 127
radius = 30
BG_mag = 0.0615
Qs, Q_rois, bg_mags, src_mags = run_flux_simulations(mat_type, order, BG_mag)
data = np.vstack((Qs, Q_rois, bg_mags, src_mags))
np.savetxt(folder+'Flux_SAMOS_with_skyBG'+str(radius)+'radius_Vmag_'+str(BG_mag)+str(mat_type)+str(order)+'.txt', data, delimiter='\t')    
#%%
mat_type = 'S'
order = 127
radius = 30
BG_mag = 0.01
Qs, Q_rois, bg_mags, src_mags = run_flux_simulations(mat_type, order, BG_mag)
data = np.vstack((Qs, Q_rois, bg_mags, src_mags))
np.savetxt(folder+'Flux_SAMOS_with_skyBG'+str(radius)+'radius_Vmag_'+str(BG_mag)+str(mat_type)+str(order)+'.txt', data, delimiter='\t')    
#%%
mat_type = 'S'
order = 127
radius = 30
BG_mag = 0.14
Qs, Q_rois, bg_mags, src_mags = run_flux_simulations(mat_type, order, BG_mag)
data = np.vstack((Qs, Q_rois, bg_mags, src_mags))
np.savetxt(folder+'Flux_SAMOS_with_skyBG'+str(radius)+'radius_Vmag_'+str(BG_mag)+str(mat_type)+str(order)+'.txt', data, delimiter='\t')    

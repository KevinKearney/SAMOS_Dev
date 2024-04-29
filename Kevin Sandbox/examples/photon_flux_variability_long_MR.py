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
import sys, os
#from scipy.constants import h
#from scipy.constants import c
cwd = os.getcwd()
parent = os.path.abspath(os.path.join(cwd, os.pardir))
sys.path.append(parent)
sys.path.append(os.path.join(parent,'sandbox'))

from hadamard_class_v3 import HTSI_Models
#import hadamard_class_v3

HTSI = HTSI_Models()

from data_sim_class import *
SIM = data_sim()
#%
# Observation properties
#folder = 'C:/Users/Kate/Documents/hadamard/'
folder = os.path.join(parent,'sandbox')

#mat_type = 'S'
#order=127

def run_flux_simulations(sky, mat_type, order):
    
    Xp = order*slit_width*1.2 # THIS ACCOUNTS FOR THE SAMPLING of the DMD on the CCD. SHOULD BE 1.125pix/slit.
    Yp = order*slit_width*1.2 # 4112 # number of pixels in y dimension
    num_pix = Xp * Yp # total number of pixels 
    CCD = [t, dt, num_pix, rn, gain, max_adu]
    
    mos_SNRs_obj = []
    htsi_SNRs_obj = []
    htsi_mses = []
    htsi_mse_obj = []
    mos_mse_obj = []
    mos_mses = []
    Qs = []
    fs = []
    obj_total_S = []
    obj_avg_S = []
    obj_max_S = []
    Q_rois = []
    M_ccd = []
    ccd = []
    c = order/2
    x1, x2, y1, y2 = int(c-15),int(c+15),int(c-15),int(c+15)
    #f2 = [0.06,0.08,0.1,0.2,0.4,0.6,0.8,1,1.5,2,2.5,3,3.5,4,4.5,5,6,7,8,9,10,15,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,220,240,260,280,300,400,500]
    f2 = [0.06,0.08,0.1,0.2,0.4,0.6,0.8,1,2,4,6,8,10,15,20,30,40,50,60,70,80,90,100,120,140,160,180,200,220,240,260,280,300,400,500,600,700,800,900,1000,1200,1500,2000,3000,4000,5000,8000,10000]
    f2=[1000] #MODIFIED BY MASSIMO TO SET THE fi FACTOR BELOW = 1, AND ONOY ONE LOOP
    #f2 is a coefficient applied to the sky brighness, we test several values
    S = []
    avgS = []
    f2 = np.asarray(f2)
    f= f2 *0.001#* (0.0001)
    for i in range (0, len(f)):
        #x = exponents[i]
        fi = f[i]
        sky1 = sky*fi     #here one applied the factor
        if mat_type == 'H':
            htsi_data, sky_ADU, htsi_mse, ccd_noise, photon_noise = HTSI.htsi(sky1, order, slit_width, DMD_contrast, CCD)
        if mat_type == 'S':
            htsi_data, sky_ADU, htsi_mse, ccd_noise, photon_noise = HTSI.S_htsi(sky1, order, slit_width, DMD_contrast, CCD)
        mos_data, sky_ADU_mos, mos_mse, M_ccd_noise, M_photon_noise = HTSI.MOS(sky1, order, slit_width, DMD_contrast, CCD)
       
        Q_whole = np.divide((np.sqrt(np.median(mos_mse))),(np.sqrt(np.median(htsi_mse))))
        
        htsi_S, mos_S, h_spec, m_spec, s_spec, Q, h_mse, m_mse = HTSI.SNR_Q_ROI(x1,x2,y1,y2, htsi_data, mos_data, sky_ADU)
        
        total_s = np.sum(s_spec)
        avg_s = np.mean(s_spec)
        S = np.append(S,total_s)
        avgS = np.append(avgS, avg_s)
        M_ccd = np.append(M_ccd, M_ccd_noise)
        ccd = np.append(ccd, ccd_noise)

        obj_total_S = np.append(obj_total_S,total_s) 
        obj_avg_S = np.append(obj_avg_S,avg_s)
        obj_max_S = np.append(obj_max_S, np.max(s_spec))
        
        mos_mses = np.append(mos_mses,np.mean(mos_mse))
        Qs = np.append(Qs, Q_whole)
        htsi_mses = np.append(htsi_mses, np.mean(htsi_mse))
        
        htsi_Ss_obj = np.append(htsi_SNRs_obj, htsi_S)
        mos_Ss_obj = np.append(mos_SNRs_obj, mos_S)
        
        htsi_mse_obj = np.append(htsi_mse_obj,h_mse) 
        mos_mse_obj = np.append(mos_mse_obj,m_mse)
        
        Q_rois = np.append(Q_rois,Q)
        
        print ('iteration '+str(i+1)+' of '+str(len(f)))
        fs = np.append(fs, fi)
#MOSIFIED TO GET IN OUTPUT THE IMAGE INSTEAD OF THE STATISTICS
#    return fs, obj_total_S, obj_avg_S, Qs, htsi_mses, mos_mses, htsi_Ss_obj, mos_Ss_obj, htsi_mse_obj, mos_mse_obj,obj_max_S,ccd, M_ccd, Q_rois
    return htsi_data

#%%SAMOS low res, repeat with high res & repeat
lam0, lam = 400.0,800.0  #wl; in nm
nm_sample = 2   #sampling every 2 nm we have 400/2 = 200 wavelengths
slit_width = 1
t= 1.00 # exposure time   #can be modified depending on the brightness of the scene. Kathy original has 100s
DMD_contrast = 2000
n = 1024 # 2d img array size (nxn)
########
pix_size = 15 * 10**-6
full_well_cap = 350000 # Pixel full well capactiy in e- 
bitdepth = 16
max_adu = int(2**bitdepth -1)  #65K 
#max_adu  = 9000000
gain = 2.1 #ADU/e-
dt = 0.0008 # 0.0002  # The thermal dark signal in e-/pix/sec at the operating temp of -90C
rn = 3.8 # e-, readout noise
#%%

################################################### SAMOS low-Resolution #############################################################
mat_type = 'S'
order = 127

"""
#ORIGINAL Kathy CODE 
radius = 30
sky_orig, wls = SIM.sim_data_flux_basic(lam0, lam, nm_sample, n, radius)   #radius = 30 creates a 30x30 square at the center
sky_orig, T = SIM.apply_thruput(sky_orig, wls, 'SAMOS_low')
sky_orig = sky_orig*t    #multiply by exposure time
sky = SIM.apply_psf(sky_orig, 'moffat', 4) # 4 ~ 0.4" seeing
"""

#MASSIMO's VARIATION USING ONE OF THE jpg IMAGES 
sky, wls = SIM.sim_sky_stellar2(lam0, lam, nm_sample)
sky=sky/100.
if np.min(sky) < 0:         #make sky positive
    sky = sky+(-1.1*(np.min(sky)))
if np.min(sky) == 0:
    sky = sky+1

htsi_data = run_flux_simulations(sky, mat_type, order)

print(htsi_data.dtype)
print(sky.dtype)
#DISPLAY
import matplotlib.pyplot as plt
fig = plt.figure()
i_wavelength = 100  #modify to probe different wl between 0 (400nm) Nand 199 (800nm)
#ORIGINAL:
plt.imshow(sky[256-64:256+63,256-64:256+63,i_wavelength]) 
plt.show()
#HADAMARD
plt.imshow(htsi_data[:,:,i_wavelength]) 
plt.tight_layout()
plt.show()
sys.exit(1)
#THIS ENDS MASSIMOS'S VARIANT

    
#################################  sky has dimsions  [y,x,lambda] = (1024, 1024, 200)
fs, obj_total_S, obj_avg_S, Qs, htsi_mses, mos_mses, htsi_SNRs_obj, mos_SNRs_obj, htsi_mse_obj, mos_mse_obj,obj_max_S,NR_whole, NR_M_whole, ccd_noise, M_ccd_noise, Q_rois = run_flux_simulations(sky, mat_type, order)
data = np.vstack((fs, obj_total_S, obj_avg_S, Qs, htsi_mses, mos_mses, htsi_SNRs_obj, mos_SNRs_obj, htsi_mse_obj, mos_mse_obj,obj_max_S,NR_whole, NR_M_whole, ccd_noise, M_ccd_noise, Q_rois))
np.savetxt(folder+'Flux_SAMOS_LR_fixed4_'+str(radius)+'radius_'+str(t)+'s_'+str(mat_type)+str(order)+'.txt', data, delimiter='\t')    

data_labels = np.vstack(('fs', 'obj_total_S', 'obj_avg_S', 'Qs', 'htsi_mses', 'mos_mses', 'htsi_SNRs_obj', 'mos_SNRs_obj', 'htsi_mse_obj', 'mos_mse_obj','Max signal S (obj)', 'HTSI noise ratio', 'MOS noise ratio','HTSI CCD noise','MOS CCD noise','Q for the roi'))
del fs 
del obj_total_S
del obj_avg_S
del Qs
del htsi_mses
del mos_mses
del htsi_SNRs_obj
del mos_SNRs_obj
del htsi_mse_obj
del mos_mse_obj
del obj_max_S
print('data variables cleared.')
#%
mat_type = 'H'
order = 128
fs, obj_total_S, obj_avg_S, Qs, htsi_mses, mos_mses, htsi_SNRs_obj, mos_SNRs_obj, htsi_mse_obj, mos_mse_obj, obj_max_S, NR_whole, NR_M_whole, ccd_noise, M_ccd_noise, Q_rois = run_flux_simulations(sky, mat_type, order)
data = np.vstack((fs, obj_total_S, obj_avg_S, Qs, htsi_mses, mos_mses, htsi_SNRs_obj, mos_SNRs_obj, htsi_mse_obj, mos_mse_obj,obj_max_S,NR_whole, NR_M_whole, ccd_noise, M_ccd_noise, Q_rois))
np.savetxt(folder+'Flux_SAMOS_LR_fixed4_'+str(radius)+'radius_'+str(t)+'s_'+str(mat_type)+str(order)+'.txt', data, delimiter='\t')    

print('File: Flux_data_labels_'+str(radius)+'radius_'+str(t)+'s_'+str(mat_type)+str(order)+'.txt saved.')
del fs 
del obj_total_S
del obj_avg_S
del Qs
del htsi_mses
del mos_mses
del htsi_SNRs_obj
del mos_SNRs_obj
del htsi_mse_obj
del mos_mse_obj
del obj_max_S
print('data variables cleared.')



mat_type = 'H'
order = 64

fs, obj_total_S, obj_avg_S, Qs, htsi_mses, mos_mses, htsi_SNRs_obj, mos_SNRs_obj, htsi_mse_obj, mos_mse_obj, obj_max_S, NR_whole, NR_M_whole, ccd_noise, M_ccd_noise, Q_rois = run_flux_simulations(sky, mat_type, order)
data = np.vstack((fs, obj_total_S, obj_avg_S, Qs, htsi_mses, mos_mses, htsi_SNRs_obj, mos_SNRs_obj, htsi_mse_obj, mos_mse_obj,obj_max_S,NR_whole, NR_M_whole, ccd_noise, M_ccd_noise, Q_rois))
np.savetxt(folder+'Flux_SAMOS_LR_fixed4_'+str(radius)+'radius_'+str(t)+'s_'+str(mat_type)+str(order)+'.txt', data, delimiter='\t')    

print('File: Flux_data_labels_'+str(radius)+'radius_'+str(t)+'s_'+str(mat_type)+str(order)+'.txt saved.')
del fs 
del obj_total_S
del obj_avg_S
del Qs
del htsi_mses
del mos_mses
del htsi_SNRs_obj
del mos_SNRs_obj
del htsi_mse_obj
del mos_mse_obj
del obj_max_S
print('data variables cleared.')


mat_type = 'S'
order = 63
fs, obj_total_S, obj_avg_S, Qs, htsi_mses, mos_mses, htsi_SNRs_obj, mos_SNRs_obj, htsi_mse_obj, mos_mse_obj, obj_max_S, NR_whole, NR_M_whole, ccd_noise, M_ccd_noise, Q_rois = run_flux_simulations(sky, mat_type, order)
data = np.vstack((fs, obj_total_S, obj_avg_S, Qs, htsi_mses, mos_mses, htsi_SNRs_obj, mos_SNRs_obj, htsi_mse_obj, mos_mse_obj,obj_max_S,NR_whole, NR_M_whole, ccd_noise, M_ccd_noise, Q_rois))
np.savetxt(folder+'Flux_SAMOS_LR_fixed4_'+str(radius)+'radius_'+str(t)+'s_'+str(mat_type)+str(order)+'.txt', data, delimiter='\t')    

print('File: Flux_data_labels_'+str(radius)+'radius_'+str(t)+'s_'+str(mat_type)+str(order)+'.txt saved.')
del fs 
del obj_total_S
del obj_avg_S
del Qs
del htsi_mses
del mos_mses
del htsi_SNRs_obj
del mos_SNRs_obj
del htsi_mse_obj
del mos_mse_obj
del obj_max_S
print('data variables cleared.')




mat_type = 'S'
order = 103


fs, obj_total_S, obj_avg_S, Qs, htsi_mses, mos_mses, htsi_SNRs_obj, mos_SNRs_obj, htsi_mse_obj, mos_mse_obj, obj_max_S, NR_whole, NR_M_whole, ccd_noise, M_ccd_noise, Q_rois = run_flux_simulations(sky, mat_type, order)
data = np.vstack((fs, obj_total_S, obj_avg_S, Qs, htsi_mses, mos_mses, htsi_SNRs_obj, mos_SNRs_obj, htsi_mse_obj, mos_mse_obj,obj_max_S,NR_whole, NR_M_whole, ccd_noise, M_ccd_noise, Q_rois))
np.savetxt(folder+'Flux_SAMOS_LR_fixed4_'+str(radius)+'radius_'+str(t)+'s_'+str(mat_type)+str(order)+'.txt', data, delimiter='\t')    

del fs 
del obj_total_S
del obj_avg_S
del Qs
del htsi_mses
del mos_mses
del htsi_SNRs_obj
del mos_SNRs_obj
del htsi_mse_obj
del mos_mse_obj
del obj_max_S
print('data variables cleared.')
#################################### BEGIN THE NO MAX ADU ##############################

max_adu = 9000000
fs, obj_total_S, obj_avg_S, Qs, htsi_mses, mos_mses, htsi_SNRs_obj, mos_SNRs_obj, htsi_mse_obj, mos_mse_obj, obj_max_S, NR_whole, NR_M_whole, ccd_noise, M_ccd_noise, Q_rois = run_flux_simulations(sky, mat_type, order)
data = np.vstack((fs, obj_total_S, obj_avg_S, Qs, htsi_mses, mos_mses, htsi_SNRs_obj, mos_SNRs_obj, htsi_mse_obj, mos_mse_obj,obj_max_S,NR_whole, NR_M_whole, ccd_noise, M_ccd_noise, Q_rois))
np.savetxt(folder+'Flux_SAMOS_nomaxADU4_'+str(radius)+'radius_'+str(t)+'s_'+str(mat_type)+str(order)+'.txt', data, delimiter='\t')    
del fs 
del obj_total_S
del obj_avg_S
del Qs
del htsi_mses
del mos_mses
del htsi_SNRs_obj
del mos_SNRs_obj
del htsi_mse_obj
del mos_mse_obj
del obj_max_S
print('data variables cleared.')



max_adu = 9000000
mat_type = 'S'
order = 127

fs, obj_total_S, obj_avg_S, Qs, htsi_mses, mos_mses, htsi_SNRs_obj, mos_SNRs_obj, htsi_mse_obj, mos_mse_obj, obj_max_S, NR_whole, NR_M_whole, ccd_noise, M_ccd_noise, Q_rois = run_flux_simulations(sky, mat_type, order)
data = np.vstack((fs, obj_total_S, obj_avg_S, Qs, htsi_mses, mos_mses, htsi_SNRs_obj, mos_SNRs_obj, htsi_mse_obj, mos_mse_obj,obj_max_S,NR_whole, NR_M_whole, ccd_noise, M_ccd_noise, Q_rois))
np.savetxt(folder+'Flux_SAMOS_nomaxADU4_'+str(radius)+'radius_'+str(t)+'s_'+str(mat_type)+str(order)+'.txt', data, delimiter='\t')    

del fs 
del obj_total_S
del obj_avg_S
del Qs
del htsi_mses
del mos_mses
del htsi_SNRs_obj
del mos_SNRs_obj
del htsi_mse_obj
del mos_mse_obj
del obj_max_S
print('data variables cleared.')
#################################### BEGIN THE NO MAX ADU ##############################

mat_type = 'H'
order = 128
fs, obj_total_S, obj_avg_S, Qs, htsi_mses, mos_mses, htsi_SNRs_obj, mos_SNRs_obj, htsi_mse_obj, mos_mse_obj, obj_max_S, NR_whole, NR_M_whole, ccd_noise, M_ccd_noise, Q_rois = run_flux_simulations(sky, mat_type, order)
data = np.vstack((fs, obj_total_S, obj_avg_S, Qs, htsi_mses, mos_mses, htsi_SNRs_obj, mos_SNRs_obj, htsi_mse_obj, mos_mse_obj,obj_max_S,NR_whole, NR_M_whole, ccd_noise, M_ccd_noise, Q_rois))
np.savetxt(folder+'Flux_SAMOS_nomaxADU4_'+str(radius)+'radius_'+str(t)+'s_'+str(mat_type)+str(order)+'.txt', data, delimiter='\t')    
del fs 
del obj_total_S
del obj_avg_S
del Qs
del htsi_mses
del mos_mses
del htsi_SNRs_obj
del mos_SNRs_obj
del htsi_mse_obj
del mos_mse_obj
del obj_max_S
print('data variables cleared.')

mat_type = 'H'
order = 64

fs, obj_total_S, obj_avg_S, Qs, htsi_mses, mos_mses, htsi_SNRs_obj, mos_SNRs_obj, htsi_mse_obj, mos_mse_obj, obj_max_S, NR_whole, NR_M_whole, ccd_noise, M_ccd_noise, Q_rois = run_flux_simulations(sky, mat_type, order)
data = np.vstack((fs, obj_total_S, obj_avg_S, Qs, htsi_mses, mos_mses, htsi_SNRs_obj, mos_SNRs_obj, htsi_mse_obj, mos_mse_obj,obj_max_S,NR_whole, NR_M_whole, ccd_noise, M_ccd_noise, Q_rois))
np.savetxt(folder+'Flux_SAMOS_nomaxADU4_'+str(radius)+'radius_'+str(t)+'s_'+str(mat_type)+str(order)+'.txt', data, delimiter='\t')    
del fs 
del obj_total_S
del obj_avg_S
del Qs
del htsi_mses
del mos_mses
del htsi_SNRs_obj
del mos_SNRs_obj
del htsi_mse_obj
del mos_mse_obj
del obj_max_S
print('data variables cleared.')

mat_type = 'S'
order = 63
fs, obj_total_S, obj_avg_S, Qs, htsi_mses, mos_mses, htsi_SNRs_obj, mos_SNRs_obj, htsi_mse_obj, mos_mse_obj, obj_max_S, NR_whole, NR_M_whole, ccd_noise, M_ccd_noise, Q_rois = run_flux_simulations(sky, mat_type, order)
data = np.vstack((fs, obj_total_S, obj_avg_S, Qs, htsi_mses, mos_mses, htsi_SNRs_obj, mos_SNRs_obj, htsi_mse_obj, mos_mse_obj,obj_max_S,NR_whole, NR_M_whole, ccd_noise, M_ccd_noise, Q_rois))
np.savetxt(folder+'Flux_SAMOS_nomaxADU4_'+str(radius)+'radius_'+str(t)+'s_'+str(mat_type)+str(order)+'.txt', data, delimiter='\t')    
del fs 
del obj_total_S
del obj_avg_S
del Qs
del htsi_mses
del mos_mses
del htsi_SNRs_obj
del mos_SNRs_obj
del htsi_mse_obj
del mos_mse_obj
del obj_max_S
print('data variables cleared.')

########################################################################################################################################################
lam0, lam = 400.0,800.0
nm_sample = 2
slit_width = 1
t= 90 # exposure time
DMD_contrast = 2000
n = 1024 # 2d img array size (nxn)
########
pix_size = 15 * 10**-6
full_well_cap = 350000 # Pixel full well capactiy in e- 
bitdepth = 16
max_adu = int(2**bitdepth -1)
#max_adu  = 9000000
gain = 5 #ADU/e-
dt = 0.0008 # 0.0002  # The thermal dark signal in e-/pix/sec at the operating temp of -90C
rn = 5 # e-, readout noise
################################################### Other #############################################################
mat_type = 'S'
order = 127

radius = (order/2)*0.90
sky_orig, wls = SIM.sim_data_flux_basic(lam0, lam, nm_sample, n, radius)
sky_orig, T = SIM.apply_thruput(sky_orig, wls, 'SAMOS_low')
sky_orig = sky_orig*t
sky = SIM.apply_psf(sky_orig, 'moffat', 4) # 4 ~ 0.4" seeing
if np.min(sky) < 0:
    sky = sky+(-1.1*(np.min(sky)))
if np.min(sky) == 0:
    sky = sky+1
#################################

fs, obj_total_S, obj_avg_S, Qs, htsi_mses, mos_mses, htsi_SNRs_obj, mos_SNRs_obj, htsi_mse_obj, mos_mse_obj, obj_max_S, NR_whole, NR_M_whole, ccd_noise, M_ccd_noise, Q_rois = run_flux_simulations(sky, mat_type, order)
data = np.vstack((fs, obj_total_S, obj_avg_S, Qs, htsi_mses, mos_mses, htsi_SNRs_obj, mos_SNRs_obj, htsi_mse_obj, mos_mse_obj,obj_max_S,NR_whole, NR_M_whole, ccd_noise, M_ccd_noise, Q_rois))
np.savetxt(folder+'Flux_SAMOS_other4_'+str(radius)+'radius_'+str(t)+'s_'+str(mat_type)+str(order)+'.txt', data, delimiter='\t')    

data_labels = np.vstack(('fs', 'obj_total_S', 'obj_avg_S', 'Qs', 'htsi_mses', 'mos_mses', 'htsi_SNRs_obj', 'mos_SNRs_obj', 'htsi_mse_obj', 'mos_mse_obj','Max signal S (obj)', 'HTSI noise ratio', 'MOS noise ratio','HTSI CCD noise','MOS CCD noise','Q for the roi'))
del fs 
del obj_total_S
del obj_avg_S
del Qs
del htsi_mses
del mos_mses
del htsi_SNRs_obj
del mos_SNRs_obj
del htsi_mse_obj
del mos_mse_obj
del obj_max_S
print('data variables cleared.')
#%
mat_type = 'H'
order = 128
fs, obj_total_S, obj_avg_S, Qs, htsi_mses, mos_mses, htsi_SNRs_obj, mos_SNRs_obj, htsi_mse_obj, mos_mse_obj, obj_max_S, NR_whole, NR_M_whole, ccd_noise, M_ccd_noise, Q_rois = run_flux_simulations(sky, mat_type, order)
data = np.vstack((fs, obj_total_S, obj_avg_S, Qs, htsi_mses, mos_mses, htsi_SNRs_obj, mos_SNRs_obj, htsi_mse_obj, mos_mse_obj,obj_max_S,NR_whole, NR_M_whole, ccd_noise, M_ccd_noise, Q_rois))
np.savetxt(folder+'Flux_SAMOS_othe4r_'+str(radius)+'radius_'+str(t)+'s_'+str(mat_type)+str(order)+'.txt', data, delimiter='\t')    

data_labels = np.vstack(('fs', 'obj_total_S', 'obj_avg_S', 'Qs', 'htsi_mses', 'mos_mses', 'htsi_SNRs_obj', 'mos_SNRs_obj', 'htsi_mse_obj', 'mos_mse_obj','Max signal S (obj)', 'HTSI noise ratio', 'MOS noise ratio','HTSI CCD noise','MOS CCD noise','Q for the roi'))
del fs 
del obj_total_S
del obj_avg_S
del Qs
del htsi_mses
del mos_mses
del htsi_SNRs_obj
del mos_SNRs_obj
del htsi_mse_obj
del mos_mse_obj
del obj_max_S
print('data variables cleared.')

mat_type = 'S'
order = 35


fs, obj_total_S, obj_avg_S, Qs, htsi_mses, mos_mses, htsi_SNRs_obj, mos_SNRs_obj, htsi_mse_obj, mos_mse_obj, obj_max_S, NR_whole, NR_M_whole, ccd_noise, M_ccd_noise, Q_rois = run_flux_simulations(sky, mat_type, order)
data = np.vstack((fs, obj_total_S, obj_avg_S, Qs, htsi_mses, mos_mses, htsi_SNRs_obj, mos_SNRs_obj, htsi_mse_obj, mos_mse_obj,obj_max_S,NR_whole, NR_M_whole, ccd_noise, M_ccd_noise, Q_rois))
np.savetxt(folder+'Flux_SAMOS_other4_'+str(radius)+'radius_'+str(t)+'s_'+str(mat_type)+str(order)+'.txt', data, delimiter='\t')    

data_labels = np.vstack(('fs', 'obj_total_S', 'obj_avg_S', 'Qs', 'htsi_mses', 'mos_mses', 'htsi_SNRs_obj', 'mos_SNRs_obj', 'htsi_mse_obj', 'mos_mse_obj','Max signal S (obj)', 'HTSI noise ratio', 'MOS noise ratio','HTSI CCD noise','MOS CCD noise','Q for the roi'))
del fs 
del obj_total_S
del obj_avg_S
del Qs
del htsi_mses
del mos_mses
del htsi_SNRs_obj
del mos_SNRs_obj
del htsi_mse_obj
del mos_mse_obj
del obj_max_S
print('data variables cleared.')
#%

mat_type = 'S'
order = 83


fs, obj_total_S, obj_avg_S, Qs, htsi_mses, mos_mses, htsi_SNRs_obj, mos_SNRs_obj, htsi_mse_obj, mos_mse_obj, obj_max_S, NR_whole, NR_M_whole, ccd_noise, M_ccd_noise, Q_rois = run_flux_simulations(sky, mat_type, order)
data = np.vstack((fs, obj_total_S, obj_avg_S, Qs, htsi_mses, mos_mses, htsi_SNRs_obj, mos_SNRs_obj, htsi_mse_obj, mos_mse_obj,obj_max_S,NR_whole, NR_M_whole, ccd_noise, M_ccd_noise, Q_rois))
np.savetxt(folder+'Flux_SAMOS_other4_'+str(radius)+'radius_'+str(t)+'s_'+str(mat_type)+str(order)+'.txt', data, delimiter='\t')    

data_labels = np.vstack(('fs', 'obj_total_S', 'obj_avg_S', 'Qs', 'htsi_mses', 'mos_mses', 'htsi_SNRs_obj', 'mos_SNRs_obj', 'htsi_mse_obj', 'mos_mse_obj','Max signal S (obj)', 'HTSI noise ratio', 'MOS noise ratio','HTSI CCD noise','MOS CCD noise','Q for the roi'))
del fs 
del obj_total_S
del obj_avg_S
del Qs
del htsi_mses
del mos_mses
del htsi_SNRs_obj
del mos_SNRs_obj
del htsi_mse_obj
del mos_mse_obj
del obj_max_S
print('data variables cleared.')
#%

mat_type = 'H'
order = 64

fs, obj_total_S, obj_avg_S, Qs, htsi_mses, mos_mses, htsi_SNRs_obj, mos_SNRs_obj, htsi_mse_obj, mos_mse_obj, obj_max_S, NR_whole, NR_M_whole, ccd_noise, M_ccd_noise, Q_rois = run_flux_simulations(sky, mat_type, order)
data = np.vstack((fs, obj_total_S, obj_avg_S, Qs, htsi_mses, mos_mses, htsi_SNRs_obj, mos_SNRs_obj, htsi_mse_obj, mos_mse_obj,obj_max_S,NR_whole, NR_M_whole, ccd_noise, M_ccd_noise, Q_rois))
np.savetxt(folder+'Flux_SAMOS_other_'+str(radius)+'radius_'+str(t)+'s_'+str(mat_type)+str(order)+'.txt', data, delimiter='\t')    

data_labels = np.vstack(('fs', 'obj_total_S', 'obj_avg_S', 'Qs', 'htsi_mses', 'mos_mses', 'htsi_SNRs_obj', 'mos_SNRs_obj', 'htsi_mse_obj', 'mos_mse_obj','Max signal S (obj)', 'HTSI noise ratio', 'MOS noise ratio','HTSI CCD noise','MOS CCD noise','Q for the roi'))
del fs 
del obj_total_S
del obj_avg_S
del Qs
del htsi_mses
del mos_mses
del htsi_SNRs_obj
del mos_SNRs_obj
del htsi_mse_obj
del mos_mse_obj
del obj_max_S
print('data variables cleared.')
#%

mat_type = 'S'
order = 71


fs, obj_total_S, obj_avg_S, Qs, htsi_mses, mos_mses, htsi_SNRs_obj, mos_SNRs_obj, htsi_mse_obj, mos_mse_obj, obj_max_S, NR_whole, NR_M_whole, ccd_noise, M_ccd_noise, Q_rois = run_flux_simulations(sky, mat_type, order)
data = np.vstack((fs, obj_total_S, obj_avg_S, Qs, htsi_mses, mos_mses, htsi_SNRs_obj, mos_SNRs_obj, htsi_mse_obj, mos_mse_obj,obj_max_S,NR_whole, NR_M_whole, ccd_noise, M_ccd_noise, Q_rois))
np.savetxt(folder+'Flux_SAMOS_other_'+str(radius)+'radius_'+str(t)+'s_'+str(mat_type)+str(order)+'.txt', data, delimiter='\t')    

data_labels = np.vstack(('fs', 'obj_total_S', 'obj_avg_S', 'Qs', 'htsi_mses', 'mos_mses', 'htsi_SNRs_obj', 'mos_SNRs_obj', 'htsi_mse_obj', 'mos_mse_obj','Max signal S (obj)', 'HTSI noise ratio', 'MOS noise ratio','HTSI CCD noise','MOS CCD noise','Q for the roi'))
del fs 
del obj_total_S
del obj_avg_S
del Qs
del htsi_mses
del mos_mses
del htsi_SNRs_obj
del mos_SNRs_obj
del htsi_mse_obj
del mos_mse_obj
del obj_max_S
print('data variables cleared.')
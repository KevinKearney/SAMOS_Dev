# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 21:16:20 2022

@author: Kate
"""
import numpy as np
import sys
sys.path.append('C:/Users/Kate/Documents/hadamard/sandbox')
import matplotlib.pyplot as plt

from hadamard_class_v2 import *
HTSI = HTSI_Models2()
from data_sim_class import *
SIM = data_sim()


def temporal_PSF(sky, PSF,x,y,l):
# Purpose of this function is to create a set of flattened sky cubes with varrying parameters (PSF, jitter, and/or intensity)
    r = [0]
    if PSF[0] == 'Y':
        flat_skies = np.ndarray.flatten(sky)
        for k in range(0,np.int(PSF[3])): # replace the 3 with PSF[1]
            l = np.random.randint(np.int(PSF[1]),np.int(PSF[2])) # replace intervals with PSF[2,3]
            sky2 = SIM.apply_psf(sky, 'moffat',l)
            flat_sky = np.ndarray.flatten(sky2)
            flat_skies = np.vstack((flat_skies,flat_sky))
            r = np.append(r,l)
            del sky2
            print ('PSF iteration'+str(k+1)+' of '+str(PSF[3]))
    return flat_skies, r

def temporal_jitter_post_psf(flat_skies, Jitter, x, y, l, sky_edge):
    #x,y,l = np.shape(sky)[0], np.shape(sky)[1], np.shape(sky)[2] # x,y, and wavelength (l) dimensions of the input array
    xis = []
    yis = []
    # Data for the Jitter variations
    pix = np.int(Jitter[1])#np.int(jitter[1])
    p = 10*pix
    p2 = np.int(p/2)
    y1,y2 = p2, p2+y
    x1,x2 = p2, p2+x
    #sky_edge = sky_orig[0,0,:]
    padded_sky = np.ones((x+p,y+p,l))
    padded_sky = padded_sky*np.mean(sky_edge)
    for j in range(0,np.shape(flat_skies)[0]):
        sky_unflat = np.reshape(flat_skies[j,:],(x,y,l))
        padded_sky[x1:x2,y1:y2] = sky_unflat
    
        xi = np.random.randint(-pix,pix)
        yi = np.random.randint(-pix,pix)
        new_sky = padded_sky[x1+xi:x2+xi,y1+yi:y2+yi,:]
        xis = np.append(xis,xi)
        yis = np.append(yis, yi)
        new_skyF = np.ndarray.flatten(new_sky)
        flat_skies[j,:] = np.ndarray.flatten(new_skyF)
    return flat_skies, xis, yis
            
def temporal_jitter(sky, Jitter, sky_edge):
    x,y,l = np.shape(sky)[0], np.shape(sky)[1], np.shape(sky)[2] # x,y, and wavelength (l) dimensions of the input array
    xis = []
    yis = []
    pix = np.int(Jitter[1])
    p = 10*pix
    p2 = np.int(p/2)
    y1,y2 = p2, p2+y
    x1,x2 = p2, p2+x
    #sky_edge = sky_orig[0,0,:]
    padded_sky = np.ones((x+p,y+p,l))
    padded_sky = padded_sky*np.mean(sky_edge)
    flat_skies = np.ndarray.flatten(sky)
    for j in range(0,np.int(Jitter[2])):
        padded_sky[x1:x2,y1:y2] = sky
        xi = np.random.randint(-pix,pix)
        yi = np.random.randint(-pix,pix)
        new_sky = padded_sky[x1+xi:x2+xi,y1+yi:y2+yi,:]
        xis = np.append(xis,xi)
        yis = np.append(yis, yi)
        new_skyF = np.ndarray.flatten(new_sky)
        flat_skies = np.vstack((flat_skies,new_skyF))
    return flat_skies, xis, yis

def temporal_I_post(flat_skies, Intensity, x, y, l):        
    Is = []
    I = np.int(Intensity[1])
    for k in range(0,np.shape(flat_skies)[0]):
        Ii = np.random.randint(0,I) # change 5 to be Intensity[1]
        sky_unflat = np.reshape(flat_skies[k,:],(x,y,l))
        II = (100-Ii)/100
        new_sky = sky_unflat*II
        Is = np.append(Is, II)
        flat_skies[k,:] = np.ndarray.flatten(new_sky)
    return flat_skies, Is
               
def temporal_I(sky, Intensity):
    Is = []
    I = np.int(Intensity[1])
    flat_skies = np.ndarray.flatten(sky)
    for k in range(0,np.int(Intensity[2])):
        Ii = np.random.randint(0,I) # change 5 to be Intensity[1]
        II = (100-Ii)/100
        new_sky = sky*II
        Is = np.append(Is, II)
        flat_skies = np.vstack((flat_skies,new_sky))
    return flat_skies, Is

def all_temporal(sky, PSF, Jitter, Intensity, x,y,l, sky_edge):
    flat_skies, r = temporal_PSF(sky, PSF,x,y,l)
    print('PSF variations done.')
    flat_skies, xis, yis = temporal_jitter_post_psf(flat_skies, Jitter, x, y, l, sky_edge)
    print('Jitter variations done.')
    flat_skies, Is = temporal_I_post(flat_skies, Intensity, x, y, l)
    print('Intensity variations done.')
    return flat_skies, r, xis, yis, Is

#%% Variable Parameters
folder = 'C:/Users/Kate/Documents/hadamard/'

lam0, lam = 300.0, 850.0 # wavelength range in nm
nm_sample = 2 # wavelength resolution in nm
#
dt = 0.0002 # 0.0002  # The thermal dark signal in e-/pix/sec at the operating temp of -90C
rn = 5 # e-, readout noise
D = 4 #Telescope diameter [m]
A = np.pi*(D/2)**2 #Area of the telesscope [m] (make sure to account for obstructions)
t = 90 # exposure time
DMD_contrast = 2000 # DMD contrast ratio

# Other parameters
n = 1024 # 2d img array size (nxn)
pix_size = 15 * 10**-6
full_well_cap = 100000 # Pixel full well capactiy in e- 
gain = 5.0 #ADU/e-
bitdepth = 16
max_adu = np.int(2**bitdepth -1)


#sky, wls = SIM.sim_sky_stellar1(lam0,lam,nm_sample) # Generates the sky scene
#obj = 'ngc6' #stellar 1
#flux_factor = 0.05

#sky, wls = SIM.sim_sky_stellar2(lam0,lam,nm_sample) # Generates the sky scene
#obj = 'm79' # stellar 2

sky, wls = SIM.sim_sky_nebula(lam0,lam,nm_sample) # Generates the sky scene
obj = 'nebula'
flux_factor = 0.5
#lux_factor = 0.15
#
#sky, wls = SIM.sim_sky_cygnus(lam0,lam,nm_sample) # Generates the sky scene#
#obj = 'cygnus'
#flux_factor = 0.00001


#sky, wls = SIM.sim_sky_galaxy(lam0,lam,nm_sample) # Generates the sky scene
#obj = 'galaxy'
#flux_factor = 0.055

n = np.shape(sky[0])
sky, T = SIM.apply_thruput(sky, wls, 'SAMOS_low') # Applies telescope throughput
sky = SIM.vary_psf_along_wl(sky,D,wls) # adds in the PSF (as a function of wavelength)
sky = sky*t*A* flux_factor


noise_ratio = np.sqrt(np.mean(sky))/(((dt*t)**2)+(rn**2))
#%%Untouched HTSI 1
mat_type = 'H'
order = 128
slit_width = 1 # slit width in micromirrors

x1,x2 = np.int((order/2)-20), np.int((order/2)+20)# Coordinates to sample in output cubes
y1,y2 = x1,x2
Xp = order*slit_width*1.2
Yp = order*slit_width*1.2 # 4112 # number of pixels in y dimension
num_pix = Xp * Yp # total number of pixels 
CCD = [t, dt, num_pix, rn, gain, max_adu]

if mat_type == 'H':
    htsi_data, sky_ADU, htsi_mse,ccdN, phoN = HTSI.htsi(sky, order, slit_width, DMD_contrast, CCD)
if mat_type == 'S':
    htsi_data, sky_ADU, htsi_mse, ccdN, phoN = HTSI.S_htsi(sky, order, slit_width, DMD_contrast, CCD)
mos_data, sky_ADU_mos, mos_mse,ccdN, phoN = HTSI.MOS(sky, order, slit_width, DMD_contrast, CCD)
Q_whole = np.divide((np.sqrt(np.median(mos_mse))),(np.sqrt(np.median(htsi_mse))))
htsi_SNR, mos_SNR, h_spec, m_spec, s_spec, Q, h_mse, m_mse = HTSI.SNR_Q_ROI(x1,x2,y1,y2, htsi_data, mos_data, sky_ADU)
total_s = np.sum(s_spec)
avg_s = np.mean(sky_ADU)

data = np.transpose(np.vstack((Q_whole,Q,h_mse, m_mse, avg_s)))
np.savetxt(folder+'HTSI_results_slitwidth_'+str(slit_width)+'_'+str(mat_type)+str(order)+'_'+str(obj)+'.txt', data, delimiter='\t')    
#%Untouched HTSI 2
mat_type = 'H'
order = 128
slit_width = 3 # slit width in micromirrors

x1,x2 = np.int((order/2)-20), np.int((order/2)+20)# Coordinates to sample in output cubes
y1,y2 = x1,x2
Xp = order*slit_width*1.2
Yp = order*slit_width*1.2 # 4112 # number of pixels in y dimension
num_pix = Xp * Yp # total number of pixels 
CCD = [t, dt, num_pix, rn, gain, max_adu]

if mat_type == 'H':
    htsi_data2, sky_ADU2, htsi_mse2,ccdN, phoN = HTSI.htsi(sky, order, slit_width, DMD_contrast, CCD)
if mat_type == 'S':
    htsi_data2, sky_ADU2, htsi_mse2, ccdN, phoN = HTSI.S_htsi(sky, order, slit_width, DMD_contrast, CCD)
mos_data2, sky_ADU_mos2, mos_mse2,ccdN, phoN = HTSI.MOS(sky, order, slit_width, DMD_contrast, CCD)
Q_whole2 = np.divide((np.sqrt(np.median(mos_mse2))),(np.sqrt(np.median(htsi_mse2))))
htsi_SNR, mos_SNR, h_spec, m_spec, s_spec, Q2, h_mse, m_mse = HTSI.SNR_Q_ROI(x1,x2,y1,y2, htsi_data2, mos_data2, sky_ADU2)
total_s = np.sum(s_spec)
avg_s2 = np.mean(sky_ADU2)

data = np.transpose(np.vstack((Q_whole2,Q2,h_mse, m_mse, avg_s2)))
np.savetxt(folder+'HTSI_results_slitwidth_'+str(slit_width)+'_'+str(mat_type)+str(order)+'_'+str(obj)+'.txt', data, delimiter='\t')    
#% HTSI 3
mat_type = 'H'
order = 64
slit_width = 4 # slit width in micromirrors

x1,x2 = np.int((order/2)-20), np.int((order/2)+20)# Coordinates to sample in output cubes
y1,y2 = x1,x2
Xp = order*slit_width*1.2
Yp = order*slit_width*1.2 # 4112 # number of pixels in y dimension
num_pix = Xp * Yp # total number of pixels 
CCD = [t, dt, num_pix, rn, gain, max_adu]


if mat_type == 'H':
    htsi_data3, sky_ADU3, htsi_mse3,ccdN, phoN = HTSI.htsi(sky, order, slit_width, DMD_contrast, CCD)
if mat_type == 'S':
    htsi_data3, sky_ADU3, htsi_mse3, ccdN, phoN = HTSI.S_htsi(sky, order, slit_width, DMD_contrast, CCD)
mos_data3, sky_ADU_mos3, mos_mse3,ccdN, phoN = HTSI.MOS(sky, order, slit_width, DMD_contrast, CCD)
Q_whole3 = np.divide((np.sqrt(np.median(mos_mse3))),(np.sqrt(np.median(htsi_mse3))))
htsi_SNR, mos_SNR, h_spec, m_spec, s_spec, Q3, h_mse, m_mse = HTSI.SNR_Q_ROI(x1,x2,y1,y2, htsi_data3, mos_data3, sky_ADU3)
total_s = np.sum(s_spec)
avg_s3 = np.mean(sky_ADU3)

data = np.transpose(np.vstack((Q_whole3,Q3,h_mse, m_mse, avg_s3)))
np.savetxt(folder+'HTSI_results_slitwidth_'+str(slit_width)+'_'+str(mat_type)+str(order)+'_'+str(obj)+'.txt', data, delimiter='\t')    
#% HTSI 4
mat_type = 'H'
order = 256
slit_width = 1 # slit width in micromirrors

x1,x2 = np.int((order/2)-20), np.int((order/2)+20)# Coordinates to sample in output cubes
y1,y2 = x1,x2
Xp = order*slit_width*1.2
Yp = order*slit_width*1.2 # 4112 # number of pixels in y dimension
num_pix = Xp * Yp # total number of pixels 
CCD = [t, dt, num_pix, rn, gain, max_adu]


if mat_type == 'H':
    htsi_data4, sky_ADU4, htsi_mse4,ccdN, phoN = HTSI.htsi(sky, order, slit_width, DMD_contrast, CCD)
if mat_type == 'S':
    htsi_data4, sky_ADU4, htsi_mse4, ccdN, phoN = HTSI.S_htsi(sky, order, slit_width, DMD_contrast, CCD)
mos_data4, sky_ADU_mos4, mos_mse4,ccdN, phoN = HTSI.MOS(sky, order, slit_width, DMD_contrast, CCD)
Q_whole4 = np.divide((np.sqrt(np.median(mos_mse4))),(np.sqrt(np.median(htsi_mse2))))
htsi_SNR, mos_SNR, h_spec, m_spec, s_spec, Q4, h_mse, m_mse = HTSI.SNR_Q_ROI(x1,x2,y1,y2, htsi_data4, mos_data4, sky_ADU4)
total_s = np.sum(s_spec)
avg_s4 = np.mean(sky_ADU4)

data = np.transpose(np.vstack((Q_whole4,Q4,h_mse, m_mse, avg_s4)))
np.savetxt(folder+'HTSI_results_slitwidth_'+str(slit_width)+'_'+str(mat_type)+str(order)+'_'+str(obj)+'.txt', data, delimiter='\t')    
#% HTSI 5
mat_type = 'H'
order = 64
slit_width = 2 # slit width in micromirrors

x1,x2 = np.int((order/2)-20), np.int((order/2)+20)# Coordinates to sample in output cubes
y1,y2 = x1,x2
Xp = order*slit_width*1.2
Yp = order*slit_width*1.2 # 4112 # number of pixels in y dimension
num_pix = Xp * Yp # total number of pixels 
CCD = [t, dt, num_pix, rn, gain, max_adu]


if mat_type == 'H':
    htsi_data5, sky_ADU5, htsi_mse5,ccdN, phoN = HTSI.htsi(sky, order, slit_width, DMD_contrast, CCD)
if mat_type == 'S':
    htsi_data5, sky_ADU5, htsi_mse5, ccdN, phoN = HTSI.S_htsi(sky, order, slit_width, DMD_contrast, CCD)
mos_data5, sky_ADU_mos5, mos_mse5,ccdN, phoN = HTSI.MOS(sky, order, slit_width, DMD_contrast, CCD)
Q_whole5 = np.divide((np.sqrt(np.median(mos_mse5))),(np.sqrt(np.median(htsi_mse5))))
htsi_SNR, mos_SNR, h_spec, m_spec, s_spec, Q5, h_mse, m_mse = HTSI.SNR_Q_ROI(x1,x2,y1,y2, htsi_data5, mos_data5, sky_ADU2)
total_s = np.sum(s_spec)
avg_s5 = np.mean(sky_ADU5)

data = np.transpose(np.vstack((Q_whole2,Q2,h_mse, m_mse, avg_s2)))
np.savetxt(folder+'HTSI_results_slitwidth_'+str(slit_width)+'_'+str(mat_type)+str(order)+'_'+str(obj)+'.txt', data, delimiter='\t')    
#% HTSI 6
mat_type = 'H'
order = 64
slit_width = 1 # slit width in micromirrors

x1,x2 = np.int((order/2)-20), np.int((order/2)+20)# Coordinates to sample in output cubes
y1,y2 = x1,x2
Xp = order*slit_width*1.2
Yp = order*slit_width*1.2 # 4112 # number of pixels in y dimension
num_pix = Xp * Yp # total number of pixels 
CCD = [t, dt, num_pix, rn, gain, max_adu]


if mat_type == 'H':
    htsi_data6, sky_ADU6, htsi_mse6,ccdN, phoN = HTSI.htsi(sky, order, slit_width, DMD_contrast, CCD)
if mat_type == 'S':
    htsi_data6, sky_ADU6, htsi_mse6, ccdN, phoN = HTSI.S_htsi(sky, order, slit_width, DMD_contrast, CCD)
mos_data6, sky_ADU_mos6, mos_mse6,ccdN, phoN = HTSI.MOS(sky, order, slit_width, DMD_contrast, CCD)
Q_whole6 = np.divide((np.sqrt(np.median(mos_mse6))),(np.sqrt(np.median(htsi_mse6))))
htsi_SNR, mos_SNR, h_spec, m_spec, s_spec, Q6, h_mse, m_mse = HTSI.SNR_Q_ROI(x1,x2,y1,y2, htsi_data6, mos_data6, sky_ADU2)
total_s = np.sum(s_spec)
avg_s6 = np.mean(sky_ADU6)

data = np.transpose(np.vstack((Q_whole6,Q6,h_mse, m_mse, avg_s6)))
np.savetxt(folder+'HTSI_results_slitwidth_'+str(slit_width)+'_'+str(mat_type)+str(order)+'_'+str(obj)+'.txt', data, delimiter='\t')    

#%% FOR REGULAR HTSI
wl_slice = 67

plt.subplot(2,3,1)
plt.imshow(htsi_data[:,:,wl_slice],clim = (0,250),cmap = 'spring')
plt.title('H-128, sw=1')
plt.axis('off')
plt.subplot(2,3,2)
plt.imshow(htsi_data2[:,:,wl_slice],clim = (0,250),cmap = 'gray')
plt.title('H-128, sw=2')
plt.axis('off')
plt.subplot(2,3,3)
plt.imshow(htsi_data3[:,:,wl_slice],clim = (0,250),cmap = 'cool')
plt.title('H-128, sw=3')
plt.axis('off')
plt.subplot(2,4,5)
plt.imshow(htsi_data4[:,:,wl_slice],clim = (0,250),cmap = 'viridis')
plt.title('H-256, sw=1')
plt.axis('off')
plt.subplot(2,5,4)
plt.imshow(htsi_data5[:,:,wl_slice],clim = (0,250),cmap = 'plasma')
plt.title('S-64, sw=2')
plt.axis('off')
plt.subplot(2,6,3)
plt.imshow(htsi_data6[:,:,wl_slice],clim = (0,250),cmap = 'cividis')
plt.title('S-64, sw=3')
plt.axis('off')
#%% MOS SPEC
wl_slice = 67

plt.subplot(2,3,1)
plt.imshow(mos_data[:,:,wl_slice],clim = (0,350),cmap = 'inferno')
plt.title('H-128, sw=1')
plt.axis('off')
plt.subplot(2,3,2)
plt.imshow(mos_data2[:,:,wl_slice],clim = (0,350),cmap = 'inferno')
plt.title('H-128, sw=2')
plt.axis('off')
plt.subplot(2,3,6)
plt.imshow(mos_data3[:,:,wl_slice],clim = (0,350),cmap = 'inferno')
plt.title('H-64, sw=1')
plt.axis('off')
plt.subplot(2,3,5)
plt.imshow(mos_data5[:,:,wl_slice],clim = (0,350),cmap = 'inferno')
plt.title('H-64, sw=2')
plt.axis('off')
plt.subplot(2,3,4)
plt.imshow(mos_data4[:,:,wl_slice],clim = (0,350),cmap = 'inferno')
plt.title('S-127, sw=')
plt.axis('off')
plt.subplot(2,3,3)
plt.imshow(mos_data6[:,:,wl_slice],clim = (0,350),cmap = 'inferno')
plt.title('S-103, sw=')
plt.axis('off')

#%% FIGURE SPECTRA
xs, ys = 55,41# spectrum points

sky_spec= sky_ADU[xs,ys,:] * (1/gain) *(1/t)
H_spec =htsi_data[xs,ys,:] * (1/gain) *(1/t)
LS_spec = mos_data[xs,ys,:] * (1/gain) *(1/t)
plt.plot(wls,mos_data[xs,ys,:], color = 'pink')
plt.plot(wls,htsi_data[xs,ys,:], color = 'darkorchid')
plt.plot(wls,sky_ADU[xs,ys,:], color = 'black')
#plt.plot(wls,htsi_data2[xs,ys,:], color = 'darkorchid')
#plt.plot(wls,mos_data2[xs,ys,:], color = 'darkorchid')

plt.legend(loc='best')
plt.xlabel('Wavelength [nm]')
plt.ylabel('Flux [e-/sec/nm]')
plt.title('Total Signal ')

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 16}

plt.rc('font', **font)
plt.subplot(1,5,1)
plt.imshow(sky_ADU[b,:,:],clim = (0,80),cmap = 'inferno')
plt.axis('off')
plt.subplot(1,5,2)
plt.imshow(htsi_data[b,:,:],clim = (0,80),cmap = 'inferno')
plt.axis('off')
plt.subplot(1,5,3)
plt.imshow(mos_data[b,:,:],clim = (0,80),cmap = 'inferno')
plt.axis('off')
plt.subplot(1,5,4)
plt.imshow(htsi_data2[b,:,:],clim = (0,80),cmap = 'inferno')
plt.axis('off')
plt.subplot(1,5,5)
plt.imshow(mos_data2[b,:,:],clim = (0,80),cmap = 'inferno')
plt.axis('off')
#
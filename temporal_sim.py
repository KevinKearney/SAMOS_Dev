# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 21:16:20 2022

@author: Kate
"""
import numpy as np
import sys
sys.path.append('C:/Users/Kate/Documents/hadamard/sandbox')
import matplotlib.pyplot as plt
#from skimage import structural_similarity as ssim
#from skimage.metrics import mean_squared_error

from hadamard_class import *
HTSI = HTSI_Models()
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
mat_type = 'S'
order = 127
flux_factor = 0.03 # Adjusts the spectral flux across all wavelengths
Jitter = ['Y','1','5'] # Y or N, # pixels to jitter, # of cubes to generate
PSF = ['Y','3','6','5'] # Y or N, # max psf variability, # of cubes to generate
# PSF ranges: 3 = ~0.3", SAMOS best. 5 =~ 0.5" SAMOS avg. 10 = ~1", typical avg observatory
Intensity = ['N','2','5'] # Y or N, # max % of Intensity variations, # of cubes to generate
#
lam0, lam = 400.0, 700.0 # wavelength range in nm
nm_sample = 1 # wavelength resolution in nm
slit_width = 1 # slit width in micromirrors
#
dt = 0.0002 # 0.0002  # The thermal dark signal in e-/pix/sec at the operating temp of -90C
rn = 5 # e-, readout noise
#
D = 4 #Telescope diameter [m]
A = np.pi*(D/2)**2 #Area of the telesscope [m] (make sure to account for obstructions)
t = 90 # exposure time
DMD_contrast = 2000 # DMD contrast ratio
x1,x2 = np.int((order/2)-25), np.int((order/2)+25)# Coordinates to sample in output cubes
y1,y2 = x1,x2

# Other parameters
n = 1024 # 2d img array size (nxn)
pix_size = 15 * 10**-6
full_well_cap = 100000 # Pixel full well capactiy in e- 
gain = 4.0 #ADU/e-
bitdepth = 16
max_adu = np.int(2**bitdepth -1)
Xp = order*slit_width*1.2
Yp = order*slit_width*1.2 # 4112 # number of pixels in y dimension
num_pix = Xp * Yp # total number of pixels 
CCD = [t, dt, num_pix, rn, gain, max_adu]

#% Simulate the sky spectrum -- Change the function to modify what the input scene is
sky, wls = SIM.sim_sky_nebula(lam0,lam,nm_sample) # Generates the sky scene
obj = 'nebula' #object name reference tag
flux_factor = 0.015 #0.055 % Actually just a made up factor to scale the intensity
#%% Finding magnitude (quick fix needs to be checked)
sky_0 = sky*flux_factor
vw1, vw2 = (550-(86/2))+1, (550+(86/2)+1)
bw1, bw2 = (650-(132/2)), (650+(132/2))
v1, v2 = np.where(wls == vw1)[0][0], np.where(wls == vw2)[0][0]
#b1, b2 = np.where(wls == bw1)[0][0], np.where(wls == bw2)[0][0]

V_band_data = sky_0[100:150,100:150,v1:v2] 
#R_band_data = sky_0[100:150,100:150,b1:b2]
V_meanF = np.mean(V_band_data)*(1/200000) #1/100,000 * nm_sample to convert to photons/sec/cm^2/Angstrom
#R_meanF = np.mean(R_band_data)*(1/200000) #1/100,000 * nm_sample to convert to photons/sec/cm^2/Angstrom

mV = -2.5*np.log10(np.divide(V_meanF,995.5))
#mR = -2.5*np.log10(np.divide(R_meanF,702.0))

#%% Creating the input sky data cube
n = np.shape(sky[0])

sky, T = SIM.apply_thruput(sky, wls, 'SAMOS_low') # Applies telescope throughput
sky = SIM.vary_psf_along_wl(sky,D,wls) # adds in the PSF (as a function of wavelength)
sky = sky*t*A* flux_factor
#sky = sky*0.0001
sky_edge = sky[0,0,:]
x,y,l = np.shape(sky)[0], np.shape(sky)[1], np.shape(sky)[2] # x,y, and wavelength (l) dimensions of the input array
sky_shape = np.shape(sky)

noise_ratio = np.sqrt(np.mean(sky))/(((dt*t)**2)+(rn**2))
sky1 = SIM.apply_psf(sky, 'moffat',3)

#%% Regular HTSI (no temporal variations)
if mat_type == 'H':
    htsi_data2, sky_ADU2, htsi_mse2,ccdN, phoN = HTSI.htsi(sky1, order, slit_width, DMD_contrast, CCD)
if mat_type == 'S':
    htsi_data2, sky_ADU2, htsi_mse2, ccdN, phoN = HTSI.S_htsi(sky1, order, slit_width, DMD_contrast, CCD) # The HTSI result
mos_data2, sky_ADU_mos2, mos_mse2,ccdN, phoN = HTSI.MOS(sky1, order, slit_width, DMD_contrast, CCD) # The MOS or slit scanning result
Q_whole2 = np.divide((np.sqrt(np.median(mos_mse2))),(np.sqrt(np.median(htsi_mse2))))
htsi_SNR, mos_SNR, h_spec, m_spec, s_spec, Q2, h_mse, m_mse = HTSI.SNR_Q_ROI(x1,x2,y1,y2, htsi_data2, mos_data2, sky_ADU2) #calculatint the gain in SNR
total_s = np.sum(s_spec)
avg_s2 = np.mean(sky_ADU2)

data = np.transpose(np.vstack((Q_whole2,Q2,h_mse, m_mse, avg_s2)))
np.savetxt(folder+'Regular_HTSI_results_'+str(mat_type)+str(order)+'_'+str(obj)+'.txt', data, delimiter='\t')    

#%% PLOTTING THE REGULAR HTSI RESULTS (no temporal variations in here)
wl_slice = 128
b = 60 # slice in spatial axis
xs, ys = 60,47# spectrum points

plt.subplot(3,3,1)
plt.imshow(sky_ADU2[:,:,wl_slice], cmap = 'inferno', clim=(0,80))
plt.title('Original Sky')
plt.axis('off')
plt.subplot(3,3,2)
plt.imshow(htsi_data2[:,:,wl_slice],cmap = 'inferno', clim=(0,80))
plt.title('HTSI')
plt.axis('off')
plt.subplot(3,3,3)
plt.imshow(mos_data2[:,:,wl_slice],cmap = 'inferno', clim=(0,80))
plt.title('MOS')
plt.axis('off')

plt.subplot(3,3,4),
plt.imshow(sky_ADU2[:,b,:],cmap = 'inferno', clim=(0,60))
plt.axis('off')
plt.subplot(3,3,5)
plt.imshow(htsi_data2[:,b,:],cmap = 'inferno', clim=(0,60))
plt.axis('off')
plt.subplot(3,3,6)
plt.imshow(mos_data2[:,b,:],cmap = 'inferno', clim=(0,60))
plt.axis('off')

plt.subplot(3,3,7)
plt.plot(wls,sky_ADU2[xs,ys,:], color = 'darkorchid')
plt.ylim(0,120)
plt.subplot(3,3,8)
plt.plot(wls,htsi_data2[xs,ys,:], color = 'darkorchid')
plt.ylim(0,120)
plt.subplot(3,3,9)
plt.plot(wls,mos_data2[xs,ys,:],color = 'darkorchid')
plt.ylim(0,120)
#%% OPTIONAL- Run simulations with temporal variations. 

Jitter = ['Y','1','5'] # Y or N, # pixels to jitter, # of cubes to generate
PSF = ['Y','3','6','5'] # Y or N, # max psf variability, # of cubes to generate
# PSF ranges: 3 = ~0.3", SAMOS best. 5 =~ 0.5" SAMOS avg. 10 = ~1", typical avg observatory
Intensity = ['Y','7','5'] # Y or N, # max % of Intensity variations, # of cubes to generate

#%Temporal Simulations
# To include all 3:
flat_skies, r, xis, yis, Is = all_temporal(sky, PSF, Jitter, Intensity, x,y,l, sky_edge)
## To include just jitter and Intensity:
#flat_skies, xis, yis = temporal_jitter(sky, Jitter, sky_edge)
#flat_skies, Is = temporal_I_post(flat_skies, Intensity, x,y,l)

if mat_type == 'H':
    htsi_data, sky_indices = HTSI.htsi_temporal(flat_skies, sky_shape, order, slit_width, DMD_contrast, CCD)
if mat_type == 'S':
    htsi_data, sky_indices = HTSI.S_htsi_temporal(flat_skies, sky_shape, order, slit_width, DMD_contrast, CCD)
mos_data, sky_indices = HTSI.MOS_temporal(flat_skies, sky_shape, order, slit_width, DMD_contrast, CCD, sky_indices)

Q, qs, mos_mse, htsi_mse, sky_ADU = HTSI.compute_MSEs_Q(sky, htsi_data, mos_data, max_adu, order, gain, slit_width)
Q_whole = np.divide((np.sqrt(np.median(mos_mse))),(np.sqrt(np.median(htsi_mse)))) 
htsi_SNR, mos_SNR, h_spec, m_spec, s_spec, Q, h_mse, m_mse = HTSI.SNR_Q_ROI(x1,x2,y1,y2, htsi_data, mos_data, sky_ADU)
total_s = np.sum(s_spec) # 
avg_s = np.mean(s_spec) # Avg sky signal
#avg_s = np.mean(sky_ADU)
# Save results into a text file
ID = np.random.randint(1,100)
data = np.vstack((Q_whole,Q,avg_s,avg_s,h_mse,m_mse))
np.savetxt(folder+'Temporal_HTSI_results_'+str(mat_type)+str(order)+'_'+str(obj)+str(ID)+'.txt', data, delimiter='\t')    

sky_img = np.sum(sky_ADU, axis = 2)
htsi_img = np.sum(sky_ADU, axis = 2)
htsi_img_temporal = np.sum(sky_ADU, axis = 2)
mos_img = np.sum(sky_ADU, axis = 2)
mos_img_temporal = np.sum(sky_ADU, axis = 2)

# Create and write to logfile for this run
logfile = open(folder+'simulation_log_'+str(mat_type)+str(order)+'_'+str(obj)+str(ID)+'.txt', "w+")

logfile.write('Jitter: '+str(Jitter)+'\n')  
logfile.write('PSF: '+str(PSF)+'\n')  
logfile.write('Intensity: '+str(Intensity)+'\n')  
logfile.write('CCD Readnoise: '+str(lam0)+', '+str(lam)+'\n')  
logfile.write('Wavelength Resolution :'+str(nm_sample)+'\n')  
logfile.write('Telescope Area: '+str(A)+'\n')  
logfile.write('Matrix Type & Order: '+str(mat_type)+str(order)+'\n')  
logfile.write('Q (obj): '+str(Q)+'\n')  
logfile.write('Avg h_mse: '+str(np.mean(h_mse))+'\n')  
#logfile.write('Telescope Transmission profile: '+str(telescope)+'\n')  
logfile.write('DMD Contrast: '+str(DMD_contrast)+'\n')  
logfile.write('Whole Cube Q: '+str(Q_whole)+'\n')  
logfile.write('Avg Flux [e-/pix]: '+str(avg_s)+'\n')  
logfile.write('Noise Ratio: '+str(noise_ratio)+'\n')
logfile.write('sky_indices : '+str(sky_indices)+'\n')  
logfile.write('r (PSF): '+str(r)+'\n')  
logfile.write('xis : '+str(xis)+'\n')  
logfile.write('yis : '+str(xis)+'\n')  
logfile.write('Is : '+str(Is)+'\n')  
logfile.close() 


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

#%% Plot results with a comparison to untouched htsi
wl_slice = 255
b = 55 # slice in spatial axis
xs, ys = 55,41 # spectrum points in cube (pixels)

plt.subplot(3,5,1)
plt.imshow(sky_ADU2[:,:,wl_slice],clim = (0,70),cmap = 'inferno')
plt.title('Original Sky')
plt.axis('off')
plt.subplot(3,5,2)
plt.imshow(htsi_data[:,:,wl_slice],clim = (0,70),cmap = 'inferno')
plt.title('HTSI')
plt.axis('off')
plt.subplot(3,5,3)
plt.imshow(mos_data[:,:,wl_slice],clim = (0,70),cmap = 'inferno')
plt.title('MOS')
plt.axis('off')
plt.subplot(3,5,4)
plt.imshow(htsi_data2[:,:,wl_slice],clim = (0,70),cmap = 'inferno')
plt.title('HTSI without temporal variations')
plt.axis('off')
plt.subplot(3,5,5)
plt.imshow(mos_data2[:,:,wl_slice],clim = (0,70),cmap = 'inferno')
plt.title('MOS without temporal variations')
plt.axis('off')

plt.subplot(3,5,6)
plt.imshow(sky_ADU[b,:,:],clim = (0,80),cmap = 'inferno')
plt.axis('off')
plt.subplot(3,5,7)
plt.imshow(htsi_data[b,:,:],clim = (0,80),cmap = 'inferno')
plt.axis('off')
plt.subplot(3,5,8)
plt.imshow(mos_data[b,:,:],clim = (0,80),cmap = 'inferno')
plt.axis('off')
plt.subplot(3,5,9)
plt.imshow(htsi_data2[b,:,:],clim = (0,80),cmap = 'inferno')
plt.axis('off')
plt.subplot(3,5,10)
plt.imshow(mos_data2[b,:,:],clim = (0,80),cmap = 'inferno')
plt.axis('off')

plt.subplot(3,5,11)
plt.plot(wls,sky_ADU[xs,ys,:], color = 'darkorchid')
plt.subplot(3,5,12)
plt.plot(wls,htsi_data[xs,ys,:], color = 'darkorchid')
plt.subplot(3,5,13)
plt.plot(wls,mos_data[xs,ys,:], color = 'darkorchid')
plt.subplot(3,5,14)
plt.plot(wls,htsi_data2[xs,ys,:], color = 'darkorchid')
plt.subplot(3,5,15)
plt.plot(wls,mos_data2[xs,ys,:], color = 'darkorchid')


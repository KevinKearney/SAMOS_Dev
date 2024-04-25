# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 20:10:20 2020

@author: Kate
"""
from astropy.io import fits
from PIL import Image, ImageOps
import numpy as np
import glob
#import cv2
import matplotlib.pyplot as plt
from scipy.constants import h
from scipy.constants import c    
from astropy.convolution import AiryDisk2DKernel, Moffat2DKernel, Gaussian2DKernel
from astropy.convolution import convolve, convolve_fft
from scipy.ndimage import gaussian_filter1d

import os,sys
cwd = os.getcwd()
parent = os.path.abspath(os.path.join(cwd, os.pardir))
sys.path.append(parent)
sys.path.append(os.path.join(parent,'sandbox'))

class data_sim():
    
    def reshape_spectra(self, spec, wls, lam0, lam, nm_sample): # To be used in actual modeling code, this reshaes the spectra to account for spectral resoltion    
        l1 = np.where( wls == lam0)[0][0] # The index of lower end of desired wavelength range in nm
        l2 = np.where( wls == lam)[0][0] # The index of upper end of desired wavelength range in nm    
        spectra = spec[l1:l2:2*nm_sample]    
        return spectra

    def open_fits_spectra(self, filename, wls, lam0, lam, nm_sample):
        hdul = fits.open(filename)
        data = hdul[0].data
        data = self.reshape_spectra(data,wls,lam0,lam,nm_sample)
        data = data #*1000 # The data is flux callibrated
        return data 

    def gaussNoise(self, img):
        # Variations in illumination etc. 
        row, col = img.shape
        mean = 0 
        var = 0.1
        sigma = var **0.5
        gauss = np.random.normal(mean,sigma,(row,col))
        gauss = gauss.reshape(row,col)
        gauss = gauss 
        gNoise = img + gauss
        RMSnoise = np.sqrt(np.mean(gauss**2))
        return gNoise, RMSnoise

    def flatten(self, img):
        y =  np.shape(img)[1]
        flat_array = []
        for i in range(0,y):
            row = img[:,i]
            flat_array = np.append(flat_array, row)
        return flat_array
    
    def sum_spectra(self, indices, filenames, lam0, lam, nm_sample, wls):
        # Open the fits files for a number of stellar spectra
        # This data is the Flux in F_lambda units callibrated flux (https://www.eso.org/sci/facilities/paranal/decommissioned/isaac/tools/lib.html)
        l = len(indices)
        spectra = self.open_fits_spectra(filenames[indices[0]], wls, lam0, lam, nm_sample)
        for i in range(1,l):
            spec = self.open_fits_spectra(filenames[indices[i]], wls, lam0, lam, nm_sample)
            spectra = spectra+spec
        return spectra
    
    def flux_spectra(self, fname, lam0, lam, nm_sample):   
        ### Data cube based on a image and spectral data
        hdul = fits.open(fname)
        spec = hdul[1].data
        wls = []
        flux = []
        for j in range (0, len(spec)):
            wl = spec[j][0] 
            wl_nm = wl/10
            wls = np.append(wls, wl_nm)
            #Watts = spec[j][1]*(10**(-7)) # converts from ergs to watts
            wl_um = wl/10000
            PFD = spec[j][1] *(wl_um)*(1/(1.988*(10**(-10))))
            #A = np.pi * (200**2)# telescope area in cm2
            #Photons = Watts*wl*(1/(h*c))*A
            Photons = PFD*wl#*A # flux in photons per sec
            flux = np.append(flux, Photons) 
        # reshape/account for wl resolution
        spectra = []
        new_wls = []
        half = nm_sample/2
        wl_list = np.arange(lam0,lam, nm_sample)
        for z in range (0, len(wl_list)):
            wli = wl_list[z]
            index = (np.abs(wls - wli)).argmin()   
            if (nm_sample % 2) == 0: # checks if sampling is odd or even
                # Then you can easily integrate wls 
                flux_list = flux[int(index-half):int(index+half)]
                integrated_flux = np.sum(flux_list)
            else:
                half1 = half-0.5
                half2 = half+1
                flux_list = flux[int(index-half1):int(index+half2)]
                integrated_flux = np.sum(flux_list)
            spectra = np.append(spectra,integrated_flux)
           # spectra = np.append(spectra, flux[index])
            new_wls = np.append(new_wls, np.round(wls[index]).astype('int'))
            
        #wls = np.round(wls).astype('int')
        #l1 = np.where( wls == lam0)[0][0] # The index of lower end of desired wavelength range in nm
        #l2 = np.where( wls == lam)[0][0] # The index of upper end of desired wavelength range in nm
        #spectra = flux[l1:l2:2*nm_sample] # This just extracts every n samples from spectrum, but what we really want is to integrate over the spectral resolution for each point
        #new_wls = wls[l1:l2:2*nm_sample] /10
        #spectra = spectra * nm_sample # An APPROXIMATE way to integrate the flux over the wavelength chunk (aka resolution) 
        
        return spectra, new_wls # The output is flux in photons/sec/nm/m^2 and wavelengths in nm
     
    def sim_data_flux_basic_BG(self, lam0, lam, nm_sample, n, radius, BG_mag,f):
        ### Data cube based on a image and spectral data
        # Create an array with the full wavelength data
        print('START')
        A = np.pi*(2.0)**2
        # Get the image filenames
        filenames = glob.glob('C:/Users/Kate/Documents/hadamard/STIS NGSL/'+'*.fits') # Find the images in a given directory
        #https://archive.stsci.edu/prepds/stisngsl/
        fname = filenames[47]
        fname2 = filenames[40]
       
        spec1, wls = self.flux_spectra(fname,lam0,lam,nm_sample) # mag 12
        spec2, wls2 = self.flux_spectra(fname2,lam0,lam,nm_sample) # mag 2.5
        spec1 = spec1*A
        spec2 = spec2*A
        spec2= np.round(spec2)
        spec1 = np.round(spec1)
        sky = np.ones((n,n,len(wls))) # empty sky data cube
        
        # Generate sky bg:
       # BG_mag = 0.14, 0.1, 0.08, 0.0615, 0.055, 0.03, 0.022
       # f = 0.05, 0.08, 0.1, 0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19, 0.2, 0.22,0.24,0.26,0.28, 0.3, 0.32,0.34,0.36,0.38,0.4,0.5
       # f = 1, 0.5, 0.3, 0.2, 0.19, 0.18, 0.17, 0.16, 0.15, 0.12, 0.11, 0.105, 0.1, 0.09, 0.08, 0.07, 0.05, 0.03, 0.01
        bg_flux = spec2
        bg_flux = bg_flux*BG_mag #0.022
        
        ###########f = .11
        spec1 = (spec1+bg_flux)*f
        
        sky = sky*bg_flux
        ###
        bg_mag = bg_flux*(1/100000)*(1/nm_sample)
        l1 = np.argmin(np.abs(np.array(wls)-508))
        l2 = np.argmin(np.abs(np.array(wls)-594))
        bg_mag = bg_mag[l1:l2]
        bg_mag = np.mean(bg_mag)
        mV = -2.5*np.log10(np.divide(bg_mag,995.5))
        SB_V = mV-(2.5*np.log10(0.165)) 
            
        
        
        s_mag = spec1*(1/100000)*(1/nm_sample)
        l1 = np.argmin(np.abs(np.array(wls)-508))
        l2 = np.argmin(np.abs(np.array(wls)-594))
        s_mag = s_mag[l1:l2]
        s_mag = np.mean(s_mag)
        S_mV = -2.5*np.log10(np.divide(s_mag,995.5))
        S_SB_V = S_mV-(2.5*np.log10(0.165)) 
        
        #print(SB_V)
        #print(S_SB_V)
        
        x1 = int((n/2) - radius)
        x2 = int((n/2) + radius)
        for j in range (x1,x2):
            for k in range (x1,x2):
                sky[j,k,:] = spec1
                
        return sky, wls, SB_V, S_SB_V
    
    
    
    def sim_data_flux_basic(self, lam0, lam, nm_sample, n, radius):
        # Create an array with the full wavelength data
        print('START')
        A = np.pi*(2.0)**2
        # Get the image filenames
        STIS_folder = os.path.join(parent,'data/STIS NGSL/')
        filenames = glob.glob(STIS_folder+'*.fits') # Find the images in a given directory
        #https://archive.stsci.edu/prepds/stisngsl/
        fname = filenames[47]
        #fname2 = filenames[59]
        spec1, wls = self.flux_spectra(fname,lam0,lam,nm_sample) # mag 12
        #spec2, wls2 = self.flux_spectra(fname2,lam0,lam,nm_sample) # mag 2.5
        spec1 = spec1*A
        spec1 = np.round(spec1)
        sky = np.ones((n,n,len(wls))) # empty sky data cube
        
        # Generate sky bg:
        bg_flux = np.mean(spec1)*10**(-15)
        sky = sky*bg_flux
        sky = sky*bg_flux
        x1 = int((n/2) - radius)
        x2 = int((n/2) + radius)
        for j in range (x1,x2):
            for k in range (x1,x2):
                sky[j,k,:] = spec1
        return sky, wls
    
    def extend_values(self, wls, val_wls, vals):
        new_wls = []
        new_vals = []
        for i in range(0, len(wls)):
            lam = wls[i]
            index = (np.abs(val_wls - lam)).argmin()    
            new_wls = np.append(new_wls, val_wls[index])
            new_vals = np.append(new_vals, vals[index])
        
        return new_vals
    
    def apply_thruput(self, sky, wls, telescope):
        # This function applie realistic throughput curves and PSF to the simulated data
        # "telescope" is a string that indicates which throughput model to use
        # First, obtain the data files
        folder = os.path.join(parent,'data/optical_data/')
        # SAMOS Throughput Data
        SAMOS = np.loadtxt(folder+'SAMOS_throughput.txt', dtype='str', delimiter='\t') # Opens the text file containing the list of exposure times for each wavelength
        wl_lowres_blue = SAMOS[1:10,0].astype('float')
        wl_lowres_red = SAMOS[9:,0].astype('float')
        lowres_blue = SAMOS[1:10,1].astype('float')
        lowres_red = SAMOS[9:,2].astype('float')
        lo_res_wl = np.append(wl_lowres_blue, wl_lowres_red)
        lo_res_T = np.append(lowres_blue, lowres_red)
        wl_hi_res_blue = SAMOS[1:10,4].astype('float')
        wl_hi_res_red = SAMOS[1:12,6].astype('float')
        hi_res_blue = SAMOS[1:10,5].astype('float')
        hi_res_red = SAMOS[1:12,7].astype('float')
        hi_res_wl = np.append(wl_hi_res_blue,wl_hi_res_red)
        hi_res_T = np.append(hi_res_blue, hi_res_red)
        # Measured DMD Reflectance   
        DMD_R = np.loadtxt(folder+'XGA_reflectance.txt', dtype='str', delimiter='\t') 
        wl_dmd = DMD_R[0:,0].astype('float')
        DMD_reflectance = DMD_R[0:,1].astype('float')
        # CCD QE example
        CCD_QE = np.loadtxt(folder+'e2v_ccd_qe_approx.txt', dtype='str', delimiter='\t') 
        wl_qe = CCD_QE[1:,0].astype('float')
        ccd_qe = CCD_QE[1:,1].astype('float')
        
        ##########################################################################################33
        new_sky = np.zeros((np.shape(sky)[0],np.shape(sky)[1],np.shape(sky)[2]))
        DMD = self.extend_values(wls, wl_dmd, DMD_reflectance)
        CCD = self.extend_values(wls, wl_qe, ccd_qe)

        if telescope == 'Gemini':
            # Gemini Throughput
            Gemini = np.loadtxt(folder+'Telescope_throughput_Gemini.txt', dtype='str', delimiter='\t') 
            wl_Gemini = Gemini[1:,0].astype('float')
            T_Gemini = Gemini[1:,1].astype('float')
            T1 = self.extend_values(wls, wl_Gemini, T_Gemini)
            T = T1*DMD*CCD
        if telescope == 'SAMOS_low':
            T1 = self.extend_values(wls, lo_res_wl, lo_res_T)
            T = T1*CCD
        if telescope == 'SAMOS_hi':
            T1 = self.extend_values(wls, hi_res_wl, hi_res_T)
            T = T1*CCD
        if telescope == 'aluminum':
            # Bulk Al Reflectance
            Al_data = np.loadtxt(folder+'reflectance_flatmirror.txt', dtype='str', delimiter='\t') 
            wl_Al = Al_data[1:,0]
            R_Al = Al_data[1:,4]/100
            T1 = self.extend_values(wls, wl_Al, R_Al)
            T = (T1**3)*DMD*CCD
        if telescope == 'silver':
            # Bulk Protected Silver Reflectance
            silver_data = np.loadtxt(folder+'protected_silver_R.txt', dtype='str', delimiter='\t') 
            wl_silver = silver_data[1:,0]
            R_silver = silver_data[1:,1]
            T1 = self.extend_values(wls, wl_silver, R_silver)
            T = (T1**3)*DMD*CCD

        for i in range(0,np.shape(sky)[1]):
            img = sky[:,i,:]
            img2 = img * T # Multiplies by the throughput of the instument+detector QE            
            new_sky[:,i,:]= img2 
        return new_sky, T
        
    def apply_psf(self, sky, kernel_type, a):
        if kernel_type == 'airydisk':
            kernel = AiryDisk2DKernel(a)
        if kernel_type == 'moffat':
            b = 4.765
            #a is the core width
            #b is the power index and should be kept constant
            kernel = Moffat2DKernel(a, b)    
        if kernel_type == 'gauss':
            kernel = Gaussian2DKernel(a)
        for i in range(0,np.shape(sky)[2]):
                img = sky[:,:,i]
                img_psf = convolve_fft(img, kernel)
                sky[:,:,i]= img_psf 
        return sky
    
    def vary_psf_along_wl(self, sky, D, wls):
        for i in range(0,np.shape(sky)[2]):
            wl = wls[i]
            pix_scale = 0.13 # arcsec per pixel
            a0 = (wl*10**(-9))*1.22/D # diffraction limited
            a1 = (a0*360/(2*np.pi)) * 3600 # converts from radians to arcsec
            a = a1/pix_scale
            kernel = AiryDisk2DKernel(a)
            img = sky[:,:,i]
            img_psf = convolve_fft(img, kernel)
            sky[:,:,i]= img_psf               
        return sky

    def convert_photon_flux_to_ergs(flux_in, wls_in, Area, nm_sample):
        # Converts flux given in units of Photons/sec/nm* delta nm (as collected by a telescope of Area A(m^2)) to Flux Ergs/sec/Angstrom/cm^2
        Flux_ergs = []
        wls_A = []
        for i in range(0, len(flux_in)):
            Fo = flux_in[i]/nm_sample #divides by "bandwidth" aka spectral res
            Wo = wls_in[i]
            W_Ang = Wo*10 # converts nm to Angstroms
            W_um = W_Ang/10000 # converts wavelength to microns
            F_new = Fo*(1.988*(10**(-10)))*(1/W_um)*(1/Area)
            Flux_ergs = np.append(Flux_ergs, F_new)
            wls_A = np.append(wls_A, W_Ang)
        return Flux_ergs, wls_A

    #def convert_flux_to_mag(flux_in, wls):
        # Takes flux in ergs/sec/Ang/cm^2 and converts to apparent mag
        # Flux must have already been adjusted for instrument throughput

######################################################################################################
        #### Functions to create more complex input sky data
    
    def sim_sky_stellar1(self, lam0, lam, nm_sample):
        ### Data cube based on a image and spectral data
        ngc6563 = Image.open(os.path.join(parent,'data','imgs','potw1452a.tif')) # 
        ngc6 = ImageOps.grayscale(ngc6563)
        ngc6 = ngc6.resize((512,512))
        img = np.asarray(ngc6, dtype='int64') #KJK
        # https://esahubble.org/images/potw1452a/
        # Create an array with the full wavelength data
        filenames = glob.glob(os.path.join(parent,'data','STIS NGSL','*.fits')) # Find the images in a given directory
        spec1, wls = self.flux_spectra(filenames[0],lam0,lam,nm_sample)
        stellar_specs = np.zeros((len(spec1),len(filenames)))
        for i in range (0, len(filenames)):
            spec, wls = self.flux_spectra(filenames[47],lam0,lam,nm_sample) 
            stellar_specs[:,i] = spec
            
        mix = np.mean(stellar_specs,axis=1)
    
        x,y = np.shape(img)[0], np.shape(img)[1]
        data_cube = np.ones((x,y,len(spec1)))*(mix*0.05)
        star_indices = np.where(img >75)
        
        for k in range(0, np.shape(star_indices)[1]):
            l = np.random.randint(0,48)
            spec = stellar_specs[:,l]
            px, py = star_indices[0][k], star_indices[1][k]
            p = img[px,py]
            data_cube[px,py,:]=spec*p
        print('Spectral data generated')
        sky = (data_cube + 0.5)/255
        print('Data cube combined')
        return sky, wls  
    
    def sim_sky_stellar2(self, lam0, lam, nm_sample):
        ### Data cube based on a image and spectral data
        messier79 = Image.open(os.path.join(parent,'data','imgs','potw1751a.tif')) # 
        m79 = ImageOps.grayscale(messier79)
        m79 = m79.resize((512,512))
        img = np.asarray(m79, dtype='int64') #KJK
        # Create an array with the full wavelength data
        filenames = glob.glob(os.path.join(parent,'data','STIS NGSL','*.fits')) # Find the images in a given directory
        spec1, wls = self.flux_spectra(filenames[0],lam0,lam,nm_sample)
        stellar_specs = np.zeros((len(spec1),len(filenames)))
        for i in range (0, len(filenames)):
            spec, wls = self.flux_spectra(filenames[47],lam0,lam,nm_sample) 
            stellar_specs[:,i] = spec 
        mix = np.mean(stellar_specs,axis=1)
        x,y = np.shape(img)[0], np.shape(img)[1]
        data_cube = np.ones((x,y,len(spec1)))*(mix**(0.5))*0.33
        star_indices = np.where(img >75)
        for k in range(0, np.shape(star_indices)[1]):
            l = np.random.randint(0,48)
            spec = stellar_specs[:,l]
            px, py = star_indices[0][k], star_indices[1][k]
            p = img[px,py]
            data_cube[px,py,:]=spec*p*0.55
        print('Spectral data generated')
        sky = (data_cube)/255
        print('Data cube combined')
        return sky, wls   

    def sim_sky_nebula(self,lam0, lam, nm_sample):
        ### Data cube based on a image and spectral data
        IRAS05240 = Image.open('C:/Users/Kate/Documents/hadamard/modeling/imgs/opo9923c.tif') # https://esahubble.org/images/opo9923c/
        IRAS = ImageOps.grayscale(IRAS05240)
        IRAS2 = IRAS.resize((400,400))
        img = np.asarray(IRAS2, dtype='int64') #KJK
        
        # Create an array with the full wavelength data
        neb_name = 'C:/Users/Kate/Documents/hadamard/modeling/STScI/pne/pn_spectrum_070k_normal.fits' # location of a HII region or other spectra
        neb_name2 = 'C:/Users/Kate/Documents/hadamard/modeling/STScI/pne/pn_spectrum_100k_normal.fits' # location of a HII region or other spectra
        neb_spec, wls = self.flux_spectra(neb_name, lam0,lam,nm_sample)
        neb_spec2, wls = self.flux_spectra(neb_name2, lam0,lam,nm_sample)
        peak = np.where(neb_spec == np.max(neb_spec))[0][0]
        neb_spec[peak] = neb_spec[peak]*0.15
        neb_spec = gaussian_filter1d(neb_spec, 6)
        neb_spec2 = gaussian_filter1d(neb_spec, 3)


        x,y = np.shape(img)[0], np.shape(img)[1]
        data_cube = np.zeros((x,y,len(neb_spec)))
    
        neb_indices = np.where(img >12)
        bg_indices = np.where(img< 12)
        
        for i in range (0, np.shape(neb_indices)[1]):
            px, py = neb_indices[0][i], neb_indices[1][i]
            p = img[px,py]
            data_cube[px,py,:] = neb_spec2*p
        
        for k in range (0, np.shape(bg_indices)[1]):
            px, py = bg_indices[0][k], bg_indices[1][k]
            p = img[px,py]
            data_cube[px,py,:] = (np.mean(neb_spec2)*p*0.25)
            
        print('Spectral data generated')
        sky = (data_cube)
        print('Data cube combined')
        return sky, wls   

    def sim_sky_squid(self, lam0, lam, nm_sample):
        ### Data cube based on a image and spectral data
        M77squid = Image.open('C:/Users/Kate/Documents/hadamard/modeling/imgs/noao-m77.tif') # 
        squid = ImageOps.grayscale(M77squid)
        squid = squid.resize((1640,1360))
        img = np.asarray(squid, dtype='int64') #KJK
        img = img[113:1137,278:1302]
        agn_name = 'C:/Users/Kate/Documents/hadamard/modeling/STScI/agn/ngc1068_template.fits' # location of a HII region or other spectra
        agn_spec, wls = self.flux_spectra(agn_name, lam0,lam,nm_sample)
        # Create an array with the full wavelength data
        filenames = glob.glob('C:/Users/Kate/Documents/hadamard/STIS NGSL/'+'*.fits') # Find the images in a given directory
        spec1, wls = self.flux_spectra(filenames[47],lam0,lam,nm_sample) # mag 12
        spec2, wls = self.flux_spectra(filenames[46],lam0,lam,nm_sample) # mag 
        spec3, wls = self.flux_spectra(filenames[45],lam0,lam,nm_sample) # mag 
        spec4, wls = self.flux_spectra(filenames[44],lam0,lam,nm_sample) # mag 
        spec5, wls = self.flux_spectra(filenames[43],lam0,lam,nm_sample) # mag 
        stars = np.transpose(np.vstack((spec1,spec2,spec3,spec4,spec5)))
        mix = np.mean(stars,axis=1)
        x,y = np.shape(img)[0], np.shape(img)[1]
        data_cube = np.ones((x,y,len(spec1)))#*mix
        for i in range(0,x):
            pxi = i
            for j in range(0,y):
                pyi = j
                p = img[pxi,pyi]
                data_cube[pxi,pyi,:]= p * mix *2.2
        agn_indices = np.where(img >30)
        for k in range(0, np.shape(agn_indices)[1]):
            px, py = agn_indices[0][k], agn_indices[1][k]
            p = img[px,py]
            data_cube[px,py,:]= p * agn_spec * 0.9
        print('Spectral data generated')
        sky = data_cube#+0.5 #bg level
        sky = sky/255
        print('Data cube combined')
        return sky, wls   
    
    def sim_sky_cygnus(self, lam0, lam, nm_sample):
        ### Data cube based on a image and spectral data
        cygnus_loop = Image.open('C:/Users/Kate/Documents/hadamard/modeling/imgs/heic0006c.jpg') # 
        cygnus = ImageOps.grayscale(cygnus_loop)
        cyg_x = np.shape(cygnus)[1]
        cyg_y = np.shape(cygnus)[0]
        cygnus2 = cygnus.resize((int(cyg_x/2),int(cyg_y/2)))
        cygnus2 = np.asarray(cygnus2, dtype='int64') #KJK
        cygnus22 = cygnus2[:800,900:-76] # A good crop for HII spectra
        cygnus2b = np.copy(cygnus22)
        bg0 = cygnus2b[510:634,20:276]
        chunk = np.zeros((124,1024))
        chunk[:,0:256] = bg0
        chunk[:,256:512] = bg0
        chunk[:,512:-256] = bg0
        chunk[:,-256:] = bg0
        #plt.imshow(chunk)
        img = np.ones((1024,1024))
        img[124:-100,:] = cygnus2b
        img[0:124,:] = chunk
        img[-100:,:] = cygnus2[800:900,900:-76]#chunk[:100,:]
        # Create an array with the full wavelength data
        filenames = glob.glob('C:/Users/Kate/Documents/hadamard/STIS NGSL/'+'*.fits') # Find the images in a given directory
       # spec1, wls = SIM.flux_spectra(filenames[0],lam0,lam,nm_sample)
        spec1, wls = self.flux_spectra(filenames[47],lam0,lam,nm_sample) # mag 12
        spec2, wls = self.flux_spectra(filenames[46],lam0,lam,nm_sample) # mag 
        spec3, wls = self.flux_spectra(filenames[45],lam0,lam,nm_sample) # mag 
        spec4, wls = self.flux_spectra(filenames[44],lam0,lam,nm_sample) # mag 
        spec5, wls = self.flux_spectra(filenames[43],lam0,lam,nm_sample) # mag 
        stars = np.transpose(np.vstack((spec1,spec2,spec3,spec4,spec5)))
        mix = np.mean(stars,axis=1)
        neb_name = 'C:/Users/Kate/Documents/hadamard/modeling/STScI/pn_nebula_only_smooth.fits' # location of a HII region or other spectra
        neb_spec, wls = self.flux_spectra(neb_name, lam0,lam,nm_sample)
        x,y = np.shape(img)[0], np.shape(img)[1]
        data_cube = np.zeros((x,y,len(spec1)))    
        kernel = Gaussian2DKernel(3)
        sky_blur = convolve_fft(img,kernel)
        neb_indices = np.where(sky_blur >50)
        star_indices = np.where(sky_blur <50)
        #bg_spec = np.ones((len(spec1)))*np.min(spec1)    
        for i in range (0, np.shape(neb_indices)[1]):
            px, py = neb_indices[0][i], neb_indices[1][i]
            p = img[px,py]
            data_cube[px,py,:] = neb_spec*p
            if px < 124:
                data_cube[px,py,:]=mix*p*0.01    
        for k in range(0, np.shape(star_indices)[1]):
            l = np.random.randint(0,4)
            spectra = stars[:,l]
            px, py = star_indices[0][k], star_indices[1][k]
            p = img[px,py]
            if p > 25:
                data_cube[px,py,:]=spectra*p*8
            if p <=25:
                data_cube[px,py,:]=mix*p*0.01        
        print('Spectral data generated')
        sky = data_cube+0.5 #bg level
        sky = sky/255
        print('Data cube combined')
        return sky, wls   
    
    def sim_sky_n7(self, lam0, lam, nm_sample):
        ### Data cube based on a image and spectral data
        n7_neb = Image.open('C:/Users/Kate/Documents/hadamard/modeling/imgs/n7.jpg') # 
        n7_neb= ImageOps.grayscale(n7_neb)
        n7 = n7_neb.resize((512,512))
        img = np.asarray(n7, dtype='int64') #KJK
        img = img[191:447,103:359]
        agn_name = 'C:/Users/Kate/Documents/hadamard/modeling/STScI/pne/pn_spectrum_030k_normal.fits' # location of a HII region or other spectra
        agn_spec, wls = self.flux_spectra(agn_name, lam0,lam,nm_sample)
        # Create an array with the full wavelength data
        filenames = glob.glob('C:/Users/Kate/Documents/hadamard/STIS NGSL/'+'*.fits') # Find the images in a given directory
        spec1, wls = self.flux_spectra(filenames[47],lam0,lam,nm_sample) # mag 12
        spec2, wls = self.flux_spectra(filenames[46],lam0,lam,nm_sample) # mag 
        spec3, wls = self.flux_spectra(filenames[45],lam0,lam,nm_sample) # mag 
        spec4, wls = self.flux_spectra(filenames[44],lam0,lam,nm_sample) # mag 
        spec5, wls = self.flux_spectra(filenames[43],lam0,lam,nm_sample) # mag 
        stars = np.transpose(np.vstack((spec1,spec2,spec3,spec4,spec5)))
        mix = np.mean(stars,axis=1)
        x,y = np.shape(img)[0], np.shape(img)[1]
        data_cube = np.ones((x,y,len(spec1)))#*mix
        for i in range(0,x):
            pxi = i
            for j in range(0,y):
                pyi = j
                p = img[pxi,pyi]
                data_cube[pxi,pyi,:]= p * mix* 0.01
        agn_indices = np.where(img >40)
        for k in range(0, np.shape(agn_indices)[1]):
            px, py = agn_indices[0][k], agn_indices[1][k]
            p = img[px,py]
            data_cube[px,py,:]= p * agn_spec *10
        star_indices = np.where(img >180)
        for l in range(0, np.shape(star_indices)[1]):
            a = np.random.randint(0,4)
            sspec = stars[:,a]
            px, py = star_indices[0][l], star_indices[1][l]
            p = img[px,py]
            data_cube[px,py,:]= p * sspec *0.8
        print('Spectral data generated')
        sky = data_cube + 0.05 #bg level
        sky = sky/255
        print('Data cube combined')
        return sky, wls   

    def sim_sky_galaxy(self, lam0, lam, nm_sample):
        ### Data cube based on a image and spectral data
        ngc1614 = Image.open('C:/Users/Kate/Documents/hadamard/modeling/imgs/ngc1614.jpg') # https://esahubble.org/images/heic0810ax/
        ngc1614= ImageOps.grayscale(ngc1614)
        nGal = ngc1614.resize((256,256))
        img = np.asarray(nGal, dtype='int64') #KJK
        gal_name = 'C:/Users/Kate/Documents/hadamard/modeling/STScI/brown/ngc_1614_spec.fits' # location of a HII region or other spectra
        gal_spec, wls = self.flux_spectra(gal_name, lam0,lam,nm_sample)
        # Create an array with the full wavelength data
        filenames = glob.glob('C:/Users/Kate/Documents/hadamard/STIS NGSL/'+'*.fits') # Find the images in a given directory
        spec1, wls = self.flux_spectra(filenames[47],lam0,lam,nm_sample) # mag 12
        spec2, wls = self.flux_spectra(filenames[46],lam0,lam,nm_sample) # mag 
        spec3, wls = self.flux_spectra(filenames[45],lam0,lam,nm_sample) # mag 
        spec4, wls = self.flux_spectra(filenames[44],lam0,lam,nm_sample) # mag 
        spec5, wls = self.flux_spectra(filenames[43],lam0,lam,nm_sample) # mag 
        stars = np.transpose(np.vstack((spec1,spec2,spec3,spec4,spec5)))
        mix = np.mean(stars,axis=1)
        x,y = np.shape(img)[0], np.shape(img)[1]
        data_cube = np.ones((x,y,len(spec1)))#*mix
        for i in range(0,x):
            pxi = i
            for j in range(0,y):
                pyi = j
                p = img[pxi,pyi]
                data_cube[pxi,pyi,:]= p * mix #*0.15
        gal_indices = np.where(img >15)
        for k in range(0, np.shape(gal_indices)[1]):
            px, py = gal_indices[0][k], gal_indices[1][k]
            p = img[px,py]
            data_cube[px,py,:]= p * gal_spec 
        print('Spectral data generated')
        sky = data_cube +0.5 #bg level
        sky = sky/255
        print('Data cube combined')
        
        
        
        return sky, wls   
    
    def sim_sky_agn2(self, lam0, lam, nm_sample):
        ### Data cube based on a image and spectral data
        AGN1068 = Image.open('C:/Users/Kate/Documents/hadamard/modeling/imgs/1068 ag.jpg') # 
        AGN1068= ImageOps.grayscale(AGN1068)
        ag = AGN1068.resize((256,256))
        img = np.asarray(ag, dtype='int64') #KJK
        #img = img[191:447,103:359]
    
        agn_name = 'C:/Users/Kate/Documents/hadamard/modeling/STScI/agn/seyfert1_template.fits' # location of a HII region or other spectra
        agn_spec, wls = self.flux_spectra(agn_name, lam0,lam,nm_sample)
        # Create an array with the full wavelength data
        gal_name1 = 'C:/Users/Kate/Documents/hadamard/modeling/STScI/brown/ngc_4889_spec.fits' # location of a HII region or other spectra
        gal_name2 = 'C:/Users/Kate/Documents/hadamard/modeling/STScI/brown/ngc_4385_spec.fits' # location of a HII region or other spectra
        gal_name3 = 'C:/Users/Kate/Documents/hadamard/modeling/STScI/brown/ngc_1068_spec.fits' # location of a HII region or other spectra
    
        spec1, wls = self.flux_spectra(gal_name1,lam0,lam,nm_sample) # mag 12
        spec2, wls = self.flux_spectra(gal_name2,lam0,lam,nm_sample) # mag 
        spec3, wls = self.flux_spectra(gal_name3,lam0,lam,nm_sample) # mag 
        stars = np.transpose(np.vstack((spec1,spec2,spec3)))
        mix = np.mean(stars,axis=1)
        x,y = np.shape(img)[0], np.shape(img)[1]
        data_cube = np.ones((x,y,len(spec1)))#*np.mean(mix)
        for i in range(0,x):
            pxi = i
            for j in range(0,y):
                pyi = j
                p = img[pxi,pyi]
                data_cube[pxi,pyi,:]= p * np.mean(mix)*0.01
        gal_indices = np.where(img >20)
        for k in range(0, np.shape(gal_indices)[1]):
            px, py = gal_indices[0][k], gal_indices[1][k]
            p = img[px,py]
            data_cube[px,py,:]= p * mix *0.15
        agn_indices = np.where(img >180)
        for l in range(0, np.shape(agn_indices)[1]):
            px, py = agn_indices[0][l], agn_indices[1][l]
            p = img[px,py]
            data_cube[px,py,:]= p * agn_spec*0.3
        print('Spectral data generated')
        sky = data_cube + 0.05 #bg level
        sky = sky/255
        print('Data cube combined')
        return sky, wls 
    
    
    def sim_sky_old_gals(self, lam0, lam, nm_sample):
        ### Data cube based on a image and spectral data
        old_gals = Image.open('C:/Users/Kate/Documents/hadamard/modeling/imgs/potw1812a.jpg') # https://esahubble.org/images/potw1812a/
        old_gals= ImageOps.grayscale(old_gals)
        gal2 = old_gals.resize((1024,512))
        img = np.asarray(gal2, dtype='int64') #KJK
        img = img[84:340,331:587]
        gal_name = 'C:/Users/Kate/Documents/hadamard/modeling/STScI/swire_galaxies/spiral_swire_sa.fits' # location of a HII region or other spectra
        gal_spec, wls = self.flux_spectra(gal_name, lam0,lam,nm_sample)
        # Create an array with the full wavelength data
        filenames = glob.glob('C:/Users/Kate/Documents/hadamard/modeling/STScI/brown/'+'*.fits') # Find the images in a given directory
        starnames = glob.glob('C:/Users/Kate/Documents/hadamard/STIS NGSL/'+'*.fits') # Find the images in a given directory
        spec1, wls = self.flux_spectra(filenames[0],lam0,lam,nm_sample) # mag 12
        spec2, wls = self.flux_spectra(filenames[1],lam0,lam,nm_sample) # mag 
        spec3, wls = self.flux_spectra(filenames[3],lam0,lam,nm_sample) # mag 
        spec4, wls = self.flux_spectra(starnames[47],lam0,lam,nm_sample) # mag 
        spec5, wls = self.flux_spectra(starnames[43],lam0,lam,nm_sample) # mag 
        stars = np.transpose(np.vstack((spec1,spec2,spec3,spec4,spec5)))
        mix = np.mean(stars,axis=1)
        x,y = np.shape(img)[0], np.shape(img)[1]
        data_cube = np.ones((x,y,len(spec1)))#*mix
        spec, wls = self.flux_spectra(filenames[5],lam0,lam,nm_sample) # mag 


        for i in range(0,x):
            pxi = i
            for j in range(0,y):
                pyi = j
                p = img[pxi,pyi]
                data_cube[pxi,pyi,:]= p * (mix**(0.5))

        star_indices = np.where(img>100)
        for ii in range(0, np.shape(star_indices)[1]):
            a = np.random.randint(0,4)
            star_spec = stars[:,a]
            px, py = star_indices[0][ii], star_indices[1][ii]
            p = img[px,py]
            data_cube[px,py,:]= p * star_spec 
            
        gal_indices = np.where(img >70)
        for k in range(0, np.shape(gal_indices)[1]):
            px, py = gal_indices[0][k], gal_indices[1][k]
            if px >41:
                if px <228:
                    if py>50:
                        if py<195:
                            p = img[px,py]
                            data_cube[px,py,:]= p * spec *3

        print('Spectral data generated')
        sky = data_cube +1 #bg level
        sky = sky/255
        print('Data cube combined')
        return sky, wls   

    
    
    def sim_sky_gal_cluster(self, lam0, lam, nm_sample):
        ### Data cube based on a image and spectral data
        gal_cluster = Image.open('C:/Users/Kate/Documents/hadamard/modeling/imgs/ann1105a.tif') # https://esahubble.org/images/ann1105a/
        gal_cluster= ImageOps.grayscale(gal_cluster)
        gc = gal_cluster.resize((512,512))
        img = np.asarray(gc, dtype='int64') #KJK
        gal_name = 'C:/Users/Kate/Documents/hadamard/modeling/STScI/swire_galaxies/spiral_swire_sa.fits' # location of a HII region or other spectra
        gal_spec, wls = self.flux_spectra(gal_name, lam0,lam,nm_sample)
        # Create an array with the full wavelength data
        filenames = glob.glob('C:/Users/Kate/Documents/hadamard/modeling/STScI/brown/'+'*.fits') # Find the images in a given directory
        starnames = glob.glob('C:/Users/Kate/Documents/hadamard/STIS NGSL/'+'*.fits') # Find the images in a given directory
        spec1, wls = self.flux_spectra(filenames[0],lam0,lam,nm_sample) # mag 12
        spec2, wls = self.flux_spectra(filenames[1],lam0,lam,nm_sample) # mag 
        spec3, wls = self.flux_spectra(filenames[3],lam0,lam,nm_sample) # mag 
        spec4, wls = self.flux_spectra(starnames[47],lam0,lam,nm_sample) # mag 
        spec5, wls = self.flux_spectra(starnames[43],lam0,lam,nm_sample) # mag 
        stars = np.transpose(np.vstack((spec1,spec2,spec3,spec4,spec5)))
        mix = np.mean(stars,axis=1)
        x,y = np.shape(img)[0], np.shape(img)[1]
        data_cube = np.ones((x,y,len(spec1)))#*mix
        galaxies = np.zeros((len(spec1),len(filenames)))
        for l in range(0,5):
            spec, wls = self.flux_spectra(filenames[l],lam0,lam,nm_sample) # mag 
            galaxies[:,l] = spec 
        for i in range(0,x):
            pxi = i
            for j in range(0,y):
                pyi = j
                p = img[pxi,pyi]
                data_cube[pxi,pyi,:]= p * mix *0.01
        gal_indices = np.where(img >30)
        for k in range(0, np.shape(gal_indices)[1]):
            a = np.random.randint(0,len(filenames))
            gal_spec = galaxies[:,a]
            px, py = gal_indices[0][k], gal_indices[1][k]
            p = img[px,py]
            data_cube[px,py,:]= p * gal_spec 
        print('Spectral data generated')
        sky = data_cube +0.5 #bg level
        sky = sky/255
        print('Data cube combined')
        return sky, wls   

    def sim_sky_spiral(self, lam0, lam, nm_sample):
        ### Data cube based on a image and spectral data
        spiral_galaxy = Image.open('C:/Users/Kate/Documents/hadamard/modeling/imgs/eso0221a.jpg') # 
        spiral_galaxy= ImageOps.grayscale(spiral_galaxy)
        spiral = spiral_galaxy.resize((512,512))
        img = np.asarray(spiral, dtype='int64') #KJK
        gal_name = 'C:/Users/Kate/Documents/hadamard/modeling/STScI/swire_galaxies/spiral_swire_sa.fits' # location of a HII region or other spectra
        gal_spec, wls = self.flux_spectra(gal_name, lam0,lam,nm_sample)
        # Create an array with the full wavelength data
        filenames = glob.glob('C:/Users/Kate/Documents/hadamard/STIS NGSL/'+'*.fits') # Find the images in a given directory
        spec1, wls = self.flux_spectra(filenames[47],lam0,lam,nm_sample) # mag 12
        spec2, wls = self.flux_spectra(filenames[46],lam0,lam,nm_sample) # mag 
        spec3, wls = self.flux_spectra(filenames[45],lam0,lam,nm_sample) # mag 
        spec4, wls = self.flux_spectra(filenames[44],lam0,lam,nm_sample) # mag 
        spec5, wls = self.flux_spectra(filenames[43],lam0,lam,nm_sample) # mag 
        stars = np.transpose(np.vstack((spec1,spec2,spec3,spec4,spec5)))
        mix = np.mean(stars,axis=1)
        x,y = np.shape(img)[0], np.shape(img)[1]
        data_cube = np.ones((x,y,len(spec1)))#*mix
        for i in range(0,x):
            pxi = i
            for j in range(0,y):
                pyi = j
                p = img[pxi,pyi]
                data_cube[pxi,pyi,:]= p * mix *0.15
        gal_indices = np.where(img >15)
        for k in range(0, np.shape(gal_indices)[1]):
            px, py = gal_indices[0][k], gal_indices[1][k]
            p = img[px,py]
            data_cube[px,py,:]= p * gal_spec 
        print('Spectral data generated')
        sky = data_cube +0.5 #bg level
        sky = sky/255
        print('Data cube combined')
        return sky, wls   


    def sim_sky_globular(self, lam0, lam, nm_sample):
         ### Data cube based on a image and spectral data

#        glob1751 = Image.open('C:/Users/Kate/Documents/hadamard/modeling/imgs/potw1751a.tif') # https://esahubble.org/images/potw1751a/
        glob1751 = Image.open(os.path.join(cwd,'data/imgs/potw1751a.tif')) # https://esahubble.org/images/potw1751a/
        glob1751= ImageOps.grayscale(glob1751)
        globy = glob1751.resize((512,512))
        img = np.asarray(globy, dtype='int64') #KJK
        # Create an array with the full wavelength data
 #       filenames = glob.glob('C:/Users/Kate/Documents/hadamard/STIS NGSL/'+'*.fits') # Find the images in a given directory
        filenames = glob.glob(os.path.join(cwd,'data/STIS NGSL/'+'*.fits')) # Find the images in a given directory
        spec1, wls = self.flux_spectra(filenames[0],lam0,lam,nm_sample)
        stellar_specs = np.zeros((len(spec1),len(filenames)))
        for i in range (0, len(filenames)):
            spec, wls = self.flux_spectra(filenames[47],lam0,lam,nm_sample) 
            stellar_specs[:,i] = spec 
        mix = np.mean(stellar_specs,axis=1)
        x,y = np.shape(img)[0], np.shape(img)[1]
        data_cube = np.ones((x,y,len(spec1)))*(mix*0.01)
        star_indices = np.where(img >45)
        
        for k in range(0, np.shape(star_indices)[1]):
            l = np.random.randint(0,48)
            spec = stellar_specs[:,l]
            px, py = star_indices[0][k], star_indices[1][k]
            p = img[px,py]
            data_cube[px,py,:]=spec*p
        print('Spectral data generated')
        sky = (data_cube)/255
        print('Data cube combined')
        return sky, wls   

 
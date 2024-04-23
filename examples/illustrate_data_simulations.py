# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 16:34:25 2022

@author: Kate
"""
import numpy as np
import sys
sys.path.append('C:/Users/Kate/Documents/hadamard/sandbox')
from astropy.io import fits
from PIL import Image, ImageOps
import glob
import cv2
import matplotlib.pyplot as plt

from hadamard_class_v2 import *
HTSI = HTSI_Models2()
from data_sim_class import *
SIM = data_sim()

#%%




ngc1614 = Image.open('C:/Users/Kate/Documents/hadamard/modeling/imgs/ngc1614.jpg') # https://esahubble.org/images/heic0810ax/
ngc1614= ImageOps.grayscale(ngc1614)
nGal = ngc1614.resize((256,256))
img = np.asarray(nGal)
gal_name = 'C:/Users/Kate/Documents/hadamard/modeling/STScI/brown/ngc_1614_spec.fits' # location of a HII region or other spectra
gal_spec = 150
# Create an array with the full wavelength data
filenames = glob.glob('C:/Users/Kate/Documents/hadamard/STIS NGSL/'+'*.fits') # Find the images in a given directory
#
spec1 = 50
spec2 = 45
spec3 = 40
spec4 = 35
spec5 = 30
stars = np.transpose(np.vstack((spec1,spec2,spec3,spec4,spec5)))
mix = np.mean(stars)
x,y = np.shape(img)[0], np.shape(img)[1]
img_new = np.copy(img)
for i in range(0,x):
    pxi = i
    for j in range(0,y):
        pyi = j
        p = img[pxi,pyi]
        img_new[pxi,pyi]= 5 #background #p * mix *0.15
gal_indices = np.where(img >15)
for k in range(0, np.shape(gal_indices)[1]):
    px, py = gal_indices[0][k], gal_indices[1][k]
    p = img[px,py]
    img_new[px,py]= 150 #background #p * mix *0.15


#%%
    
old_gals = Image.open('C:/Users/Kate/Documents/hadamard/modeling/imgs/potw1812a.jpg') # https://esahubble.org/images/potw1812a/
old_gals= ImageOps.grayscale(old_gals)
gal2 = old_gals.resize((1024,512))
img = np.asarray(gal2)
img = img[84:340,331:587]
gal_name = 'C:/Users/Kate/Documents/hadamard/modeling/STScI/swire_galaxies/spiral_swire_sa.fits' # location of a HII region or other spectra
# Create an array with the full wavelength data
filenames = glob.glob('C:/Users/Kate/Documents/hadamard/modeling/STScI/brown/'+'*.fits') # Find the images in a given directory
starnames = glob.glob('C:/Users/Kate/Documents/hadamard/STIS NGSL/'+'*.fits') # Find the images in a given directory
x,y = np.shape(img)[0], np.shape(img)[1]

img_new = np.copy(img)
for i in range(0,x):
    pxi = i
    for j in range(0,y):
        pyi = j
        p = img[pxi,pyi]
        img_new[pxi,pyi]= 10 #background #p * mix *0.15

star_indices = np.where(img>100)
for ii in range(0, np.shape(star_indices)[1]):
    px, py = star_indices[0][ii], star_indices[1][ii]
    p = img[px,py]
    img_new[px,py]= 50 #background 
            
gal_indices = np.where(img >70)
for k in range(0, np.shape(gal_indices)[1]):
    px, py = gal_indices[0][k], gal_indices[1][k]
    if px >41:
        if px <228:
            if py>50:
                if py<195:
                    p = img[px,py]
                    img_new[px,py]= 150 
       

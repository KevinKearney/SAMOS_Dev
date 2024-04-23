This code simulates the performance of the Hadamard Transform Spectral Imaging (HTSI) mode of a DMD-based multi-object spectrometer.
The class file 'data_sim_class.py' includes various functions to create simulate input data for the simulations. 
The class file 'hadamard_class.py' includes functions for modeling the images collected by an instrument in HTSI mode. 

'Data' folder includes data used in data_sim_class for generating input sky data cubes. It also includes some instrument specifc data for
modeling reflectance, contrast, and other properties. 
'examples' includes python scripts that utilize functions within data_sim_class and hadamard_class for various modeling cases and implementations. 

'generate_DMD_patterns_SAMOS' is a script to generate binary files to send to the DC2K DMD in SAMOS (via the DMD load pattern from image mode)
The 'mask_sets' folder contains example mask sets generated with the the script. 
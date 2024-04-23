import numpy as np
import matplotlib.pyplot as plt
 
# Generate example data for sky
sky = np.random.rand(512, 512, 200)  # Generating random data for 200 wavelengths
 
# Generate example data for htsi_data
htsi_data = np.random.rand(512, 512, 200)  # Generating random data for 200 wavelengths
 
# Display original and Hadamard transformed images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
 
# Modify i_wavelength to probe different wavelengths
i_wavelength = 100  # Change this to probe different wavelengths between 0 and 199
 
# Display original image
axes[0].imshow(sky[256-64:256+63, 256-64:256+63, i_wavelength])
axes[0].set_title("Original")
 
# Display Hadamard transformed image
axes[1].imshow(htsi_data[:, :, i_wavelength])
axes[1].set_title("Hadamard")
 
plt.tight_layout()
plt.show()
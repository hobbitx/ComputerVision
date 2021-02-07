import os
import cv2 as cv
import numpy as np
import math  
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
import statistics


def transform(img):
    dft = cv.dft(np.float32(img),flags = cv.DFT_COMPLEX_OUTPUT)
    return dft

def phases_amplitude(matrix):
    dft_shift = np.fft.fftshift(matrix)
    amplitude,phases = cv.cartToPolar(dft_shift[:,:,0],dft_shift[:,:,1])
    return phases,amplitude

def re_transform(phases,amplitude):
    x,y = cv.polarToCart(amplitude,phases)
    back = cv.merge([x, y])
    back_ishift = np.fft.ifftshift(back)
    img_back = cv.idft(back_ishift)
    img_back = cv.magnitude(img_back[:,:,0], img_back[:,:,1])
    
    return img_back


img_1 = cv.imread("p1.jpg",0)
img_2 = cv.imread("p2.jpg",0)
if img_1 is None or img_2 is None:
    sys.exit("Could not read he image.")

phase_alpha = 0.9
amplitude_alpha = 1000
phase_operation = "*"
amplitude_operation = "*"
matrix_1 = transform(img_1)
matrix_2 = transform(img_2)

phases_1,amplitudes_1 = phases_amplitude(matrix_1)
phases_2,amplitudes_2 = phases_amplitude(matrix_2)

phases_1_mod = [element * phase_alpha for element in phases_1]
amplitudes_1_mod = [element * amplitude_alpha for element in amplitudes_1]
phases_2_mod = [element * phase_alpha for element in phases_2]
amplitudes_2_mod = [element * amplitude_alpha for element in amplitudes_2]

magnitude_spectrum_1 = 20*np.log(amplitudes_1)
magnitude_spectrum_2 = 20*np.log(amplitudes_2)

img_back_P_1 = re_transform(np.float32(phases_1_mod),amplitudes_1)
img_back_M_1 = re_transform(phases_1,np.float32(amplitudes_1_mod))
new_img_P_1 = cv.normalize(img_back_P_1, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
new_img_M_1 = cv.normalize(img_back_M_1, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)

img_back_P_2 = re_transform(np.float32(phases_2_mod),amplitudes_2)
img_back_M_2 = re_transform(phases_2,np.float32(amplitudes_2_mod))
new_img_P_2 = cv.normalize(img_back_P_2, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
new_img_M_2 = cv.normalize(img_back_M_2, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)

plt.subplot(2,5,1),plt.imshow(img_1, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(2,5,2),plt.imshow(magnitude_spectrum_1, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(2,5,3),plt.imshow(phases_1, cmap = 'gray')
plt.title('Phases Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(2,5,5),plt.imshow(new_img_P_1, cmap = 'gray')
plt.title('Phases Modify('+phase_operation+str(phase_alpha)+")"), plt.xticks([]), plt.yticks([])
plt.subplot(2,5,4),plt.imshow(new_img_M_1, cmap = 'gray')
plt.title('Magnitude Modify('+amplitude_operation+str(amplitude_alpha)+")"), plt.xticks([]), plt.yticks([])

plt.subplot(2,5,6),plt.imshow(img_2, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(2,5,7),plt.imshow(magnitude_spectrum_2, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(2,5,8),plt.imshow(phases_2, cmap = 'gray')
plt.title('Phases Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(2,5,10),plt.imshow(new_img_P_2, cmap = 'gray')
plt.title('Phases Modify('+phase_operation+str(phase_alpha)+")"), plt.xticks([]), plt.yticks([])
plt.subplot(2,5,9),plt.imshow(new_img_M_2, cmap = 'gray')
plt.title('Magnitude Modify('+amplitude_operation+str(amplitude_alpha)+")"), plt.xticks([]), plt.yticks([])
mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())
plt.show()
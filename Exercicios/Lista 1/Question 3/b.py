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


img = cv.imread("messi.jpg",0)
if img is None:
    sys.exit("Could not read he image.")

matrix = transform(img)
rows,cols = img.shape
phases,amplitudes = phases_amplitude(matrix)

phases_mod = [element * 2 for element in phases]
amplitudes_mod = [element * 2 for element in amplitudes]

magnitude_spectrum = 20*np.log(amplitudes)
img_back = re_transform(np.float32(phases_mod),amplitudes)
img_back_2 = re_transform(phases,np.float32(amplitudes_mod))
new_img = cv.normalize(img_back, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
new_img_2 = cv.normalize(img_back_2, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)

plt.subplot(2,3,1),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,2),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,3),plt.imshow(phases, cmap = 'gray')
plt.title('Phases Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,6),plt.imshow(new_img, cmap = 'gray')
plt.title('Phases Modify'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,5),plt.imshow(new_img_2, cmap = 'gray')
plt.title('Magnitude Modify'), plt.xticks([]), plt.yticks([])
plt.show()
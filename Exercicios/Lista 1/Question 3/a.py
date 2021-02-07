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


img_1 = cv.imread("messi.jpg",0)
img_2 = cv.imread("flor.png",0)
if img_1 is None or img_2 is None:
    sys.exit("Could not read he image.")

matrix_1 = transform(img_1)
matrix_2 = transform(img_2)
rows,cols = img_1.shape
phases_1,amplitudes_1 = phases_amplitude(matrix_1)
phases_2,amplitudes_2 = phases_amplitude(matrix_2)

img_back = re_transform(phases_1,amplitudes_2)
new_img = cv.normalize(img_back, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
img_back_2 = re_transform(phases_2,amplitudes_1)
new_img_2 = cv.normalize(img_back_2, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)


magnitude_spectrum_1 = 20*np.log(amplitudes_1)
magnitude_spectrum_2 = 20*np.log(amplitudes_2)

plt.subplot(2,4,1),plt.imshow(img_1, cmap = 'gray')
plt.title('Input Image 1'), plt.xticks([]), plt.yticks([])
plt.subplot(2,4,5),plt.imshow(img_2, cmap = 'gray')
plt.title('Input Image 2'), plt.xticks([]), plt.yticks([])

plt.subplot(2,4,2),plt.imshow(magnitude_spectrum_1, cmap = 'gray')
plt.title('Magnitude 1'), plt.xticks([]), plt.yticks([])
plt.subplot(2,4,6),plt.imshow(magnitude_spectrum_2, cmap = 'gray')
plt.title('Magnitude 2'), plt.xticks([]), plt.yticks([])

plt.subplot(2,4,3),plt.imshow(phases_1, cmap = 'gray')
plt.title('Phase 1'), plt.xticks([]), plt.yticks([])
plt.subplot(2,4,7),plt.imshow(phases_2, cmap = 'gray')
plt.title('Phase 1'), plt.xticks([]), plt.yticks([])

plt.subplot(2,4,4),plt.imshow(new_img, cmap = 'gray')
plt.title('Mod'), plt.xticks([]), plt.yticks([])
plt.subplot(2,4,8),plt.imshow(new_img_2, cmap = 'gray')
plt.title('Mod'), plt.xticks([]), plt.yticks([])

plt.show()

cv.waitKey(0)
cv.destroyAllWindows()
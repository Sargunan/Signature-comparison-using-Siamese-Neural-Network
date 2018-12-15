from normalize import normalize_image, resize_image, crop_center, preprocess_signature
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

# Functions to load and pre-process the images:
from scipy.misc import imread, imsave
import cv2

original = cv2.imread('DataBase/U11019_354.jpg',0)

normalized = 255 - normalize_image(original, size=(952, 1360))
resized = resize_image(normalized, (170, 242))
cropped = crop_center(normalized, (80,80))

f, ax = plt.subplots(4,1, figsize=(6,15))
ax[0].imshow(original, cmap='Greys_r')
ax[1].imshow(normalized)
ax[2].imshow(resized)
ax[3].imshow(cropped)

ax[0].set_title('Original')
ax[1].set_title('Background removed/centered')
#ax[2].set_title('Resized')
ax[3].set_title('Cropped center of the image')
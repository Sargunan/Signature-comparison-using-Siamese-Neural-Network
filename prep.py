from normalize import normalize_image, resize_image, crop_center, preprocess_signature
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

# Functions to load and pre-process the images:
from scipy.misc import imread, imsave
import cv2
import os
for root, dirs, files in os.walk("Test-Signature-Dataset/Test"):
    for file in files:
        
        completefliePath = os.path.join(root, file)
        print(completefliePath)
        original = cv2.imread(completefliePath,0)
        
        normalized = 255 - normalize_image(original, size=(952, 1360))
        resized = resize_image(normalized, (170, 242))
        cropped = crop_center(resized, (100,100))
        print (file)
        cv2.imwrite(file, cropped)

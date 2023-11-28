# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 15:02:21 2023
@author: zzha962
"""
import cv2
import numpy as np
from skimage import morphology


def remove_small_objects(mask, min_size=2500):
    cleaned_mask = morphology.remove_small_objects(mask.astype(bool), min_size=min_size)
    return cleaned_mask.astype(np.uint8) * 255


# Read the original image and the binary mask
original_image = cv2.imread('1096_tp1/20_60_original.tif')
binary_mask = cv2.imread('1096_tp1_pred/20_60_original.png', cv2.IMREAD_GRAYSCALE)  

# Resize the binary mask to match the original image size
resized_mask = cv2.resize(binary_mask, (original_image.shape[1], original_image.shape[0]))
cleaned_mask = remove_small_objects(resized_mask)

# Create a 3-channel mask for overlaying on the original image
overlay_mask = np.zeros_like(original_image)
overlay_mask[:,:,1] = cleaned_mask  # Green channel


# Overlay the mask on the original image
result = cv2.addWeighted(original_image, 1, overlay_mask, 0.5, 0)

# Display the result
cv2.imshow('Segmentation Overlay', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('demo.png', result)





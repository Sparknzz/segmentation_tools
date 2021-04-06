"""
RGB forground image to Binary Masks
"""
import os
import cv2
import numpy as np
import PIL.Image

ori_img = cv2.imread('/Users/ningyupeng/development/segmentation_tools/examples/original/1.jpg')
predict_mask = cv2.imread('/Users/ningyupeng/development/segmentation_tools/examples/predict/1.png', 0)
gt_mask = cv2.imread('/Users/ningyupeng/development/segmentation_tools/examples/gt/1.png', 0)

# H, W, C = predict_mask.shape

# calculate mae / max F /
def calculateMAE(imageA, imageB):
    """
    Calculate MAE between 2 images
    np: numpy

    """
    mae = np.absolute((imageB.astype("float") - imageA.astype("float"))).mean()

    return mae


print(calculateMAE(predict_mask/255, gt_mask/255))
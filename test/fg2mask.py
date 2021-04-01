"""
RGB forground image to Binary Masks
"""
import os
import cv2
import numpy as np
import PIL.Image

ori_img = cv2.imread('/Users/ningyupeng/development/segmentation_tools/examples/1-ori.jpg')
lbl_img = cv2.imread('/Users/ningyupeng/development/segmentation_tools/examples/1_viz.png')
bg_img = cv2.imread('/Users/ningyupeng/development/segmentation_tools/examples/1-removebg.png')

assert ori_img.shape == bg_img.shape

H, W, C = bg_img.shape

mask = np.zeros((H, W))

lbl_bg = np.where(lbl_img==255, ori_img, [0, 0, 0])
# find the counters
for i in range(H):
    for j in range(W):
        if ori_img[i, j, 0] != bg_img[i, j, 0] and ori_img[i, j, 1] != bg_img[i, j, 1] and ori_img[i, j, 2] != bg_img[i, j, 2]:
            bg_img[i, j] = [0, 0, 0]

# bg_img = np.where(c1 != c2, 0, bg_img)


# idx = np.where((bg_img[:,:,0] >= 230) & (bg_img[:,:,1] >= 230) & (bg_img[:,:,2] >= 230))

# bg_img[idx[0], idx[1], :] = (0, 0, 0)
cv2.imwrite('r.png', lbl_bg)





# mask[idx[0], idx[1]] = 1



# lbl_viz = PIL.Image.open('r.png')
# lbl_viz.putpalette([0, 0, 0,
#                     255, 255, 255,
#                     255, 0, 0,
#                     255, 255, 0,
#                     255, 153, 0])
# lbl_viz.save('r1.png')

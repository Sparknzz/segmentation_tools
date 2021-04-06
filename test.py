"""
convert binary mask to labelme json
"""
import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tools import *
import json

mask_dir = './test.jpeg'
json_temp = {"version": "4.5.7", "flags": {}}

mask = cv2.imread(mask_dir, 0)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
mask = cv2.dilate(mask, kernel)
mask = cv2.resize(mask, (mask.shape[1]//4, mask.shape[0]//4))

annotation = Annotation.from_mask(mask//255, image=None)

polygon_obj, hierarchy = annotation.polygons
polygons = polygon_obj.polygons

# image = cv2.imread('t.jpeg')
# ps = [ps.reshape(-1, 2) * 4 for ps in polygons]
# ps = list(filter(lambda x: len(x)>10, ps))
# cv2.drawContours(image, ps, -1, (0, 0, 255), 1)
# cv2.imwrite('b.jpg', image)

json_temp['shapes'] = []
json_temp['imagePath'] = ''
json_temp['imageData'] = None

assert len(polygons) == len(hierarchy.reshape(-1, 4))

if len(polygons)==1:
    if hierarchy[0][0][-1] == -1:
        shape_dict = {}
        shape_dict['label'] = 'person'
        shape_dict['points'] = [[int(x[0]) * 4, int(x[1]) * 4] for x in polygons[0].reshape(-1, 2)]

        shape_dict['group_id'] = None
        shape_dict['shape_type'] = 'polygon'
        shape_dict['flag'] = {}

        json_temp['shapes'].append(shape_dict)

else:
    for p, h in zip(polygons, hierarchy.squeeze()):

        if len(p)<50:
            continue

        if h[-1] == -1:
            shape_dict = {}
            shape_dict['label'] = 'person'
            shape_dict['points'] = [[int(x[0]) * 4, int(x[1]) * 4] for x in p.reshape(-1, 2)]

            shape_dict['group_id'] = None
            shape_dict['shape_type'] = 'polygon'
            shape_dict['flag'] = {}

            json_temp['shapes'].append(shape_dict)
            # represents have parents
            # then save as person label
        else:
            # save as _background_ label
            shape_dict = {}
            shape_dict['label'] = '_background_'
            shape_dict['points'] = [[int(x[0]) * 4, int(x[1]) * 4] for x in p.reshape(-1, 2)]
            shape_dict['group_id'] = None
            shape_dict['shape_type'] = 'polygon'
            shape_dict['flag'] = {}

            json_temp['shapes'].append(shape_dict)

# json_path = os.path.join('/'.join(m_pth.replace('mask', 'image').split('/')[:-1]), json_temp['imagePath'].split('.')[0] + '.json')
# with open(json_path, 'w') as f:
#     json.dump(json_temp, f)
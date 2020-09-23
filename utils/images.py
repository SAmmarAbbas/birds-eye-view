# Author: Syed Ammar Abbas
# VGG, 2019

import cv2
import numpy as np


def resize_image_with_vps(my_img, my_vps, resize_dims):
    orig_height, orig_width, orig_channels = my_img.shape

    re_width, re_height = resize_dims

    re_img = cv2.resize(my_img, dsize=(re_width, re_height), interpolation=cv2.INTER_CUBIC)

    re_vps = np.zeros_like(my_vps)
    re_vps[:, 0] = (my_vps[:, 0] * re_width) / orig_width
    re_vps[:, 1] = (my_vps[:, 1] * re_height) / orig_height

    return re_img, re_vps

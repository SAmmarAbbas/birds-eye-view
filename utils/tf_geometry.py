# Author: Syed Ammar Abbas
# VGG, 2019

import tensorflow as tf

from math import pi


def tf_deg2rad(deg):
    pi_on_180 = pi / 180
    return deg * pi_on_180


def rotate_vps(origin, point, angle):
    """
    Rotate a point counterclockwise by a given (+ve) angle around a given origin.

    The angle should be given in radians.
    """

    ox, oy = origin
    oy = -oy

    col1 = ox + tf.cos(angle) * (point[:, 0] - ox) - tf.sin(angle) * (-point[:, 1] - oy)
    col2 = -(oy + tf.sin(angle) * (point[:, 0] - ox) + tf.cos(angle) * (-point[:, 1] - oy))
    rot_vps = tf.stack([col1, col2], axis=1)

    return rot_vps


def offset_vps(vps, offset_h, offset_w):
    offset_height = tf.cast(offset_h, dtype=tf.float64)
    offset_width = tf.cast(offset_w, dtype=tf.float64)

    col1 = vps[:, 0] - offset_width
    col2 = vps[:, 1] - offset_height
    crop_vps = tf.stack([col1, col2], axis=1)

    return crop_vps


def center_crop_vps(vps, orig_dims, crop_dims):
    image_width, image_height = orig_dims

    crop_width, crop_height = crop_dims

    offset_height = tf.round((image_height - crop_height) / 2)
    offset_width = tf.round((image_width - crop_width) / 2)

    col1 = vps[:, 0] - offset_width
    col2 = vps[:, 1] - offset_height
    crop_vps = tf.stack([col1, col2], axis=1)

    return crop_vps


def resize_vps(my_vps, orig_dims, resize_dims):
    orig_width, orig_height = orig_dims

    re_width, re_height = resize_dims

    col1 = (my_vps[:, 0] * re_width) / orig_width
    col2 = (my_vps[:, 1] * re_height) / orig_height
    re_vps = tf.stack([col1, col2], axis=1)

    return re_vps


# ref: https://stackoverflow.com/questions/5789239/calculate-largest-rectangle-in-a-rotated-rectangle#7519376
def rotatedRectWithMaxArea_tf(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle (maximal area) within the rotated rectangle.
    """
    assert w == h

    # since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a, cos_a = tf.abs(tf.sin(angle)), tf.abs(tf.cos(angle))
    # assert(side_short > 2.*sin_a*cos_a*side_long) # asserting this means that square image, and correct calculations
    # i.e. fully constrained case: crop touches all 4 sides
    cos_2a = cos_a * cos_a - sin_a * sin_a
    wr, hr = (w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a

    return wr, hr
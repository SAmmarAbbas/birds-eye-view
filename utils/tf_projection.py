# Author: Syed Ammar Abbas
# VGG, 2019

import tensorflow as tf
import numpy as np

from tensorflow.python.ops import math_ops

import utils.projection as utils_projection


def get_horizon_normal_from_points_tf(p1, p2, sphere_centre):
    q = sphere_centre

    u = p2 - p1
    pq = q - p1
    w2 = pq - tf.multiply(u, (tf.matmul(tf.reshape(pq, (1, 3)), tf.reshape(u, (3, 1))) / (tf.norm(u) ** 2)))
    point = q - w2
    return point


def get_projection_on_sphere(image_coord, sphere_centre, sphere_radius):
    no_points = image_coord.shape[0]

    point_from_sphere_centre = image_coord - sphere_centre
    length_of_point = np.linalg.norm(point_from_sphere_centre, axis=1)

    # Scale the vector so that it has length equal to the radius of the sphere:
    vector_from_center = point_from_sphere_centre * (sphere_radius / length_of_point).reshape(no_points, 1)

    return vector_from_center


def get_projection_on_sphere_tf(image_coord, sphere_centre, sphere_radius):
    no_points = image_coord.shape[0]

    point_from_sphere_centre = image_coord - sphere_centre
    length_of_point = tf.norm(point_from_sphere_centre, axis=1)

    # Scale the vector so that it has length equal to the radius of the sphere:
    vector_from_center = point_from_sphere_centre * tf.reshape(sphere_radius / length_of_point, (no_points, 1))

    return vector_from_center


def get_pointonsphere_given_sphere_2points_tf(sphere_centre, sphere_radius, p1, p2):
    # returns points with respect to sphere centre

    v1 = p1 - sphere_centre
    v2 = p2 - sphere_centre
    nor = tf.cross(v1, v2)
    nor /= nor[2]

    # by solving normal with equation of sphere where origin is 0 to find intersection point
    is_undef = tf.reduce_any(tf.is_nan(nor))

    def f1():
        return tf.zeros(3, dtype=tf.float64)

    def f2():
        def less0():
            return tf.sqrt((sphere_radius ** 2) / (nor[0] ** 2 + nor[1] ** 2 + nor[2] ** 2))

        def great0():
            return -tf.sqrt((sphere_radius ** 2) / (nor[0] ** 2 + nor[1] ** 2 + nor[2] ** 2))

        t = tf.cond(nor[1]<0, less0, great0)
        return t * nor

    point = tf.cond(is_undef, f1, f2)

    return point


def get_all_projected_from_3vps_modified_tf(vps, no_bins, img_dims, verbose=False):
    # img_dims is of form (width, height)
    width, height = img_dims

    sphere_radii, sphere_centres = utils_projection.get_sphere_params(width=width, height=height)
    sphere_radius_horx, sphere_radius_hory, sphere_radius_vpzx, sphere_radius_vpzy = sphere_radii
    sphere_centre_horx, sphere_centre_hory, sphere_centre_vpzx, sphere_centre_vpzy = sphere_centres

    # horizon's x-coordinate
    # -sphere_radius -> sphere_radius
    req_p_horx = get_pointonsphere_given_sphere_2points_tf(sphere_centre_horx, sphere_radius_horx, vps[0, :], vps[1, :])

    bins_horx = np.arange(-sphere_radius_horx / 2, sphere_radius_horx / 2, (sphere_radius_horx) / no_bins)
    if verbose:
        print(bins_horx)

    target_class_horx = math_ops.bucketize(input=req_p_horx[0], boundaries=bins_horx.tolist()) - 1
    if verbose:
        print(target_class_horx)
        print('-----------------------------------')

    # vpz's x-coordinate
    # -sphere_radius -> sphere_radius
    req_p_vpzx = get_projection_on_sphere_tf(tf.stack([tf.reshape(vps[2, 0], (1,)), [height * 2], [0]], axis=1),
                                             sphere_centre=sphere_centre_vpzx, sphere_radius=sphere_radius_vpzx)

    bins_vpzx = np.arange(-sphere_radius_vpzx, sphere_radius_vpzx, (sphere_radius_vpzx * 2) / no_bins)
    if verbose:
        print(bins_vpzx)

    target_class_vpzx = math_ops.bucketize(input=req_p_vpzx[0, 0], boundaries=bins_vpzx.tolist()) - 1
    if verbose:
        print(target_class_vpzx)
        print('-----------------------------------')

    # horizon's y-coordinate
    # -sphere_radius -> 0
    req_p_hory = get_pointonsphere_given_sphere_2points_tf(sphere_centre_hory, sphere_radius_hory, vps[0, :], vps[1, :])

    bins_hory = np.arange(0, sphere_radius_hory, (sphere_radius_hory) / no_bins)
    if verbose:
        print(bins_hory)

    target_class_hory = math_ops.bucketize(input=req_p_hory[2], boundaries=bins_hory.tolist()) - 1
    if verbose:
        print(target_class_hory)
        print('-----------------------------------')

    # vpz's y-coordinate
    # -some number (> -sphere radius) -> 0
    # Take z-coordinate of req_p_vpzy
    req_p_vpzy = get_projection_on_sphere_tf(tf.stack([[width / 2], tf.reshape(vps[2, 1], (1,)), [0]], axis=1),
                                             sphere_centre=sphere_centre_vpzy, sphere_radius=sphere_radius_vpzy)

    top_most_vanishing_point = get_projection_on_sphere(np.array([[width / 2, height, 0]]),
                                                        sphere_centre=sphere_centre_vpzy,
                                                        sphere_radius=sphere_radius_vpzy)[0, 2]

    bins_vpzy = np.arange(top_most_vanishing_point, 0, abs(top_most_vanishing_point) / no_bins)
    if verbose:
        print(bins_vpzy)

    target_class_vpzy = math_ops.bucketize(input=req_p_vpzy[0, 2], boundaries=bins_vpzy.tolist()) - 1
    if verbose:
        print(target_class_vpzy)
        print('-----------------------------------')

    if verbose:
        print(req_p_horx)  # take x-coordinate
        print(req_p_hory)  # take x-coordinate
        # print(req_p_vpzx)  # take y-coordinate
        # print(req_p_vpzy)  # take ZZZZZ-coordinate
        print('-------------------------')

    indices4 = tf.stack([target_class_horx, target_class_hory, target_class_vpzx, target_class_vpzy])
    #     print ("indices4", indices4)
    classes_map = tf.one_hot(indices4, depth=no_bins, dtype=tf.int32)

    #     return (req_p_horx,req_p_vpzx,req_p_hory,req_p_vpzy)
    return classes_map, indices4
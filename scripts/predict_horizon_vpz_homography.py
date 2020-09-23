# Author: Syed Ammar Abbas
# VGG, 2019

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from math import atan
from math import degrees
from math import pi
from math import radians
from timeit import default_timer as timer

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
# Main slim library
from tensorflow.contrib import slim

from nets import vgg
from nets import inception_v4
from utils.transformations import rotation_matrix
from utils.geometry import get_slope_intercept_from_abc_line


tf.app.flags.DEFINE_string(
    'img_path', None,
    'Path for the input image.')

tf.app.flags.DEFINE_string(
    'model_name', None,
    'Two models available for prediction (vgg-16 and inception-v4')

tf.app.flags.DEFINE_string(
    'train_dir', '',
    'Two models available for prediction (vgg-16 and inception-v4')

tf.app.flags.mark_flag_as_required('img_path')
tf.app.flags.mark_flag_as_required('model_name')

FLAGS = tf.app.flags.FLAGS


def my_softmax(np_array):
    """
    Input must be 2 dimensional.
    Softmax is applied separately on each row
    """
    max_val = np.max(np_array, axis=1, keepdims=True)
    predsoft = np.exp(np_array - max_val) / np.sum(np.exp(np_array - max_val), axis=1, keepdims=True)
    return predsoft


def abline(slope, intercept, color='r'):
    """
    Plot a line from slope and intercept
    """
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, color)


def get_vp_from_sphere_coordinate_xY(sphere_point, sphere_centre, sphere_radius):
    z_coords = sphere_radius - np.sqrt(sphere_radius ** 2 - np.sum((sphere_point) ** 2, axis=1, keepdims=True))
    sphere_point_3d = np.hstack((sphere_point, z_coords))

    y_coords = ((-sphere_radius / (sphere_point_3d[:, 2] - sphere_radius)) * (sphere_point_3d[:, 1])) + sphere_centre[1]
    x_coords = ((-sphere_radius / (sphere_point_3d[:, 2] - sphere_radius)) * (sphere_point_3d[:, 0])) + sphere_centre[0]
    return x_coords, y_coords


def get_vp_from_sphere_coordinate_xZ(sphere_point, sphere_centre, sphere_radius):
    # As didn't subtract from 'sphere_radius', so basically y_coords from centre of sphere
    y_coords = np.sqrt(sphere_radius ** 2 - np.sum(sphere_point ** 2, axis=1, keepdims=True))
    sphere_point_3d = np.hstack((sphere_point[:, 0], y_coords.squeeze(), sphere_point[:, 1])).reshape(1, -1)

    y_coords = ((-sphere_radius / (sphere_point_3d[:, 2])) * (sphere_point_3d[:, 1])) + sphere_centre[1]
    x_coords = ((-sphere_radius / (sphere_point_3d[:, 2])) * (sphere_point_3d[:, 0])) + sphere_centre[0]
    return x_coords, y_coords


def get_line_given_sphere_pointonspherenormaltoplane(sphere_centre, point):
    # adding sphere centre so that it is now in the coordinates of the world
    point = sphere_centre + point

    nor_to_plane = (point - sphere_centre)
    plane_eq = np.hstack((nor_to_plane, -np.dot(nor_to_plane, sphere_centre)))
    plane_eq /= plane_eq[2]

    pred_hor = np.hstack((plane_eq[:2], plane_eq[3]))
    pred_hor /= pred_hor[2]

    return pred_hor


def get_horvpz_from_projected_4indices_modified(output_label, all_bins, all_sphere_centres, all_sphere_radii):
    req_coords = np.zeros(4)
    input_points = np.zeros((2, 2))

    for label_no in range(4):
        ind = output_label[label_no]
        half_of_bin_size = (all_bins[label_no, 1] - all_bins[label_no, 0]) / 2
        req_coords[label_no] = all_bins[label_no, ind] + half_of_bin_size

    y_coord = -np.sqrt(all_sphere_radii[0] ** 2 - (req_coords[0] ** 2 + req_coords[1] ** 2))
    input_points[0, :] = get_line_given_sphere_pointonspherenormaltoplane(all_sphere_centres[0, :],
                                                                          [req_coords[0], y_coord, req_coords[1]])[:2]

    vpzx_xy_coords = np.array([req_coords[2], 0]).reshape(1, -1)
    input_points[1, 0] = get_vp_from_sphere_coordinate_xY(vpzx_xy_coords, sphere_centre=all_sphere_centres[2, :],
                                                          sphere_radius=all_sphere_radii[2])[0][0]

    vpzy_xZ_coords = np.array([0, req_coords[3]]).reshape(1, -1)
    input_points[1, 1] = get_vp_from_sphere_coordinate_xZ(vpzy_xZ_coords, sphere_centre=all_sphere_centres[3, :],
                                                          sphere_radius=all_sphere_radii[3])[1][0]

    return input_points


def plot_scaled_horizonvector_vpz_picture(image, horizonvector_vpz, net_dims, color='go', show_vz=False, verbose=False):
    # because we are gonna rescale horizon line to these dimensions
    re_height, re_width, re_channels = image.shape
    net_width, net_height = net_dims

    scaled_vpz = np.zeros_like(horizonvector_vpz[1, :])
    scaled_vpz[0] = horizonvector_vpz[1, 0] * re_width / net_width
    scaled_vpz[1] = horizonvector_vpz[1, 1] * re_height / net_height

    horizon_vectorform = np.hstack((horizonvector_vpz[0, :2], 1))
    horizon_vectorform[0] = horizon_vectorform[0] / (re_width / net_width)
    horizon_vectorform[1] = horizon_vectorform[1] / (re_height / net_height)
    horizon_vectorform = horizon_vectorform / horizon_vectorform[2]

    slope, intercept = get_slope_intercept_from_abc_line(horizon_vectorform)
    fig, ax = plt.subplots(1, 1)
    ax.imshow(image)
    abline(slope, intercept)
    if show_vz:
        ax.plot(scaled_vpz[0], scaled_vpz[1], color)

    if verbose:
        print("Horizon Line:", horizon_vectorform)
        print("Vertical Vanishing Point:", scaled_vpz)
    return ax


def get_intrinisic_extrinsic_params_from_horizonvector_vpz(img_dims, horizonvector_vpz, net_dims, verbose=False):
    re_width, re_height = img_dims
    net_width, net_height = net_dims

    image_centre = np.array([re_width / 2, re_height / 2, 0])

    scaled_vpz = np.zeros_like(horizonvector_vpz[1, :])
    scaled_vpz[0] = horizonvector_vpz[1, 0] * re_width / net_width
    scaled_vpz[1] = horizonvector_vpz[1, 1] * re_height / net_height

    horizon_vectorform = np.hstack((horizonvector_vpz[0, :2], 1))

    # rescaling the horizon line according to the new size of the image
    # see https://math.stackexchange.com/questions/2864486/how-does-equation-of-a-line-change-as-scale-of-axes-changes?
    # noredirect=1#comment5910386_2864489

    horizon_vectorform[0] = horizon_vectorform[0] / (re_width / net_width)
    horizon_vectorform[1] = horizon_vectorform[1] / (re_height / net_height)
    horizon_vectorform = horizon_vectorform / horizon_vectorform[2]

    if verbose:
        print("Horizon with top left as origin")
        print(horizon_vectorform)

    # Doing for getting horizon as image centre
    horizon_translate_coordz = horizon_vectorform[2] + (
        (horizon_vectorform[0] * (re_width / 2) + horizon_vectorform[1] * (re_height / 2)))
    horizon_vectorform_center = horizon_vectorform / horizon_translate_coordz
    if verbose:
        print("Horizon with image centre as origin")
        print(horizon_vectorform_center)

    # m = -a/b when line in vector form ([a, b, c] from ax+by+c=0)
    roll_from_horizon = (degrees(atan(-horizon_vectorform_center[0] / horizon_vectorform_center[1])))

    # Both parameters used for calculating fx/fy are currently measured from image centre
    if verbose:
        print("Stuff for fx")
        print((scaled_vpz[0] - image_centre[0]) / horizon_vectorform_center[0])
    fx = np.sqrt(np.abs((scaled_vpz[0] - image_centre[0]) / horizon_vectorform_center[0]))
    if verbose:
        print("Stuff for fy")
        print((scaled_vpz[1] - image_centre[1]) / horizon_vectorform_center[1])
    fy = np.sqrt(np.abs((scaled_vpz[1] - image_centre[1]) / horizon_vectorform_center[1]))

    norm_vpz = np.sqrt((scaled_vpz[0] - image_centre[0]) ** 2 + (scaled_vpz[1] - image_centre[1]) ** 2)
    my_tilt = 90 - degrees(atan(norm_vpz / fy))  # subtracted 90, so now tilt from top as well
    my_tilt = radians(my_tilt)
    if verbose:
        print("Tilt from vpz:", degrees(my_tilt))

    # y=mx+c -> c = y-mx. Line form: mx-y+c = 0
    hor_slope = - horizon_vectorform[0] / horizon_vectorform[1]
    perp_slope = -1 / hor_slope
    perp_intercept = image_centre[1] - perp_slope * image_centre[0]
    perp_eq = [perp_slope, -1, perp_intercept]
    perp_eq /= perp_eq[2]
    normal_to_hor_from_imcentre = np.cross(horizon_vectorform, perp_eq)
    normal_to_hor_from_imcentre /= normal_to_hor_from_imcentre[2]
    if verbose:
        print("normal_to_hor_from_imcentre:", normal_to_hor_from_imcentre)
    norm_hor = np.sqrt((normal_to_hor_from_imcentre[0] - image_centre[0]) ** 2 + (
                normal_to_hor_from_imcentre[1] - image_centre[1]) ** 2)
    my_tilt_hor = atan(norm_hor / fy)  # tilt from top
    if verbose:
        print("Tilt from hor:", degrees(my_tilt_hor))

    my_fx = np.sqrt(norm_hor * norm_vpz)
    if verbose:
        print("My way for fx:", my_fx)

    if verbose:
        print("Predicted:")
        print("fx:", fx, "fy:", fy, "roll:", roll_from_horizon, "tilt(rad):", my_tilt, "tilt(deg):", degrees(my_tilt))

    print("Focal Length of the camera (pixels):", fy)
    print("Roll of the camera (degrees):", roll_from_horizon)
    print("Tilt of the camera (degrees):", degrees(my_tilt))

    return fx, fy, roll_from_horizon, my_tilt


def get_overhead_hmatrix_from_4cameraparams(fx, fy, my_tilt, my_roll, img_dims, verbose=False):
    width, height = img_dims

    origin, xaxis, yaxis, zaxis = [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]
    K3x3 = np.array([[fx, 0, width / 2],
                     [0, fy, height / 2],
                     [0, 0, 1]])

    inv_K3x3 = np.linalg.inv(K3x3)
    if verbose:
        print("K3x3:\n", K3x3)

    R_overhead = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    if verbose:
        print("R_overhead:\n", R_overhead)

    R_slant = rotation_matrix((pi / 2) + my_tilt, xaxis)[:3, :3]
    if verbose:
        print("R_slant:\n", R_slant)

    R_roll = rotation_matrix(my_roll, zaxis)[:3, :3]

    middle_rotation = np.dot(R_overhead, np.dot(np.linalg.inv(R_slant), R_roll))

    overhead_hmatrix = np.dot(K3x3, np.dot(middle_rotation, inv_K3x3))
    est_range_u, est_range_v = modified_matrices_calculate_range_output_without_translation(height, width,
                                                                                            overhead_hmatrix,
                                                                                            verbose=False)

    if verbose:
        print("Estimated destination range: u=", est_range_u, "v=", est_range_v)

    moveup_camera = np.array([[1, 0, -est_range_u[0]], [0, 1, -est_range_v[0]], [0, 0, 1]])
    if verbose:
        print("moveup_camera:\n", moveup_camera)

    overhead_hmatrix = np.dot(moveup_camera, np.dot(K3x3, np.dot(middle_rotation, inv_K3x3)))
    if verbose:
        print("overhead_hmatrix:\n", overhead_hmatrix)

    return overhead_hmatrix, est_range_u, est_range_v


def get_scaled_homography(H, target_height, estimated_xrange, estimated_yrange):
    # if don't want to scale image, then pass target_height = np.inf

    current_height = estimated_yrange[1] - estimated_yrange[0]
    target_height = min(target_height, current_height)
    (tw, th) = int(np.round((estimated_xrange[1] - estimated_xrange[0]))), int(
        np.round((estimated_yrange[1] - estimated_yrange[0])))

    tr = target_height / float(th)
    target_dim = (int(tw * tr), target_height)

    scaling_matrix = np.array([[tr, 0, 0], [0, tr, 0], [0, 0, 1]])
    scaled_H = np.dot(scaling_matrix, H)

    return scaled_H, target_dim


def modified_matrices_calculate_range_output_without_translation(height, width, overhead_hmatrix,
                                                                 verbose=False):
    range_u = np.array([np.inf, -np.inf])
    range_v = np.array([np.inf, -np.inf])

    i = 0
    j = 0
    u, v, w = np.dot(overhead_hmatrix, [j, i, 1])
    u = u / w
    v = v / w
    out_upperpixel = v
    if verbose:
        print(u, v)
    range_u[0] = min(u, range_u[0])
    range_v[0] = min(v, range_v[0])
    range_u[1] = max(u, range_u[1])
    range_v[1] = max(v, range_v[1])
    i = height - 1
    j = 0
    u, v, w = np.dot(overhead_hmatrix, [j, i, 1])
    u = u / w
    v = v / w
    out_lowerpixel = v
    if verbose:
        print(u, v)
    range_u[0] = min(u, range_u[0])
    range_v[0] = min(v, range_v[0])
    range_u[1] = max(u, range_u[1])
    range_v[1] = max(v, range_v[1])
    i = 0
    j = width - 1
    u, v, w = np.dot(overhead_hmatrix, [j, i, 1])
    u = u / w
    v = v / w
    if verbose:
        print(u, v)
    range_u[0] = min(u, range_u[0])
    range_v[0] = min(v, range_v[0])
    range_u[1] = max(u, range_u[1])
    range_v[1] = max(v, range_v[1])
    i = height - 1
    j = width - 1
    u, v, w = np.dot(overhead_hmatrix, [j, i, 1])
    u = u / w
    v = v / w
    if verbose:
        print(u, v)
    range_u[0] = min(u, range_u[0])
    range_v[0] = min(v, range_v[0])
    range_u[1] = max(u, range_u[1])
    range_v[1] = max(v, range_v[1])

    range_u = np.array(range_u, dtype=np.int)
    range_v = np.array(range_v, dtype=np.int)

    # it means that while transforming, after some bottom lower image was transformed,
    # upper output pixels got greater than lower
    if out_upperpixel > out_lowerpixel:

        # range_v needs to be updated
        max_height = height * 3
        upper_range = out_lowerpixel
        best_lower = upper_range  # since out_lowerpixel was lower value than out_upperpixel
        #                           i.e. above in image than out_lowerpixel
        x_best_lower = np.inf
        x_best_upper = -np.inf

        for steps_h in range(2, height):
            temp = np.dot(overhead_hmatrix, np.vstack(
                (np.arange(0, width), np.ones((1, width)) * (height - steps_h), np.ones((1, width)))))
            temp = temp / temp[2, :]

            lower_range = temp.min(axis=1)[1]
            x_lower_range = temp.min(axis=1)[0]
            x_upper_range = temp.max(axis=1)[0]
            if x_lower_range < x_best_lower:
                x_best_lower = x_lower_range
            if x_upper_range > x_best_upper:
                x_best_upper = x_upper_range

            if (upper_range - lower_range) > max_height:  # enforcing max_height of destination image
                lower_range = upper_range - max_height
                break
            if lower_range > upper_range:
                lower_range = best_lower
                break
            if lower_range < best_lower:
                best_lower = lower_range
            if verbose:
                print(steps_h, lower_range, x_best_lower, x_best_upper)
        range_v = np.array([lower_range, upper_range], dtype=np.int)

        # for testing
        range_u = np.array([x_best_lower, x_best_upper], dtype=np.int)

    return range_u, range_v


def main(_):
    if FLAGS.model_name == 'vgg-16':
        net_width = 224
        net_height = 224
        consider_top = 41

        data = np.load('data/cnn_parameters/carlavp_label_to_horvpz_fov_pitch.npz')
        train_dir = 'data/saved_models/vgg16/model.ckpt-20227'

        _R_MEAN = 123.68
        _G_MEAN = 116.78
        _B_MEAN = 103.94
        resnet_average_channels = np.array(np.concatenate((np.tile(_R_MEAN, (net_height, net_width, 1)),
                                                           np.tile(_G_MEAN, (net_height, net_width, 1)),
                                                           np.tile(_B_MEAN, (net_height, net_width, 1))), axis=2),
                                           dtype=np.float32)
    elif FLAGS.model_name == 'inception-v4':
        net_width = 299
        net_height = 299
        consider_top = 53

        data = np.load('data/cnn_parameters/carlavp-299x299_label_to_horvpz_fov_pitch.npz')
        train_dir = 'data/saved_models/incp4/model.ckpt-17721'
    else:
        print("Invalid CNN model name specified")
        return

    if FLAGS.train_dir != '':
        train_dir = FLAGS.train_dir

    all_bins = data['all_bins']
    all_sphere_centres = data['all_sphere_centres']
    all_sphere_radii = data['all_sphere_radii']

    no_params_model = 4

    num_bins = 500

    img_path = FLAGS.img_path
    img_cv = cv2.imread(img_path)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    orig_height, orig_width, orig_channels = img_cv.shape

    my_img = cv2.resize(img_cv, dsize=(net_width, net_height), interpolation=cv2.INTER_CUBIC)

    if FLAGS.model_name == 'vgg-16':
        my_img = (np.array(my_img, np.float32))
        my_img = my_img - resnet_average_channels
    elif FLAGS.model_name == 'inception-v4':
        my_img = (np.array(my_img, np.float32)) * (1. / 255)
        my_img = (my_img - 0.5) * 2
    else:
        print("Invalid CNN model name specified")
        return

    with tf.Graph().as_default():
        tf.logging.set_verbosity(tf.logging.INFO)

        img = tf.reshape(my_img, [1, net_width, net_height, 3])

        if FLAGS.model_name == 'vgg-16':
            with slim.arg_scope(vgg.vgg_arg_scope()):
                logits, _ = vgg.vgg_16(img, num_classes=num_bins * no_params_model, is_training=False)
        elif FLAGS.model_name == 'inception-v4':
            with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
                logits, _ = inception_v4.inception_v4(img, num_classes=num_bins * no_params_model, is_training=False)
        else:
            print("Invalid CNN model name specified")
            return

        probabilities = tf.nn.softmax(logits)

        checkpoint_path = train_dir
        init_fn = slim.assign_from_checkpoint_fn(
            checkpoint_path,
            slim.get_variables_to_restore())

        with tf.Session() as sess:
            with slim.queues.QueueRunners(sess):
                sess.run(tf.initialize_local_variables())
                init_fn(sess)
                start = timer()
                np_probabilities, np_rawvals = sess.run([probabilities, logits])

                i = 0

                pred_indices = np.zeros(no_params_model, dtype=np.int)
                for ln in range(no_params_model):
                    predsoft = my_softmax(np_rawvals[i, :].reshape(no_params_model, -1)[ln, :][np.newaxis])
                    predsoft = predsoft.squeeze()

                    topindices = predsoft.argsort()[::-1][:consider_top]
                    probsindices = predsoft[topindices] / np.sum(predsoft[topindices])
                    pred_indices[ln] = np.abs(int(np.round(np.sum(probsindices * topindices))))

                estimated_input_points = get_horvpz_from_projected_4indices_modified(pred_indices[:4],
                                                                                     all_bins, all_sphere_centres,
                                                                                     all_sphere_radii)

                end = timer()

    print("Time taken: {0:.2f}s".format(end-start))
    print("Output of the code")
    print("------------------------------------------------")

    plot_scaled_horizonvector_vpz_picture(img_cv, estimated_input_points, net_dims=(net_width, net_height),
                                          color='go', show_vz=True, verbose=True)
    plt.show()

    fx, fy, roll_from_horizon, my_tilt = get_intrinisic_extrinsic_params_from_horizonvector_vpz(
        img_dims=(orig_width, orig_height),
        horizonvector_vpz=estimated_input_points,
        net_dims=(net_width, net_height),
        verbose=False)

    overhead_hmatrix, est_range_u, est_range_v = get_overhead_hmatrix_from_4cameraparams(fx=fx, fy=fy,
                                                                                         my_tilt=my_tilt,
                                                                                         my_roll=-radians(
                                                                                             roll_from_horizon),
                                                                                         img_dims=(orig_width,
                                                                                                   orig_height),
                                                                                         verbose=False)

    scaled_overhead_hmatrix, target_dim = get_scaled_homography(overhead_hmatrix, 1080 * 2, est_range_u, est_range_v)

    warped = cv2.warpPerspective(img_cv, scaled_overhead_hmatrix, dsize=target_dim, flags=cv2.INTER_CUBIC)

    plt.imshow(warped)
    # plt.xticks([])
    # plt.yticks([])
    plt.show()
    os.makedirs("output/", exist_ok=True)
    txt_file = 'output/' + img_path[img_path.rfind('/') + 1:img_path.rfind(
        '.')] + '_homography_matrix_' + FLAGS.model_name + '.txt'
    np.savetxt(txt_file, scaled_overhead_hmatrix)
    print("Homography matrix saved to the text file:", txt_file)
    print("------------------------------------------------")


if __name__ == '__main__':
    tf.app.run()

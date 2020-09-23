# Test on a few images to confirm if the loss~0

import tensorflow as tf
import numpy as np
import pickle

from nets import resnet_v1
from nets import vgg
from nets import vgg_m
from nets import inception_v1
from nets import inception_v4

from math import degrees
from math import radians
from math import atan

from functools import reduce

import utils.tf_io as util_tfio

slim = tf.contrib.slim


def my_softmax(np_array):
    """
    Input must be 2 dimensional.
    Softmax is applied separately on each row
    """
    max_val = np.max(np_array, axis=1, keepdims=True)
    predsoft = np.exp(np_array-max_val)/np.sum(np.exp(np_array-max_val), axis=1, keepdims=True)
    return predsoft


def factors(n):
    return set(reduce(list.__add__,
                      ([i, n // i] for i in range(1, int(pow(n, 0.5) + 1)) if n % i == 0)))


def get_line_given_sphere_pointonspherenormaltoplane(sphere_centre, point):
    # adding sphere centre so that it is now in the coordinates of the world
    point = sphere_centre + point

    nor_to_plane = (point - sphere_centre)
    plane_eq = np.hstack((nor_to_plane, -np.dot(nor_to_plane, sphere_centre)))
    plane_eq /= plane_eq[2]

    pred_hor = np.hstack((plane_eq[:2], plane_eq[3]))
    pred_hor /= pred_hor[2]

    return pred_hor


def get_vp_from_sphere_coordinate_xY(sphere_point, sphere_centre, sphere_radius):
    z_coords = sphere_radius - np.sqrt(sphere_radius ** 2 - np.sum(sphere_point ** 2, axis=1, keepdims=True))
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


def get_f_from_fov_imwidth(fov, imwidth):
    # fov is given in degrees
    # returns f in pixels
    return (imwidth * 0.5) / np.tan(radians(fov) * 0.5)


def get_intrinisic_extrinsic_params_from_horfov(img_dims, horizonvector, fov, net_dims, verbose=False):
    re_width, re_height = img_dims
    net_width, net_height = net_dims

    image_centre = np.array([re_width / 2, re_height / 2, 0])

    horizon_vectorform = np.hstack((horizonvector[0, :2], 1))

    horizon_vectorform[0] = horizon_vectorform[0] / (re_width / net_width)
    horizon_vectorform[1] = horizon_vectorform[1] / (re_height / net_height)
    horizon_vectorform = horizon_vectorform / horizon_vectorform[2]

    if verbose:
        print("Horizon with top left as origin")
        print(horizon_vectorform)

    # Doing for getting horizon as image centre
    horizon_translate_coordz = horizon_vectorform[2] + (
                horizon_vectorform[0] * (re_width / 2) + horizon_vectorform[1] * (re_height / 2))
    horizon_vectorform_center = horizon_vectorform / horizon_translate_coordz
    if verbose:
        print("Horizon with image centre as origin")
        print(horizon_vectorform_center)

    # m = -a/b when line in vector form ([a, b, c] from ax+by+c=0)
    roll_from_horizon = (degrees(atan(-horizon_vectorform_center[0] / horizon_vectorform_center[1])))
    # print (roll_from_horizon)

    # Both parameters used for calculating fx/fy are currently measured from image centre
    # focal length from field of view
    fx = get_f_from_fov_imwidth(fov, re_width)
    fy = get_f_from_fov_imwidth(fov, re_height)

    # y=mx+c -> c = y-mx. Line form: mx-y+c = 0
    hor_slope = - horizon_vectorform[0] / horizon_vectorform[1]
    perp_slope = -1 / hor_slope
    perp_intercept = image_centre[1] - perp_slope * image_centre[0]
    perp_eq = [perp_slope, -1, perp_intercept]
    perp_eq /= perp_eq[2]
    normal_to_hor_from_imcentre = np.cross(horizon_vectorform, perp_eq)
    normal_to_hor_from_imcentre /= normal_to_hor_from_imcentre[2]
    norm_hor = np.sqrt((normal_to_hor_from_imcentre[0] - image_centre[0]) ** 2 + (
                normal_to_hor_from_imcentre[1] - image_centre[1]) ** 2)
    my_tilt_hor = atan(norm_hor / fy)  # tilt from top
    if verbose:
        print("Tilt from hor:", degrees(my_tilt_hor))

    if verbose:
        print("Predicted:")
        print("fx:", fx, "fy:", fy, "roll:", roll_from_horizon, "tilt(rad):", my_tilt_hor, "tilt(deg):",
              degrees(my_tilt_hor))

    return fx, fy, roll_from_horizon, my_tilt_hor


def get_intrinisic_extrinsic_params_from_horizonvector_vpz(img_dims, horizonvector_vpz, net_dims, verbose=False):
    re_width, re_height = img_dims
    net_width, net_height = net_dims

    image_centre = np.array([re_width / 2, re_height / 2, 0])

    scaled_vpz = np.zeros_like(horizonvector_vpz[1, :])
    scaled_vpz[0] = horizonvector_vpz[1, 0] * re_width / net_width
    scaled_vpz[1] = horizonvector_vpz[1, 1] * re_height / net_height

    horizon_vectorform = np.hstack((horizonvector_vpz[0, :2], 1))

    # rescaling the horizon line according to the new size of the image
    # see https://math.stackexchange.com/questions/2864486/how-does-equation-of-a-line-change-
    # as-scale-of-axes-changes?noredirect=1#comment5910386_2864489

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
        print((scaled_vpz[0] - image_centre[0]) / horizon_vectorform_center[0])
    fx = np.sqrt(np.abs((scaled_vpz[0] - image_centre[0]) / horizon_vectorform_center[0]))
    if verbose:
        print((scaled_vpz[1] - image_centre[1]) / horizon_vectorform_center[1])
    fy = np.sqrt(np.abs((scaled_vpz[1] - image_centre[1]) / horizon_vectorform_center[1]))

    norm_vpz = np.sqrt((scaled_vpz[0] - image_centre[0]) ** 2 + (scaled_vpz[1] - image_centre[1]) ** 2)
    # print (norm_vpz)
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

    return fx, fy, roll_from_horizon, my_tilt


def main():
    data_path = '<train-CARLA-VP.tfrecords>'

    model_type = 'vgg-16'
    train_dir = '<saved_model_path>'
    est_label = 'horvpz'

    num_bins = 500

    sphere_params = np.load('<carlavp_label_to_horvpz_fov_pitch.npz>')
    all_bins = sphere_params['all_bins']
    all_sphere_centres = sphere_params['all_sphere_centres']
    all_sphere_radii = sphere_params['all_sphere_radii']

    if est_label == 'horfov':
        fov_bins = np.arange(15, 115, 100 / num_bins)
        half_fov_bin_size = (fov_bins[1] - fov_bins[0]) / 2

    if model_type == 'inceptionv4':
        net_width = 299
        net_height = 299
    else:
        net_width = 224
        net_height = 224
    if model_type == 'vgg-m':
        model = pickle.load(open("<vggm-tf.p>", "rb"))
        average_image = np.load('<vgg_average_image.npy>')
    elif model_type == 'resnet50' or model_type == 'vgg-16' or model_type == 'resnet101':
        _R_MEAN = 123.68
        _G_MEAN = 116.78
        _B_MEAN = 103.94
        resnet_average_channels = np.array(np.concatenate((np.tile(_R_MEAN, (net_height, net_width, 1)),
                                                           np.tile(_G_MEAN, (net_height, net_width, 1)),
                                                           np.tile(_B_MEAN, (net_height, net_width, 1))), axis=2),
                                           dtype=np.float32)
    elif model_type == 'inceptionv1' or model_type == 'inceptionv4':
        print("Nothing needs to be initialized for this cnn model")
    else:
        print("ERROR: No such CNN exists")
    if est_label == 'horfov':
        no_params_model = 3
    elif est_label == 'horvpz':
        no_params_model = 4
    else:
        print("ERROR: No such 'est_label'")

    max_batch_size = 60

    total_examples = sum(1 for _ in tf.python_io.tf_record_iterator(data_path))
    print("Total examples: ", total_examples)

    divs = np.array(list(factors(total_examples)))
    sorted_divs = divs[divs.argsort()]
    batch_size = sorted_divs[sorted_divs < max_batch_size][-1]
    print("Batch Size:", batch_size)

    ct = np.arange(11, 12, 4)

    best_avg_man_loss = np.inf

    for en, consider_top in enumerate(ct):

        total_manhattan_loss = np.zeros(5)

        with tf.Graph().as_default():
            tf.logging.set_verbosity(tf.logging.INFO)
            filename_queue = tf.train.string_input_producer([data_path])
            image, label, carla_width, carla_height = util_tfio.general_read_and_decode(filename_queue,
                                                                                        num_classes=8,
                                                                                        dtype=tf.float64)

            image = tf.image.resize_images(image, [net_width, net_height], method=tf.image.ResizeMethod.BILINEAR)

            if model_type == 'vgg-m':
                image = image - average_image
            elif model_type == 'resnet50' or model_type == 'vgg-16' or model_type == 'resnet101':
                image = image - resnet_average_channels
            elif model_type == 'inceptionv1' or model_type == 'inceptionv4':
                image = tf.cast(image, tf.float32) * (1. / 255)
                image = (image - 0.5) * 2
            else:
                print("ERROR: No such CNN exists")

            images, labels, carla_widths, carla_heights = tf.train.batch(
                [image, label, carla_width, carla_height],
                batch_size=batch_size,
                num_threads=1,
                capacity=5 * batch_size)

            print(images)

            if model_type == 'vgg-m':
                logits = vgg_m.cnn_vggm(images, num_classes=num_bins * no_params_model, model=model)
            elif model_type == 'resnet50':
                with slim.arg_scope(resnet_v1.resnet_arg_scope()) as scope:
                    logits, _ = resnet_v1.resnet_v1_50(images, num_classes=num_bins * no_params_model,
                                                       is_training=False, global_pool=True)  # , reuse=True)#
            elif model_type == 'resnet101':
                with slim.arg_scope(resnet_v1.resnet_arg_scope()) as scope:
                    logits, _ = resnet_v1.resnet_v1_101(images, num_classes=num_bins * no_params_model,
                                                        is_training=False, global_pool=True)  # , reuse=True)#
            elif model_type == 'vgg-16':
                with slim.arg_scope(vgg.vgg_arg_scope()) as scope:
                    logits, _ = vgg.vgg_16(images, num_classes=num_bins * no_params_model,
                                           is_training=False)  # , global_pool=False)#, reuse=True)#
            elif model_type == 'inceptionv1':
                with slim.arg_scope(inception_v1.inception_v1_arg_scope()) as scope:
                    logits, _ = inception_v1.inception_v1(images, num_classes=num_bins * no_params_model,
                                                          is_training=False)  # , global_pool=False)#, reuse=True)#
            elif model_type == 'inceptionv4':
                with slim.arg_scope(inception_v4.inception_v4_arg_scope()) as scope:
                    logits, _ = inception_v4.inception_v4(images, num_classes=num_bins * no_params_model,
                                                          is_training=False)  # , global_pool=False)#, reuse=True)#
            else:
                print("ERROR: No such CNN exists")

            checkpoint_path = train_dir
            init_fn = slim.assign_from_checkpoint_fn(
                checkpoint_path,
                slim.get_variables_to_restore())

            print("--------------------------------------------------------")
            print("No. of examples not evaluated because of batch size:", np.mod(total_examples, batch_size))
            print("--------------------------------------------------------")

            with tf.Session() as sess:
                with slim.queues.QueueRunners(sess):
                    sess.run(tf.initialize_local_variables())
                    init_fn(sess)

                    for loop_no in range(int(np.floor(total_examples / batch_size))):
                        np_rawpreds, np_images_raw, np_labels, np_width, np_height = sess.run([logits,
                                                                                               images,
                                                                                               labels,
                                                                                               carla_widths,
                                                                                               carla_heights])

                        for i in range(batch_size):
                            pred_indices = np.zeros(no_params_model, dtype=np.int32)
                            output_vals = np_rawpreds[i, :].squeeze().reshape(no_params_model, -1)

                            for ln in range(no_params_model):
                                predsoft = my_softmax(output_vals[ln, :][np.newaxis]).squeeze()

                                topindices = predsoft.argsort()[::-1][:consider_top]
                                probsindices = predsoft[topindices] / np.sum(predsoft[topindices])
                                pred_indices[ln] = np.abs(int(np.round(np.sum(probsindices * topindices))))

                            if est_label == 'horfov':
                                estimated_input_points = get_horvpz_from_projected_4indices_modified(
                                    np.hstack((pred_indices[:2], 0, 0)),
                                    all_bins, all_sphere_centres, all_sphere_radii)
                                my_fov = fov_bins[pred_indices[2]] + half_fov_bin_size
                                fx, fy, roll_from_horizon, my_tilt = get_intrinisic_extrinsic_params_from_horfov(
                                    img_dims=(np_width[i], np_height[i]),
                                    horizonvector=estimated_input_points,
                                    fov=my_fov,
                                    net_dims=(net_width, net_height))

                            elif est_label == 'horvpz':
                                estimated_input_points = get_horvpz_from_projected_4indices_modified(pred_indices[:4],
                                                                                                     all_bins,
                                                                                                     all_sphere_centres,
                                                                                                     all_sphere_radii)
                                fx, fy, roll_from_horizon, my_tilt = \
                                    get_intrinisic_extrinsic_params_from_horizonvector_vpz(
                                        img_dims=(np_width[i], np_height[i]),
                                        horizonvector_vpz=estimated_input_points,
                                        net_dims=(net_width, net_height))

                            my_fov_fx = degrees(np.arctan(np_width[i] / (2 * fx)) * 2)
                            my_fov_fy = degrees(np.arctan(np_width[i] / (2 * fy)) * 2)
                            my_tilt = -degrees(my_tilt)
                            roll_from_horizon = roll_from_horizon

                            gt_label = np_labels[i, :].reshape(4, -1)
                            gt_fov = gt_label[3, 0]
                            gt_pitch = gt_label[3, 1]
                            gt_roll = degrees(
                                atan((gt_label[1, 1] - gt_label[0, 1]) / (gt_label[1, 0] - gt_label[0, 0])))

                            manhattan_loss = [np.abs(my_fov_fx - gt_fov),
                                              np.abs(my_fov_fy - gt_fov),
                                              np.abs(((my_fov_fx + my_fov_fy) / 2) - gt_fov),
                                              np.abs(my_tilt - gt_pitch),
                                              np.abs(roll_from_horizon - gt_roll)]

                            total_manhattan_loss += manhattan_loss

        avg_manhattan_loss = total_manhattan_loss / total_examples

        print("ct:", consider_top, "Average manhattan loss per scalar: ", avg_manhattan_loss)
        print("-------------------------------------------------------------------")

        this_loss = np.mean(np.hstack((avg_manhattan_loss[1], avg_manhattan_loss[3:])))
        if this_loss < best_avg_man_loss:
            best_avg_man_loss = this_loss
            display_loss = [consider_top, -1, avg_manhattan_loss[1], avg_manhattan_loss[3], avg_manhattan_loss[4]]

    print("Best loss:", display_loss)


if __name__ == '__main__':
    main()

import os
import glob
import numpy as np
import cv2

import tensorflow as tf

from math import radians

from utils.running_stats import RunningStats

import utils.projection as util_projection
import utils.images as util_images


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def get_filenames_for_town(filenames, no: int):
    return [f for f in filenames if f.find('town_{}'.format(no)) >= 0]


def main():
    datasets = ['train', 'eval', 'test']
    no_datasets = len(datasets)

    carla_images_path = '<input_image_path>'
    filenames = []

    images = glob.glob(os.path.join(carla_images_path, 'images/*.png'))
    for img_path in images:
        only_filename = img_path[img_path.rfind('images\\') + len('images\\'):img_path.rfind('.')]

        filenames.append(only_filename)

    filenames = np.array(filenames)
    no_images = filenames.shape[0]
    print("Total images: {}".format(no_images))

    no_images_each_set = [12000, 1000, 1000]
    town_division = [[1, 4, 5, 6], [2], [3]]

    all_data = []

    for set_no in range(len(datasets)):
        set_filenames = [f for n in town_division[set_no] for f in get_filenames_for_town(filenames, n)]
        set_filenames = np.array(set_filenames)
        set_indices = np.random.permutation(len(set_filenames))
        no_images_set = no_images_each_set[set_no]

        set_idx = set_indices[:no_images_set]
        set_data = set_filenames[set_idx]
        all_data.append(set_data)
        print('Total number of images applicable to {} set:'.format(datasets[set_no]), len(set_filenames))
        print('Number of examples being written to {} set:'.format(datasets[set_no]), set_data.shape)

    width = 224
    height = 224

    output_path_tfrecords = '<output_tfrecords_path>'
    os.makedirs(output_path_tfrecords, exist_ok=False)

    num_bins = 500

    pitch_bins = np.arange(0, 40, (40 - 0) / num_bins)
    fov_bins = np.arange(15, 115, (115 - 15) / num_bins)

    collect_bins = []
    dist_fov = []
    dist_pitch = []
    dist_vpzy = []
    dist_hory = []

    rs_mean = RunningStats()
    rs_std = RunningStats()

    for en_it, write_data in enumerate(range(no_datasets)):

        no_examples = 0
        training_or_eval = datasets[write_data]
        is_training = False
        if training_or_eval == 'train':
            is_training = True

        tfrecord_filename = output_path_tfrecords + training_or_eval + '-CARLA-VP.tfrecords'
        if not is_training:
            eval_tfrecord_filename = output_path_tfrecords + training_or_eval + '-CARLA-VP_rawvals.tfrecords'

        writer = tf.python_io.TFRecordWriter(tfrecord_filename)
        if not is_training:
            eval_writer = tf.python_io.TFRecordWriter(eval_tfrecord_filename)

        for only_filename in all_data[write_data]:

            img_path = os.path.join(carla_images_path, 'images', only_filename + '.png')
            pkl_path = os.path.join(carla_images_path, 'params', only_filename + '.npz')

            img_cv = cv2.imread(img_path)
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

            params = np.load(pkl_path)
            vps = params['vps']
            field_of_view = params['fov']  # degrees(np.arctan(orig_width/(2*K3x3[0, 0]))*2)
            pitch_of_cam = params['pitch']
            camera_height = params['camera_height']

            dist_hory.append((vps[0, 1] + vps[1, 1]) / 2)
            dist_vpzy.append(vps[2, 1])
            dist_fov.append(field_of_view)
            dist_pitch.append(pitch_of_cam)

            if is_training:
                final_img = img_cv
                output_label = vps
                output_label = np.vstack((output_label, [field_of_view, pitch_of_cam]))
            else:
                eval_img = img_cv
                eval_vps = vps
                eval_vps = np.vstack((eval_vps, [field_of_view, pitch_of_cam]))

                # if evaluating, then directly just resize to destination image size
                final_img, final_vps = util_images.resize_image_with_vps(img_cv, vps, resize_dims=(width, height))

            rs_mean.push(np.mean(final_img))
            rs_std.push(np.std(final_img))

            # ---------------------------------------------------------------------------------------------------

            if not is_training:
                all_vps = np.concatenate((final_vps, np.zeros((3, 1))), axis=1)

                sphere_radius = width / 2
                sphere_centre = np.array([width / 2, height / 2, sphere_radius])

                p1 = all_vps[0, :]
                p2 = all_vps[1, :]
                point = util_projection.get_point_on_2pointline_normal_to_3rdpoint(p1, p2, q=sphere_centre)
                input_points = np.vstack((point, all_vps[2, :]))

                output_label, all_bins, all_sphere_centres, all_sphere_radii = \
                    util_projection.get_all_projected_from_3vps(all_vps, no_bins=num_bins, img_dims=(width, height))
                # ----------------------------------------------------------------------
                # binning the principal horizontal vanishing point
                a = np.hstack((input_points[0, :2], 0))
                b = sphere_centre
                if np.abs(final_vps[0, 0] - (width / 2)) < np.abs(final_vps[1, 0] - (width / 2)):
                    c = np.hstack((final_vps[0, :], 0))
                else:
                    c = np.hstack((final_vps[1, :], 0))
                ba = a - b
                bc = c - b
                cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                angle = np.arccos(cosine_angle)
                angle = np.sign((c - b)[0]) * angle
                angle_bin = np.digitize(angle, np.linspace(radians(-80), radians(80), num_bins)) - 1
                angle32 = np.zeros(num_bins, dtype=np.int32)
                angle32[angle_bin] = 1
                # ----------------------------------------------------------------------
                # binning field of view and pitch of cam
                bin_no_fov = np.digitize(field_of_view, fov_bins) - 1
                vec_fov = np.zeros(num_bins, dtype=np.int32)
                vec_fov[bin_no_fov] = 1

                bin_no_pitch = np.digitize(pitch_of_cam, pitch_bins) - 1
                vec_pitch = np.zeros(num_bins, dtype=np.int32)
                vec_pitch[bin_no_pitch] = 1
                # ----------------------------------------------------------------------
                eval_vps = eval_vps.ravel()

                ### collecting stats
                out_bins = output_label.argmax(axis=1)
                collect_bins.append(out_bins)
                ### ------ends---------------------------------------------------------------

            output_label = output_label.ravel()
            # ----------------------------------------------------------------------------------------
            if not is_training:
                output_label = np.hstack((output_label, angle32))

                output_label = np.hstack((output_label, vec_fov))
                output_label = np.hstack((output_label, vec_pitch))

            feature = {'label': _bytes_feature(output_label.tostring()),
                       'image': _bytes_feature(final_img.tostring()),
                       'width': _bytes_feature(np.array([final_img.shape[1]], dtype=np.int32).tostring()),
                       'height': _bytes_feature(np.array([final_img.shape[0]], dtype=np.int32).tostring()),
                       'camera_height': _bytes_feature(np.array([camera_height], dtype=np.float64).tostring())
                       }

            if not is_training:
                eval_feature = {'label': _bytes_feature(eval_vps.tostring()),
                                'image': _bytes_feature(eval_img.tostring()),
                                'width': _bytes_feature(np.array([eval_img.shape[1]], dtype=np.int32).tostring()),
                                'height': _bytes_feature(np.array([eval_img.shape[0]], dtype=np.int32).tostring()),
                                'camera_height': _bytes_feature(np.array([camera_height], dtype=np.float64).tostring())
                                }
                # Create an example protocol buffer
                eval_example = tf.train.Example(features=tf.train.Features(feature=eval_feature))
                # Writing the serialized example.
                eval_writer.write(eval_example.SerializeToString())

            # Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))

            # Writing the serialized example.
            writer.write(example.SerializeToString())
            no_examples += 1

        writer.close()
        if not is_training:
            eval_writer.close()
        print("No. of examples: ", no_examples)

    fov_pitch_bins = np.vstack((fov_bins, pitch_bins))
    np.savez(output_path_tfrecords + 'carlavp_label_to_horvpz_fov_pitch.npz',
             all_bins=all_bins, all_sphere_centres=all_sphere_centres, all_sphere_radii=all_sphere_radii,
             fov_pitch_bins=fov_pitch_bins)


if __name__ == '__main__':
    main()

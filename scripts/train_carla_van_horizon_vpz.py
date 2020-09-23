# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic training script that trains a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import sys
from functools import reduce

import numpy as np
import tensorflow as tf

from nets import mobilenet_v1
from nets import resnet_v1
from nets import vgg
from nets import inception_v1
from nets import inception_v4
from nets import vgg_m
from preprocessing import vgg_preprocessing

import utils.tf_geometry as util_tfgeometry
import utils.tf_images as util_tfimage
import utils.tf_projection as util_tfprojection
import utils.tf_io as util_tfio
import utils.tf_training as util_tftraining

slim = tf.contrib.slim

tf.app.flags.DEFINE_string(
    'train_dir', '/tmp/new_traindir/',
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_integer(
    'num_readers', 4,
    'The number of parallel readers that read data from the dataset.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are print.')

tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 600,
    'The frequency with which summaries are saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'save_interval_secs', 600,
    'The frequency with which the model is saved, in seconds.')

######################
# Optimization Flags #
######################

tf.app.flags.DEFINE_float(
    'weight_decay', 0.00004, 'The weight decay on the model weights.')

tf.app.flags.DEFINE_string(
    'optimizer', 'rmsprop',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')

tf.app.flags.DEFINE_float(
    'adadelta_rho', 0.95,
    'The decay rate for adadelta.')

tf.app.flags.DEFINE_float(
    'adagrad_initial_accumulator_value', 0.1,
    'Starting value for the AdaGrad accumulators.')

tf.app.flags.DEFINE_float(
    'adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')

tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')

tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')

tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                          'The learning rate power.')

tf.app.flags.DEFINE_float(
    'ftrl_initial_accumulator_value', 0.1,
    'Starting value for the FTRL accumulators.')

tf.app.flags.DEFINE_float(
    'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')

tf.app.flags.DEFINE_float(
    'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')

tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')

tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')

tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

#######################
# Learning Rate Flags #
#######################

tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')

tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.0001,
    'The minimal end learning rate used by a polynomial decay learning rate.')

#######################
# Dataset Flags #
#######################

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_string(
    'model_name', 'invalid', 'The name of the architecture to train.')

tf.app.flags.DEFINE_integer(
    'batch_size', 32, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer('max_number_of_steps', None,
                            'The maximum number of training steps.')

#####################
# Fine-Tuning Flags #
#####################

tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'The path to a checkpoint from which to fine-tune.')

tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', None,
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')

tf.app.flags.DEFINE_string(
    'trainable_scopes', None,
    'Comma-separated list of scopes to filter the set of variables to train.'
    'By default, None would train all the variables.')

tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', False,
    'When restoring a checkpoint would ignore missing variables.')

FLAGS = tf.app.flags.FLAGS


def factors(n):
    return set(reduce(list.__add__,
                      ([i, n // i] for i in range(1, int(pow(n, 0.5) + 1)) if n % i == 0)))


def my_softmax(np_array):
    """
    Input must be 2 dimensional. 
    Softmax is applied separately on each row
    """

    max_val = np_array.max(axis=1, keepdims=True)
    predsoft = np.exp(np_array - max_val) / np.sum(np.exp(np_array - max_val), axis=1, keepdims=True)
    return predsoft


def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    dropout_val = 0.8
    
    is_flip = True

    is_smoothing = True

    maintain_aspect_ratio = True

    min_perc = 0.90
    is_random_crops = False

    max_rotation = 0

    num_bins = 500
    no_output_params = 4
    num_classes = no_output_params * num_bins

    eval_num_classes = 7 * num_bins

    num_samples = sum(1 for _ in tf.python_io.tf_record_iterator(FLAGS.dataset_dir))
    print("No. of training examples: ", num_samples)

    assert max_rotation >= 0

    print('---------------------------------------------------------')
    print('Make sure that no. of training samples is actually ' + str(num_samples))
    print('---------------------------------------------------------')

    if FLAGS.model_name == 'inception-v4':
        net_width = 299
        net_height = 299
    else:
        net_width = 224
        net_height = 224

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        global_step = slim.create_global_step()

        data_path = FLAGS.dataset_dir
        filename_queue = tf.train.string_input_producer([data_path])
        image, label, carla_width, carla_height = util_tfio.general_read_and_decode(filename_queue,
                                                                                    num_classes=8,
                                                                                    dtype=tf.float64)
        print(image)
        print(label)

        # --------------------------------------------------------------------------------------------------------------------
        degree_angle = tf.random_uniform([], minval=-max_rotation, maxval=max_rotation, dtype=tf.float32)
        radian_angle = util_tfgeometry.tf_deg2rad(degree_angle)

        label = tf.reshape(label, (4, 2))
        # my_fov = label[3, 0]
        # my_pitch = label[3, 1]

        label = label[:3, :]

        if is_flip:
            image, bool_flip = util_tfimage.random_flip_left_right(image)

            def flip_gt():
                return tf.stack(([[tf.cast(carla_width, label.dtype)-label[1, 0], label[1, 1]],
                                  [tf.cast(carla_width, label.dtype)-label[0, 0], label[0, 1]],
                                  [tf.cast(carla_width, label.dtype)-label[2, 0], label[2, 1]]]))
            
            def gt():
                return label

            label = tf.cond(bool_flip,
                            flip_gt,
                            gt)

        if max_rotation > 0:
            # image rotation is buggy on GPU
            with tf.device('/cpu:0'):
                image = tf.contrib.image.rotate(image, radian_angle, interpolation='BILINEAR')
            max_width, max_height = util_tfgeometry.rotatedRectWithMaxArea_tf(carla_width, carla_height, radian_angle)
            max_height = tf.cast(tf.floor(max_height), tf.int32)
            max_width = tf.cast(tf.floor(max_width), tf.int32)
            print("max_width, height", max_width, max_height)
            image = tf.image.resize_image_with_crop_or_pad(image, target_height=max_height, target_width=max_width)

            rot_vps = util_tfgeometry.rotate_vps((carla_width / 2, carla_height / 2), label,
                                                 tf.cast(radian_angle, dtype=tf.float64))
            crop_rot_vps = util_tfgeometry.center_crop_vps(rot_vps, orig_dims=(carla_width, carla_height),
                                                           crop_dims=(max_width, max_height))
        else:
            max_width = carla_width
            max_height = carla_height
            crop_rot_vps = label

        if maintain_aspect_ratio:
            image, max_width, max_height = util_tfimage.square_random_crop(image, max_width, max_height)

        if not is_random_crops:
            image = tf.image.resize_images(image, [net_width, net_height], method=tf.image.ResizeMethod.BILINEAR)

            float_max_height = tf.cast(max_height, tf.float64)
            float_max_width = tf.cast(max_width, tf.float64)
            final_vps = util_tfgeometry.resize_vps(crop_rot_vps,
                                                   orig_dims=(float_max_width, float_max_height),
                                                   resize_dims=(net_width, net_height))
        else:
            rand_perc = tf.random_uniform(
                [],
                minval=min_perc,
                maxval=1.0)
            crop_height = tf.maximum(net_height,
                                     tf.cast(tf.floor(rand_perc*tf.cast(max_height, tf.float32)), dtype=tf.int32))
            crop_width = tf.maximum(net_width,
                                    tf.cast(tf.floor(rand_perc*tf.cast(max_width, tf.float32)), dtype=tf.int32))
            image, off_height, off_width = vgg_preprocessing._custom_random_crop([image], crop_height, crop_width)[0]
            image = tf.image.resize_images(image, [net_width, net_height], method=tf.image.ResizeMethod.BILINEAR)

            temp_final_vps = util_tfgeometry.offset_vps(crop_rot_vps, off_height, off_width)
            float_crop_height = tf.cast(crop_height, tf.float64)
            float_crop_width = tf.cast(crop_width, tf.float64)
            final_vps = util_tfgeometry.resize_vps(temp_final_vps,
                                                   orig_dims=(float_crop_width, float_crop_height),
                                                   resize_dims=(net_width, net_height))

        image = util_tfimage.distort_color(image,
                                           color_ordering=tf.random_uniform([], minval=0, maxval=4, dtype=tf.int32),
                                           fast_mode=False)

        # Value here, before pre-processing below will be 0-255
        if FLAGS.model_name == 'vgg-m':
            model = pickle.load(open("<vggm-tf.p>", "rb"))
            average_image = np.load('<vgg_average_image.npy>')
            image = image - average_image
        elif FLAGS.model_name == 'resnet-50' or FLAGS.model_name == 'resnet-101' or FLAGS.model_name == 'vgg-16':
            image = vgg_preprocessing.my_preprocess_image(image)
        elif FLAGS.model_name == 'mobilenet-v1' or FLAGS.model_name == 'inception-v1' or \
                FLAGS.model_name == 'inception-v4':
            image = tf.cast(image, tf.float32) * (1. / 255)
            image = (image - 0.5) * 2
        else:
            sys.exit("Invalid value for model name!")

        label = tf.reshape(final_vps, (3, 2))
        all_label = tf.concat([label, [[0], [0], [0]]], axis=1)

        output_label, output_indices = util_tfprojection.get_all_projected_from_3vps_modified_tf(all_label,
                                                                                                 no_bins=num_bins,
                                                                                                 img_dims=(net_width,
                                                                                                           net_height),
                                                                                                 verbose=False)
        
        if is_smoothing:
            stddev = 0.5

            max_indices = tf.argmax(output_label, axis=1)

            normalized = tf.distributions.Normal(loc=tf.reshape(tf.cast(max_indices, dtype=tf.float64),
                                                                (no_output_params, 1)),
                                                 scale=tf.constant(stddev, dtype=tf.float64))

            probs = normalized.prob(tf.tile(tf.reshape(tf.cast(tf.range(output_label.shape[1]),
                                                               dtype=tf.float64), (1, -1)),
                                            (no_output_params, 1)))

            act_normalized = probs/tf.reduce_sum(probs, axis=1, keepdims=True)
            label = tf.reshape(act_normalized, [-1])
        else:
            label = tf.reshape(output_label, [-1])

        print("SHAPE AT END:", image, label)
        # --------------------------------------------------------------------------------------------------------------------

        # shuffle requires 'min_after_dequeue' parameter (min to keep in queue)
        images, labels = tf.train.shuffle_batch(
            [image, label],
            batch_size=FLAGS.batch_size,
            num_threads=FLAGS.num_preprocessing_threads,
            capacity=6 * FLAGS.batch_size,
            min_after_dequeue=4 * FLAGS.batch_size)

        labels = tf.stop_gradient(labels)

        ###########################
        # Reading evaluation data #
        ###########################
        if FLAGS.model_name == 'inception-v4':
            eval_path = ''
        else:
            eval_path = '<eval-CARLA-VP.tfrecords'

        eval_max_batch_size = min(50, FLAGS.batch_size)
        no_eval_examples = sum(1 for _ in tf.python_io.tf_record_iterator(eval_path))
        divs = np.array(list(factors(no_eval_examples)))
        sorted_divs = divs[divs.argsort()]
        eval_batch_size = sorted_divs[sorted_divs < eval_max_batch_size][-1]
        print("EVALUATION BATCH SIZE:", eval_batch_size)
        print("Number of examples in evaluation dataset: ", no_eval_examples)
        eval_filename_queue = tf.train.string_input_producer([eval_path])  # , num_epochs=2)

        e_image, e_label = util_tfio.read_and_decode_evaluation(eval_filename_queue, eval_num_classes,
                                                                net_height, net_width)
        print("eval_num_classes:", eval_num_classes)

        # Value here, before pre-processing below will be 0-255
        if FLAGS.model_name == 'vgg-m':
            e_image = e_image - average_image
        elif FLAGS.model_name == 'resnet-50' or FLAGS.model_name == 'resnet-101' or FLAGS.model_name == 'vgg-16':
            e_image = vgg_preprocessing.my_preprocess_image(e_image)
        elif FLAGS.model_name == 'mobilenet-v1' or FLAGS.model_name == 'inception-v1' or \
                FLAGS.model_name == 'inception-v4':
            e_image = tf.cast(e_image, tf.float32) * (1. / 255)
            e_image = (e_image - 0.5) * 2
        else:
            sys.exit("Invalid value for model name!")

        e_images, e_labels = tf.train.batch(
            [e_image, e_label],
            batch_size=eval_batch_size,
            num_threads=FLAGS.num_preprocessing_threads,
            capacity=5 * eval_batch_size)
        # --------------------------

        print("PREFETCH_QUEUE, CAPACITY:", FLAGS.batch_size, ", NUM_THREADS:", FLAGS.num_preprocessing_threads)
        batch_queue = slim.prefetch_queue.prefetch_queue(
            [images, labels], capacity=FLAGS.batch_size, num_threads=FLAGS.num_preprocessing_threads)

        images, labels = batch_queue.dequeue()

        if FLAGS.model_name == 'vgg-m':
            logits = vgg_m.cnn_vggm(images, num_classes=num_classes, model=model)

            eval_logits = vgg_m.cnn_vggm(e_images, num_classes=num_classes, model=model, reuse=True)
        elif FLAGS.model_name == 'vgg-16':
            with slim.arg_scope(vgg.vgg_arg_scope()):
                logits, end_points = vgg.vgg_16(images, num_classes=num_classes, is_training=True,
                                                dropout_keep_prob=dropout_val)

                eval_logits, _ = vgg.vgg_16(e_images, num_classes=num_classes, is_training=False, reuse=True)
        elif FLAGS.model_name == 'resnet-50':
            with slim.arg_scope(resnet_v1.resnet_arg_scope()):
                logits, end_points = resnet_v1.resnet_v1_50(images, num_classes=num_classes, is_training=True)

                eval_logits, _ = resnet_v1.resnet_v1_50(e_images, num_classes=num_classes, is_training=False,
                                                        reuse=True)
        elif FLAGS.model_name == 'resnet-101':
            with slim.arg_scope(resnet_v1.resnet_arg_scope()):
                logits, end_points = resnet_v1.resnet_v1_101(images, num_classes=num_classes, is_training=True)

                eval_logits, _ = resnet_v1.resnet_v1_101(e_images, num_classes=num_classes, is_training=False,
                                                         reuse=True)
        elif FLAGS.model_name == 'inception-v1':
            with slim.arg_scope(inception_v1.inception_v1_arg_scope()):
                logits, end_points = inception_v1.inception_v1(images, num_classes=num_classes, is_training=True,
                                                               dropout_keep_prob=dropout_val)

                eval_logits, _ = inception_v1.inception_v1(e_images, num_classes=num_classes, is_training=False,
                                                           reuse=True)
        elif FLAGS.model_name == 'inception-v4':
            with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
                logits, end_points = inception_v4.inception_v4(images, num_classes=num_classes, is_training=True,
                                                               dropout_keep_prob=dropout_val)

                eval_logits, _ = inception_v4.inception_v4(e_images, num_classes=num_classes, is_training=False,
                                                           reuse=True)
        elif FLAGS.model_name == 'mobilenet-v1':
            with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope()):
                logits, end_points = mobilenet_v1.mobilenet_v1(images, num_classes=num_classes, is_training=True,
                                                               dropout_keep_prob=dropout_val)

                eval_logits, _ = mobilenet_v1.mobilenet_v1(e_images, num_classes=num_classes, is_training=False,
                                                           reuse=True)
        else:
            sys.exit("Invalid value for model name!")

        jumps = int(num_classes / no_output_params)
        classification_loss_1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=labels[:, :jumps], logits=logits[:, :jumps]))
        classification_loss_2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=labels[:, jumps:2 * jumps], logits=logits[:, jumps:2 * jumps]))
        classification_loss_3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=labels[:, 2 * jumps:3 * jumps], logits=logits[:, 2 * jumps:3 * jumps]))
        classification_loss_4 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=labels[:, 3 * jumps:4 * jumps], logits=logits[:, 3 * jumps:4 * jumps]))

        ##############################################################################################
        # try implementing L1 loss among both here to help visualize comparison with validation loss
        
        logits_ind = tf.argmax(tf.reshape(logits, (-1, no_output_params, num_bins)), axis=2)
        labels_ind = tf.argmax(tf.reshape(labels, (-1, no_output_params, num_bins)), axis=2)
        print("Logits_ind shape:", logits_ind.shape)

        train_l1_loss = tf.reduce_sum(tf.abs(logits_ind - labels_ind))

        regularization_loss = tf.add_n(slim.losses.get_regularization_losses())
        total_loss = (classification_loss_1 + classification_loss_2 + classification_loss_3 + classification_loss_4 +
                      regularization_loss)

        print("After classification loss:")
        print(logits.shape)
        print(labels.shape)
        print("---------------------------------------")

        # Gather initial summaries.
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        # Add summaries for losses.
        # for loss in tf.get_collection(tf.GraphKeys.LOSSES, first_clone_scope):
        for loss in tf.get_collection(tf.GraphKeys.LOSSES):
            summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))

        # Add summaries for variables.
        for variable in slim.get_model_variables():
            summaries.add(tf.summary.histogram(variable.op.name, variable))

        #########################################
        # Configure the optimization procedure. #
        #########################################
        learning_rate = tf.placeholder(tf.float32, shape=[], name="learning_rate")
        optimizer = util_tftraining.configure_optimizer(learning_rate, FLAGS=FLAGS)

        print("learning rate tensor:", learning_rate)

        # Variables to train.
        variables_to_train = util_tftraining.get_variables_to_train(FLAGS=FLAGS)

        print("-----------------------------------------")
        print("variables to train: ", variables_to_train)
        print("-----------------------------------------")

        train_op = slim.learning.create_train_op(total_loss=total_loss, optimizer=optimizer,
                                                 variables_to_train=variables_to_train, global_step=global_step)

        if classification_loss_1 is not None:
            tf.summary.scalar('Losses/classification_loss_1', classification_loss_1)
        if classification_loss_2 is not None:
            tf.summary.scalar('Losses/classification_loss_2', classification_loss_2)
        if classification_loss_3 is not None:
            tf.summary.scalar('Losses/classification_loss_3', classification_loss_3)
        if classification_loss_4 is not None:
            tf.summary.scalar('Losses/classification_loss_4', classification_loss_4)

        if regularization_loss is not None:
            tf.summary.scalar('Losses/regularization_loss', regularization_loss)

        # Add total_loss to summary.
        summaries.add(tf.summary.scalar('total_loss', total_loss))

        summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        # Merge all summaries together.
        tf.summary.merge(list(summaries), name='summary_op')

        session_config = tf.ConfigProto()
        session_config.allow_soft_placement = True
        session_config.gpu_options.allow_growth = True

        init_fn = util_tftraining.get_init_fn(FLAGS=FLAGS)

        print("Before learning.train", flush=True)
        print("---------------------------------------------------")
        print("---------------------------------------------------")

        early_stop_epochs = 10
        no_steps_in_epoch = int(np.ceil(num_samples / FLAGS.batch_size))
        scaffold = tf.train.Scaffold(saver=tf.train.Saver(max_to_keep=early_stop_epochs + 3))

        show_eval_loss_every_steps = no_steps_in_epoch/5
        save_checkpoint_every_steps = no_steps_in_epoch/5

        with tf.train.MonitoredTrainingSession(
                master='',
                is_chief=True,
                checkpoint_dir=FLAGS.train_dir,
                scaffold=scaffold,
                hooks=None,
                chief_only_hooks=None,
                save_checkpoint_steps=save_checkpoint_every_steps,
                save_summaries_secs=FLAGS.save_summaries_secs,
                config=session_config,
                stop_grace_period_secs=120,
                log_step_count_steps=0,
                max_wait_secs=10
        ) as mon_sess:

            print("-----------------------------------------")
            if init_fn is not None:
                init_fn(mon_sess)
                print("Succesfully loaded model")
            else:
                print("A model already exists in the 'train_dir' path")
            print("-----------------------------------------")

            last_sum_train_loss = 0
            last_sum_tl1_loss = 0
            best_sum_train_loss = np.inf
            step_no = 0
            current_lr = FLAGS.learning_rate

            no_params = 7
            consider_params = 4

            consider_top = 11

            best_eval_wa = np.inf
            best_eval_epoch = 0

            while True:
                _, train_loss, tl1_loss = mon_sess.run([train_op, total_loss, train_l1_loss],
                                                       feed_dict={learning_rate: current_lr})
                last_sum_train_loss += train_loss
                last_sum_tl1_loss += tl1_loss

                epoch_no = int(np.floor((step_no * FLAGS.batch_size) / num_samples))

                if np.mod(step_no, FLAGS.log_every_n_steps) == 0:
                    print("Epoch {}, Step {}, lr={:0.5f}, Loss: {}".format(epoch_no, step_no, current_lr, train_loss),
                          flush=True)

                # calculating evaluation loss alongside as well
                if np.mod(step_no, show_eval_loss_every_steps) == 0:
                    print("--In eval block--")

                    total_l1_loss = 0
                    total_wa_loss = 0

                    for loop_no in range(int(np.floor(no_eval_examples / eval_batch_size))):
                        np_rawpreds, np_labels = mon_sess.run([eval_logits, e_labels])

                        for i in range(eval_batch_size):

                            predicted_label = np.argmax(np_rawpreds[i, :].reshape(consider_params, -1), axis=1)
                            gt_label = np.argmax(np_labels[i, :].reshape(no_params, -1)[:consider_params, :], axis=1)

                            l1_loss = np.sum(np.abs(predicted_label - gt_label))

                            wa = 0
                            for ln in range(consider_params):
                                predsoft = my_softmax(np_rawpreds[i, :].reshape(consider_params, -1)[ln, :][np.newaxis])
                                predsoft = predsoft.squeeze()
                                labsoft = np_labels[i, :].reshape(no_params, -1)[ln, :]
                                topindices = predsoft.argsort()[::-1][:consider_top]
                                probsindices = predsoft[topindices] / np.sum(predsoft[topindices])
                                wa += np.abs(int(np.round(np.sum(probsindices * topindices))) - labsoft.argmax())

                            total_l1_loss += l1_loss
                            total_wa_loss += wa

                    avg_manhattan_loss = total_l1_loss / no_eval_examples
                    avg_wa_loss = total_wa_loss / no_eval_examples

                    print("-------------------------------------------------------------------")
                    print("Average manhattan loss per scalar:", avg_manhattan_loss / consider_params)
                    print("Average manhattan loss(Weighted avg. top 10 bins)per scalar:", avg_wa_loss / consider_params)
                    print("-------------------------------------------------------------------", flush=True)

                    if avg_wa_loss < best_eval_wa:
                        best_eval_wa = avg_wa_loss
                        best_eval_epoch = epoch_no

                    if avg_wa_loss > best_eval_wa and (
                            epoch_no - best_eval_epoch) > early_stop_epochs and current_lr < 1e-3 and epoch_no > 10:
                        print("STOPPING TRAINING at epoch: ", epoch_no, ", best epoch was:", best_eval_epoch, "(step: ",
                              best_eval_epoch * num_samples / FLAGS.batch_size, ")")
                        print("Current eval_wa:", avg_wa_loss, ", best eval_wa:", best_eval_wa)
                        break

                    if step_no > 0:
                        last_sum_train_loss /= show_eval_loss_every_steps
                        last_sum_tl1_loss /= (no_steps_in_epoch*FLAGS.batch_size*no_output_params)
                        if last_sum_train_loss > best_sum_train_loss:
                            if current_lr > FLAGS.end_learning_rate:
                                print("Dividing learning rate by 10.0")
                                current_lr /= 10.0
                                best_sum_train_loss = last_sum_train_loss
                            else:
                                print("Already reached lowest possible lr i.e. ", current_lr)
                        else:
                            best_sum_train_loss = last_sum_train_loss

                        print("last_sum_train_loss:", last_sum_train_loss)
                        print("L1_train_loss:", last_sum_tl1_loss)
                        last_sum_train_loss = 0
                        last_sum_tl1_loss = 0
                #########################################################################################

                step_no += 1

                if FLAGS.max_number_of_steps is not None:
                    if step_no >= FLAGS.max_number_of_steps:
                        break

            print("Final Step {}, Loss: {}".format(step_no, train_loss))

        print("---------------------The End-----------------------")
        print("---------------------------------------------------")
        print("---------------------------------------------------")


if __name__ == '__main__':
    tf.app.run()

# Author: Syed Ammar Abbas
# VGG, 2019

import tensorflow as tf
import numpy as np


def general_read_and_decode(filename_queue, num_classes, dtype):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={'image': tf.FixedLenFeature([], tf.string),
                  'label': tf.FixedLenFeature([], tf.string),
                  'width': tf.FixedLenFeature([], tf.string),
                  'height': tf.FixedLenFeature([], tf.string)})

    # Convert from a scalar string tensor (whose single string has
    image = tf.decode_raw(features['image'], tf.uint8)
    label = tf.decode_raw(features['label'], dtype)
    img_width = tf.decode_raw(features['width'], np.int32)[0]
    img_height = tf.decode_raw(features['height'], np.int32)[0]

    image = tf.reshape(image, [img_height, img_width, 3])
    image = tf.cast(image, tf.float32)

    label = tf.reshape(label, [num_classes])

    return image, label, img_width, img_height


def legacy_read_and_decode(filename_queue, num_classes, img_width, img_height):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={'image': tf.FixedLenFeature([], tf.string),
                  'label': tf.FixedLenFeature([], tf.string)})

    # Convert from a scalar string tensor (whose single string has
    image = tf.decode_raw(features['image'], tf.uint8)
    label = tf.decode_raw(features['label'], tf.float64)

    image = tf.reshape(image, [img_height, img_width, 3])
    image.set_shape([img_height, img_width, 3])

    label = tf.reshape(label, [num_classes])
    label.set_shape([num_classes])

    image = tf.cast(image, tf.float32)

    return image, label


def read_and_decode_evaluation(filename_queue, num_classes, height, width):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={'image': tf.FixedLenFeature([], tf.string),
                  'label': tf.FixedLenFeature([], tf.string)})

    # Convert from a scalar string tensor (whose single string has
    image = tf.decode_raw(features['image'], tf.uint8)
    label = tf.decode_raw(features['label'], tf.int32)

    image = tf.reshape(image, [height, width, 3])
    image.set_shape([height, width, 3])

    label = tf.reshape(label, [num_classes])
    label.set_shape([num_classes])

    # Convert from [0, 255]
    image = tf.cast(image, tf.float32)

    return image, label
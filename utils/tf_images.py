# Author: Syed Ammar Abbas
# VGG, 2019

import tensorflow as tf

from preprocessing import vgg_preprocessing

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import control_flow_ops


def fn0(image):
    image = tf.image.random_brightness(image, max_delta=20.)  # / 255.)
    image = tf.image.random_hue(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    return image


def fn1(image):
    image = tf.image.random_brightness(image, max_delta=20.)  # / 255.)
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    image = tf.image.random_hue(image, max_delta=0.2)
    return image


def fn2(image):
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    image = tf.image.random_hue(image, max_delta=0.2)
    image = tf.image.random_brightness(image, max_delta=20.)  # / 255.)
    return image


def fn3(image):
    image = tf.image.random_hue(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    image = tf.image.random_brightness(image, max_delta=20.)  # / 255.)
    return image


def distort_color(image, color_ordering=0, fast_mode=True, scope=None):
    """Distort the color of a Tensor image.
    Each color distortion is non-commutative and thus ordering of the color ops
    matters. Ideally we would randomly permute the ordering of the color ops.
    Rather then adding that level of complication, we select a distinct ordering
    of color ops for each preprocessing thread.
    Args:
      image: 3-D Tensor containing single image in [0, 1].
      color_ordering: Python int, a type of distortion (valid values: 0-3).
      fast_mode: Avoids slower ops (random_hue and random_contrast)
      scope: Optional scope for name_scope.
    Returns:
      3-D Tensor color-distorted image on range [0, 1]
    Raises:
      ValueError: if color_ordering not in [0, 3]
    """
    with tf.name_scope(scope, 'distort_color', [image]):
        pred0 = tf.equal(color_ordering, 0)
        pred1 = tf.equal(color_ordering, 1)
        pred2 = tf.equal(color_ordering, 2)

        image = tf.case([(pred0, lambda: fn0(image)), (pred1, lambda: fn1(image)), (pred2, lambda: fn2(image))],
                        default=lambda: fn3(image), exclusive=True)

        # The random_* ops do not necessarily clamp.
        return tf.clip_by_value(image, 0.0, 255.0)


def fix_image_flip_shape(image, result):
    """Set the shape to 3 dimensional if we don't know anything else.

    Args:
    image: original image size
    result: flipped or transformed image

    Returns:
    An image whose shape is at least None,None,None.
    """

    image_shape = image.get_shape()
    if image_shape == tensor_shape.unknown_shape():
        result.set_shape([None, None, None])
    else:
        result.set_shape(image_shape)
    return result


def random_flip_left_right(image, seed=None):
    with ops.name_scope(None, 'random_flip_left_right', [image]) as scope:
        image = ops.convert_to_tensor(image, name='image')
        assert len(image.shape) == 3
        uniform_random = random_ops.random_uniform([], 0, 1.0, seed=seed)
        mirror_cond = math_ops.less(uniform_random, .5)
        result, bool_val = control_flow_ops.cond(
            mirror_cond,
            lambda: (array_ops.reverse(image, [1]), True),
            lambda: (image, False),
            name=scope)
        return fix_image_flip_shape(image, result), bool_val


def wrapper_my_crop(image, offset_width, offset_height, crop_width, crop_height):
    return vgg_preprocessing._my_crop(image, offset_height, offset_width, crop_height, crop_width)


def square_random_crop(image, max_width, max_height):
    def crop_width():
        return wrapper_my_crop(image,
                               tf.random_uniform([], minval=0, maxval=max_width - max_height + 1, dtype=tf.int32),
                               0, max_height, max_height), max_height, max_height

    def crop_height():
        return wrapper_my_crop(image, 0,
                               tf.random_uniform([], minval=0, maxval=max_height - max_width + 1, dtype=tf.int32),
                               max_height, max_height), max_height, max_height

    return tf.cond(max_width > max_height, crop_width, crop_height)


def square_center_crop(image, max_width, max_height):
    def crop_width():
        return wrapper_my_crop(image, tf.cast((max_width-max_height)/2, tf.int32), 0, max_height, max_height),\
               max_height, max_height

    def crop_height():
        return wrapper_my_crop(image, 0, tf.cast((max_height-max_width)/2, tf.int32), max_height, max_height),\
               max_height, max_height

    return tf.cond(max_width > max_height, crop_width, crop_height)


def square_offset_crop(image, max_width, max_height, offset):
    def crop_width():
        return wrapper_my_crop(image, offset, 0, max_height, max_height), max_height, max_height

    def crop_height():
        return wrapper_my_crop(image, 0, offset, max_height, max_height), max_height, max_height

    return tf.cond(max_width > max_height, crop_width, crop_height)

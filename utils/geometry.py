# Author: Syed Ammar Abbas
# VGG, 2019

import numpy as np


def get_abcline_from_two_points(p1, p2):
    """
    :param p1: [x1, y1]
    :param p2: [x2, y2]
    :return: line of form [a, b, c] from equation of form ax+by+c=0
    """
    m = (p2[1] - p1[1]) / (p2[0] - p1[0])
    # y =mx+b => b = y-mx
    b = p2[1] - m * p2[0]
    # mx-y+b=0
    line = np.array([m, -1, b])
    line = line / line[2]
    return line


def scale_abcline(input_line, orig_dims, new_dims):
    """
    :param input_line: line of form [a, b, c] from equation of form ax+by+c=0
    :param orig_dims: tuple of (width, height) of original dimensions
    :param new_dims: tuple of (width, height) of new dimensions
    :return: scaled line of form [a', b', c'] from equation of form a'x+b'y+c'=0
    """
    # rescaling the horizon line according to the new size of the image
    # see https://math.stackexchange.com/questions/2864486/
    # how-does-equation-of-a-line-change-as-scale-of-axes-changes?noredirect=1#comment5910386_2864489
    horizon_vectorform = input_line.copy()
    net_width, net_height = orig_dims
    re_width, re_height = new_dims

    horizon_vectorform[0] = horizon_vectorform[0] / (re_width / net_width)
    horizon_vectorform[1] = horizon_vectorform[1] / (re_height / net_height)
    horizon_vectorform = horizon_vectorform / horizon_vectorform[2]

    return horizon_vectorform


def get_slope_intercept_from_abc_line(horizon_vectorform):
    """
    :param horizon_vectorform: line of form [a, b, c] from equation of form ax+by+c=0
    :return: slope and intercept of input line
    """
    slope = -horizon_vectorform[0]/horizon_vectorform[1]
    intercept = -horizon_vectorform[2]/horizon_vectorform[1]
    return slope, intercept

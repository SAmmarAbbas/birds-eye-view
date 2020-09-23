#!/usr/bin/env python

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import os
import sys

try:
    # Need to install CARLA and update the path here
    sys.path.append("CARLA_0.9.4\\PythonAPI\\carla-0.9.4-py3.7-win-amd64.egg")
except FileNotFoundError as e:
    raise e

import carla

import random
import argparse
import time

import cv2
import numpy as np
from timeit import default_timer as timer

from scipy.stats import truncnorm

from math import radians
from math import pi
from math import tan
from utils.transformations import rotation_matrix
from utils.weather import Weather


def to_bgra_array(image):
    """Convert a CARLA raw image to a BGRA numpy array."""
    if not isinstance(image, carla.Image):
        raise ValueError("Argument must be a carla.Image")
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    return array


def to_rgb_array(image):
    """Convert a CARLA raw image to a RGB numpy array."""
    array = to_bgra_array(image)
    # Convert BGRA to RGB.
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    return array


def get_uniform_random_ellipse(range_1, range_2, pos_corr=True):
    ellipse_width = np.max(range_1) - np.min(range_1)
    ellipse_height = np.max(range_2) - np.min(range_2)

    phi = np.random.rand() * 2 * np.pi
    rho = np.random.rand()

    if not pos_corr:
        x = np.sqrt(rho) * np.cos(phi + (np.pi / 4))
    else:
        x = np.sqrt(rho) * np.cos(phi - (np.pi / 4))
    y = np.sqrt(rho) * np.sin(phi)

    x = (x * ellipse_width / 2.0) + np.mean(range_1)
    y = (y * ellipse_height / 2.0) + np.mean(range_2)

    return x, y


def get_random_int(min_val, max_val, exclude=None):
    """"
    Notes:
    ------
    - [min_val, max_val)
    - if exclude is a scalar value, then that value is not chosen during randomization
    """
    assert (isinstance(exclude, int) or isinstance(exclude, type(None)))
    if exclude is not None:
        possible_choices = list(range(min_val, exclude)) + list(range(exclude+1, max_val))
        return np.random.choice(possible_choices)
    else:
        return np.random.randint(min_val, max_val)


def get_truncated_normal(mean, sd, low, upp):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


def get_vps_from_transform(fov, pitch, yaw, roll, window_width, window_height):
    """
    fov: field of view (degrees)
    pitch: positive means looking up (degrees)
    yaw: positive means looking right (degrees)
    roll: positive means rotating anti-clockwise (degrees)
    window_width: width of carla image
    windows_height: height of carla image
    """
    _, xaxis, yaxis, zaxis = [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]
    R_normal = rotation_matrix((pi/2), xaxis)[:3, :3]
    R_tilt = rotation_matrix(radians(abs(pitch)), xaxis)[:3, :3]
    R_yaw = rotation_matrix(radians(-yaw), yaxis)[:3, :3]
    R_roll = rotation_matrix(radians(-roll), zaxis)[:3, :3]
    fx = (window_width/2)/tan(radians(fov/2))
    K3x3 = np.array([[fx, 0, window_width/2],
                    [0, fx, window_height/2],
                    [0, 0,   1]])

    overall_rotation = np.dot(R_roll, np.dot(R_tilt, np.dot(R_yaw, R_normal)))
    p, q, r = np.dot(K3x3, overall_rotation[:, 0].reshape(3, 1)).squeeze()
    p = p/r
    q = q/r
    vp1 = [p, q]
    p, q, r = np.dot(K3x3, overall_rotation[:, 1].reshape(3, 1)).squeeze()
    p = p/r
    q = q/r
    vp2 = [p, q]
    p, q, r = np.dot(K3x3, overall_rotation[:, 2].reshape(3, 1)).squeeze()
    p = p/r
    q = q/r
    vp3 = [p, q]
    vps = np.array([vp1, vp2, vp3])

    return K3x3, vps


def save_annotated_image(image, image_path, town_no, start_frame):

    np_image = to_rgb_array(image)

    K3x3, vps = get_vps_from_transform(fov=image.fov,
                                       pitch=image.transform.rotation.pitch,
                                       yaw=image.transform.rotation.yaw,
                                       roll=image.transform.rotation.roll,
                                       window_width=image.width,
                                       window_height=image.height)

    cv2.imwrite(os.path.join(image_path, 'images', 'town_{}_frame_{}.png'
                             .format(town_no, image.frame_number-start_frame)), np_image[:, :, ::-1])
    np.savez(os.path.join(image_path, 'params', 'town_{}_frame_{}'.format(town_no, image.frame_number-start_frame)),
             vps=vps,
             K=K3x3,
             fov=image.fov,
             camera_height=image.transform.location.z,
             pitch=image.transform.rotation.pitch,
             yaw=image.transform.rotation.yaw,
             roll=image.transform.rotation.roll,
             image_width=image.width,
             image_height=image.height)

    return


def set_camera_attributes(blueprint, window_width, window_height, fov, post_process, capture_pause):
    blueprint.set_attribute('image_size_x', str(window_width))
    blueprint.set_attribute('image_size_y', str(window_height))
    blueprint.set_attribute('fov', str(fov))
    if post_process:
        blueprint.set_attribute('enable_postprocess_effects', 'true')
    else:
        blueprint.set_attribute('enable_postprocess_effects', 'false')
    blueprint.set_attribute('sensor_tick', str(capture_pause))
    return


def main():
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-n', '--number-of-vehicles',
        metavar='N',
        default=10,
        type=int,
        help='number of vehicles (default: 10)')
    argparser.add_argument(
        '-s', '--speed',
        metavar='FACTOR',
        default=1.0,
        type=float,
        help='rate at which the weather changes (default: 1.0)')
    args = argparser.parse_args()

    debug = True
    no_cameras = 8
    total_images_per_map = np.int32(np.floor(20*np.array([153, 83, 141, 189, 187, 179])))
    save_path = "<output_images_path>"

    if os.path.isdir(save_path):
        raise(FileExistsError("The output directory '{}' already exists.".format(save_path)))

    # image dimensions
    window_height = 432
    window_resolutions = np.ceil(np.array([1/1, 5/4, 4/3, 3/2, 16/10, 5/3, 16/9])*window_height)

    # camera intrinsic and extrinsic
    min_camera_height = 2
    max_camera_height = 22
    min_x_translation = -5
    max_x_translation = 5
    min_y_translation = -5
    max_y_translation = 5
    min_fov = 15
    max_fov = 115
    min_pitch = -40  # negative means camera is looking down
    max_pitch = 0
    min_yaw = -89
    max_yaw = 90
    roll_std = 5
    roll_min = -25
    roll_max = 25
    speed_factor = 5  # how fast the weather should change
    capture_pause = 100  # secs b/w camera captures (high val since a camera is used for 1 capture only, then destroyed)
    delay_interval_extra_vehicle = 2  # seconds (if more vehicles than the number of spawn points)

    os.makedirs(os.path.join(save_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'params'), exist_ok=True)

    client = carla.Client(args.host, args.port)
    client.set_timeout(2.0)
    no_images = 0

    camera_list = []
    vehicle_list = []

    measure_first_frame = True
    start_frame = -1

    try:
        available_maps = client.get_available_maps()
        available_maps = [m.split('/')[-1] for m in available_maps]
        if debug:
            print("Available maps: {}".format(available_maps))

        for town_no in range(len(available_maps)):
            world = client.load_world(available_maps[town_no])

            if measure_first_frame:
                # Get current frame number
                timestamp = world.wait_for_tick(seconds=30.0)
                start_frame = timestamp.frame_count
                measure_first_frame = False
            if debug:
                print("Starting frame number: {}".format(start_frame))

            weather = Weather(world.get_weather())

            blueprint_library = world.get_blueprint_library()
            camera_blueprint = blueprint_library.find('sensor.camera.rgb')

            roll_distribution = get_truncated_normal(mean=0, sd=roll_std, low=roll_min, upp=roll_max)

            spawn_points = list(world.get_map().get_spawn_points())
            random.shuffle(spawn_points)
            if debug:
                print("Number of spawn points: {}".format(len(spawn_points)))

            vehicle_count = len(spawn_points)

            vehicle_blueprints = world.get_blueprint_library().filter('vehicle.*')
            vehicle_blueprints = [x for x in vehicle_blueprints if not x.id.endswith('isetta')]  # isetta is not safe

            def try_spawn_random_vehicle_at(local_transform):
                blueprint = random.choice(vehicle_blueprints)
                if blueprint.has_attribute('color'):
                    color = random.choice(blueprint.get_attribute('color').recommended_values)
                    blueprint.set_attribute('color', color)
                blueprint.set_attribute('role_name', 'autopilot')
                vehicle = world.try_spawn_actor(blueprint, local_transform)
                if vehicle is not None:
                    vehicle_list.append(vehicle)
                    vehicle.set_autopilot()
                    return True
                return False

            for spawn_point in spawn_points:
                if try_spawn_random_vehicle_at(spawn_point):
                    vehicle_count -= 1
                if vehicle_count <= 0:
                    break

            if debug:
                print("Remaining vehicles to be spawned: {}".format(vehicle_count))

            while vehicle_count > 0:
                time.sleep(delay_interval_extra_vehicle)
                if try_spawn_random_vehicle_at(random.choice(spawn_points)):
                    vehicle_count -= 1

            if debug:
                print("Spawned extra vehicles.")
                print("Remaining vehicles to be spawned: {}".format(vehicle_count))

            time.sleep(5)  # wait for vehicles to properly spawn and start driving on road

            map_name = world.get_map().name
            if debug:
                print("Current Map: {}".format(map_name))

            spawn_no = 0
            no_images_per_map = 0

            while no_images_per_map < total_images_per_map[town_no]:

                start = timer()
                for i in range(no_cameras):
                    spawn_no = np.mod(spawn_no+1, len(spawn_points))

                    fov = get_random_int(min_fov, max_fov)
                    camera_height, pitch = get_uniform_random_ellipse([min_camera_height, max_camera_height],
                                                                      [min_pitch, max_pitch],
                                                                      pos_corr=False)
                    yaw = get_random_int(min_yaw, max_yaw, exclude=0)
                    roll = roll_distribution.rvs()
                    x_translation = get_random_int(min_x_translation, max_x_translation)
                    y_translation = get_random_int(min_y_translation, max_y_translation)

                    set_camera_attributes(camera_blueprint,
                                          window_width=window_resolutions[no_images % len(window_resolutions)],
                                          window_height=window_height,
                                          fov=fov,
                                          post_process=True,
                                          capture_pause=capture_pause)

                    if debug:
                        print("Image no: {}".format(no_images))

                    transform = carla.Transform(carla.Location(x=spawn_points[spawn_no].location.x+x_translation,
                                                               y=spawn_points[spawn_no].location.y+y_translation,
                                                               z=spawn_points[spawn_no].location.z+camera_height),
                                                carla.Rotation(pitch=pitch, roll=roll, yaw=float(yaw)))
                    camera = world.spawn_actor(camera_blueprint, transform, attach_to=None)
                    camera_list.append(camera)

                    # This function is called each time a new image is generated by the sensor.
                    camera.listen(lambda image: save_annotated_image(image, save_path, town_no+1, start_frame))

                    no_images += 1
                    no_images_per_map += 1
                    if no_images_per_map >= total_images_per_map[town_no]:
                        break

                print("Capturing one image from each of {} cameras".format(no_cameras))
                time.sleep(4)

                # Changing weather
                elapsed_time = timer() - start
                weather.tick(speed_factor * elapsed_time)
                world.set_weather(weather.weather)

                print('destroying actors')
                for actor in camera_list:
                    actor.destroy()

                camera_list[:] = []

    finally:
        print('destroying actors')
        for actor in camera_list:
            actor.destroy()


if __name__ == '__main__':
    main()

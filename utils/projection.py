# Author: Syed Ammar Abbas
# VGG, 2019

import numpy as np


# ref: https://stackoverflow.com/questions/9368436/3d-perpendicular-point-on-line-from-3d-point
def get_point_on_2pointline_normal_to_3rdpoint(p1, p2, q):
    """
    Params:
    -------
    p1, p2: belong to a line (3D)
    q: another point (3D)

    Returns:
    --------
    a 'point' on the line containing p1 and p2 that is normal to the point q,
    i.e. p1-p2 is perpendicular to point-q
    """
    u = p2 - p1
    pq = q - p1
    w2 = pq - np.multiply(u, (np.dot(pq, u) / (np.linalg.norm(u) ** 2)))

    point = q - w2
    return point


def get_point_on_sphere_normal_to_plane(sphere_centre, sphere_radius, p1, p2):
    """
    Params:
    -------
    p1, p2: belong to a line (assume that is on image) (3D)
    sphere_centre, sphere_radius: params of a sphere located on the image. sphere_centre (3D)

    Returns:
    --------
    a 'point' on the boundary of the sphere that is normal to the plane passing through the sphere centre
    and p1-p2 line, i.e. the plane contains the sphere_centre, p1, and p2
    Note: Its coordinates are with respect to the sphere centre
    """

    v1 = p1 - sphere_centre
    v2 = p2 - sphere_centre
    nor = np.cross(v1, v2)
    nor /= nor[2]

    # by solving normal with equation of sphere where origin is 0 to find intersection point
    t = np.sqrt((sphere_radius ** 2) / (nor[0] ** 2 + nor[1] ** 2 + nor[2] ** 2))
    point = t * nor

    return point


def get_projection_on_sphere(image_coord, sphere_centre, sphere_radius):
    """
    Params:
    -------
    image_coord: a point on image (3D)
    sphere_centre, sphere_radius: params of a sphere located on the image. sphere_centre (3D)

    Returns:
    --------
    a 'vector_from_center' on the boundary of the sphere on the line connecting image_coord with sphere_centre.
    Note: Its coordinates are with respect to the sphere centre
    """
    no_points = image_coord.shape[0]

    point_from_sphere_centre = image_coord - sphere_centre
    length_of_point = np.linalg.norm(point_from_sphere_centre, axis=1)

    # Scale the vector so that it has length equal to the radius of the sphere:
    vector_from_center = point_from_sphere_centre * (sphere_radius / length_of_point).reshape(no_points, 1)

    return vector_from_center


def get_sphere_params(width, height):
    sphere_radii = np.array([width / 2, width / 2, width / 1.5, width / 2])
    sphere_centres = np.array([[width / 2, height / 2, sphere_radii[0]],
                               [width / 2, height / 2, sphere_radii[1]],
                               [width / 2, height * 2, sphere_radii[2]],
                               [width / 2, -height, sphere_radii[3]]])
    return sphere_radii, sphere_centres


def get_pointonsphere_given_sphere_2points(sphere_centre, sphere_radius, p1, p2):
    """

    Parameters
    ----------
    sphere_centre
    sphere_radius
    p1
    p2

    Returns
    -------
    returns points with respect to sphere centre. Interpretation: Assume sphere is touching one side of image and its
    centre is directly in the middle of image. Consider p1 and p2 are horizontal vanishing  points. Then draw a plane
    including p1, p2 and sphere centre. The returned point is normal to this plane originating from the sphere centre

    """

    v1 = p1 - sphere_centre
    v2 = p2 - sphere_centre
    nor = np.cross(v1, v2)
    nor /= nor[2]

    # by solving normal with equation of sphere where origin is 0 to find intersection point
    t = np.sqrt((sphere_radius ** 2) / (nor[0] ** 2 + nor[1] ** 2 + nor[2] ** 2))
    point = t * nor

    return point


def get_all_projected_from_3vps(vps, no_bins, img_dims, verbose=False):
    # img_dims is of form (width, height)
    width, height = img_dims
    assert (vps.shape[1] == 3)

    sphere_radii, sphere_centres = get_sphere_params(width=width, height=height)
    sphere_radius_horx, sphere_radius_hory, sphere_radius_vpzx, sphere_radius_vpzy = sphere_radii
    sphere_centre_horx, sphere_centre_hory, sphere_centre_vpzx, sphere_centre_vpzy = sphere_centres

    # horizon's x-coordinate
    # -sphere_radius -> sphere_radius
    req_p_horx = get_pointonsphere_given_sphere_2points(sphere_centre_horx, sphere_radius_horx, vps[0, :], vps[1, :])

    bins_horx = np.arange(-sphere_radius_horx / 2, sphere_radius_horx / 2, (sphere_radius_horx) / no_bins)
    if verbose:
        print(bins_horx)

    target_class_horx = np.digitize(req_p_horx[0], bins_horx) - 1
    if verbose:
        print(target_class_horx)
        print('-----------------------------------')

    # vpz's x-coordinate
    # -sphere_radius -> sphere_radius
    req_p_vpzx = get_projection_on_sphere(np.array([[vps[2, 0], height * 2, 0]]),
                                          sphere_centre=sphere_centre_vpzx, sphere_radius=sphere_radius_vpzx)

    bins_vpzx = np.arange(-sphere_radius_vpzx, sphere_radius_vpzx, (sphere_radius_vpzx * 2) / no_bins)
    if verbose:
        print(bins_vpzx)

    target_class_vpzx = np.digitize(req_p_vpzx[0, 0], bins_vpzx) - 1
    if verbose:
        print(target_class_vpzx)
        print('-----------------------------------')

    # horizon's y-coordinate
    # -sphere_radius -> 0
    req_p_hory = get_pointonsphere_given_sphere_2points(sphere_centre_hory, sphere_radius_hory, vps[0, :], vps[1, :])

    #     bins_hory = np.arange(-sphere_radius_hory, 0, (sphere_radius_hory)/no_bins)
    bins_hory = np.arange(0, sphere_radius_hory, (sphere_radius_hory) / no_bins)
    if verbose:
        print(bins_hory)

    target_class_hory = np.digitize(req_p_hory[2], bins_hory) - 1  # '2' means taking z-coord
    if verbose:
        print(target_class_hory)
        print('-----------------------------------')

    # vpz's y-coordinate
    # -some number (> -sphere radius) -> 0
    # Take z-coordinate of req_p_vpzy
    req_p_vpzy = get_projection_on_sphere(np.array([[width / 2, vps[2, 1], 0]]),
                                          sphere_centre=sphere_centre_vpzy, sphere_radius=sphere_radius_vpzy)

    top_most_vanishing_point = get_projection_on_sphere(np.array([[width / 2, 4 * height / 4, 0]]),
                                                        sphere_centre=sphere_centre_vpzy,
                                                        # '2' means taking z-coord
                                                        sphere_radius=sphere_radius_vpzy)[0, 2]

    bins_vpzy = np.arange(top_most_vanishing_point, 0, abs(top_most_vanishing_point) / no_bins)
    if verbose:
        print(bins_vpzy)

    target_class_vpzy = np.digitize(req_p_vpzy[0, 2], bins_vpzy) - 1
    if verbose:
        print(target_class_vpzy)
        print('-----------------------------------')

    if verbose:
        print(req_p_horx)  # take x-coordinate
        print(req_p_hory)  # take x-coordinate
        print(req_p_vpzx)  # take y-coordinate
        print(req_p_vpzy)  # take ZZZZZ-coordinate
        print('-------------------------')

    classes_map = np.zeros((4, no_bins), dtype=np.int32)
    classes_map[0, target_class_horx] = 1
    classes_map[1, target_class_hory] = 1
    classes_map[2, target_class_vpzx] = 1
    classes_map[3, target_class_vpzy] = 1

    all_bins = np.vstack((bins_horx, bins_hory, bins_vpzx, bins_vpzy))
    all_sphere_centres = np.vstack((sphere_centre_horx, sphere_centre_hory, sphere_centre_vpzx, sphere_centre_vpzy))
    all_sphere_radii = [sphere_radius_horx, sphere_radius_hory, sphere_radius_vpzx, sphere_radius_vpzy]

    return classes_map, all_bins, all_sphere_centres, all_sphere_radii

"""
image_renderer/coordinate_convertor.py
This module provides functionality to convert coordinates between world, camera, and image systems.

"""

import numpy as np
from scipy.spatial.transform import Rotation as R
import transforms3d
import glog


def world_to_camera(world_point, camera_position, camera_rotation):
    """
    Turn a point in world coordinate system to camera coordinate system.

    Args:
    - world_point: A point in world coordinate system (x, y, z)
    - camera_position: The position of the camera in world coordinate system (x, y, z)
    - camera_rotation: The rotation matrix of the camera (3x3)

    Returns:
    - The point in camera coordinate system (x, y, z)
    """
    # Translate
    translated_point = world_point - camera_position
    # Rotate
    camera_point = np.dot(camera_rotation.T, translated_point)

    return camera_point


def camera_to_image(camera_point, fovx, image_width, image_height):
    """
    Project a point in camera coordinate system to the image plane.

    Args:
    - camera_point: A point in camera coordinate system (x, y, z)
    - fovx: The field of view of the camera (horizontal)
    - image_width: The width of the image (pixels)
    - image_height: The height of the image (pixels)

    Returns:
    - The point on the image plane (pixel_x, pixel_y)

    """
    if camera_point[0] > 0 and camera_point[0] < 1e-6:
        camera_point[0] = 1e-6
    if camera_point[0] < 0 and camera_point[0] > -1e-6:
        camera_point[0] = 1e-6
    # calculate focal length
    focal_length = (image_width / 2) / np.tan(fovx / 2)

    # project to the image plane

    x, y = (camera_point[1] * focal_length) / camera_point[0], (
        camera_point[2] * focal_length
    ) / camera_point[0]

    # convert to pixel coordinates
    pixel_x, pixel_y = int(image_width / 2 - x), int(image_height / 2 - y)

    return pixel_x, pixel_y


def adjust_camera_quaternion_constrained(
    camera_pos, focus_point, original_camera_pos, original_quaternion
):
    """
    Adjust camera quaternion based on camera position change with constrained rotation degrees of freedom

    Args:
        camera_pos: New camera position [x, y, z]
        focus_point: Focus point position [x, y, z]
        original_camera_pos: Original camera position [x, y, z]
        original_quaternion: Original camera quaternion

    Returns:
        quaternion: Adjusted camera quaternion
    """
    # Convert inputs to numpy arrays
    camera_pos = np.array(camera_pos)
    focus_point = np.array(focus_point)
    original_camera_pos = np.array(original_camera_pos)

    # Calculate position change
    delta_pos = camera_pos - original_camera_pos

    # Get original Euler angles
    original_euler = transforms3d.euler.quat2euler(original_quaternion, "sxyz")
    roll, pitch, yaw = original_euler

    # Adjust angles based on position change
    if abs(delta_pos[2]) > 1e-6:  # z-axis change
        # Only adjust pitch
        forward = focus_point - camera_pos
        pitch = -np.arctan2(forward[2], np.sqrt(forward[0] ** 2 + forward[1] ** 2))
        # Keep original roll and yaw
    elif abs(delta_pos[0]) > 1e-6 or abs(delta_pos[1]) > 1e-6:  # x or y axis change
        # Adjust pitch and yaw
        forward = focus_point - camera_pos
        pitch = -np.arctan2(forward[2], np.sqrt(forward[0] ** 2 + forward[1] ** 2))
        yaw = np.arctan2(forward[1], forward[0])
        # Keep original roll

    # Convert back to quaternion
    new_quaternion = transforms3d.euler.euler2quat(roll, pitch, yaw, "sxyz")

    return new_quaternion


def world_to_image(
    world_point, camera_position, camera_rotation, fovx, image_width, image_height
):
    """
    Turn a point in world coordinate system to the image plane.

    Args:
    - world_point: A point in world coordinate system (x, y, z)
    - camera_position: The position of the camera in world coordinate system (x, y, z)
    - camera_rotation: The rotation matrix of the camera (3x3)
    - fovx: The field of view of the camera (horizontal)
    - image_width: The width of the image (pixels)
    - image_height: The height of the image (pixels)

    Returns:
    - The point on the image plane (pixel_x, pixel_y)

    """
    #  glog.info(f'world_point: {world_point}, camera_position: {camera_position}, camera_rotation: {camera_rotation}, fovx: {fovx}, image_width: {image_width}, image_height: {image_height}, bound: {bound}')

    camera_point = world_to_camera(world_point, camera_position, camera_rotation)
    pixel_x, pixel_y = camera_to_image(camera_point, fovx, image_width, image_height)

    # if pixel_x < -image_width * bound:
    #     pixel_x = -image_width * bound
    # if pixel_x >= image_width + image_width * bound:
    #     pixel_x = image_width + image_width * bound - 1
    # if pixel_y < -image_height * bound:
    #     pixel_y = -image_height * bound
    # if pixel_y >= image_height + image_height * bound:
    #     pixel_y = image_height + image_height * bound - 1

    return np.array([pixel_x, pixel_y])


import numpy as np
import transforms3d


def world_rectangle_to_image_polygon(
    rect_points, camera_pos, camera_rotation, fovx, width, height
):

    def is_point_behind_camera(point_camera):
        return point_camera[2] <= 0

    def interpolate_point(p1, p2, z):
        t = (z - p1[2]) / (p2[2] - p1[2])
        return p1 + t * (p2 - p1)

    points_camera = []
    for point in rect_points:
        p_rel = point - camera_pos
        p_camera = camera_rotation.T @ p_rel
        p_camera = np.array([p_camera[1], p_camera[2], p_camera[0]])
        points_camera.append(p_camera)

    points_camera = np.array(points_camera)

    visible_points = []
    n_points = len(points_camera)

    for i in range(n_points):
        p1 = points_camera[i]
        p2 = points_camera[(i + 1) % n_points]

        p1_behind = is_point_behind_camera(p1)
        p2_behind = is_point_behind_camera(p2)

        if not p1_behind:
            visible_points.append(p1)

        if p1_behind != p2_behind:
            intersection = interpolate_point(p1, p2, 0.1)
            visible_points.append(intersection)

    if not visible_points:
        return None

    image_points = []
    aspect_ratio = width / height
    fovy = 2 * np.arctan(np.tan(fovx / 2) / aspect_ratio)

    for point in visible_points:
        point = np.array([point[2], point[0], point[1]])
        pixel_x, pixel_y = camera_to_image(point, fovx, width, height)

        image_points.append([pixel_x, pixel_y])

    if len(image_points) < 3:
        return None

    return np.array(image_points)

import sapien as sapien
import numpy as np
import transforms3d
from shapely.geometry import Polygon, Point
from scipy.spatial.transform import Rotation as R
from . import image_render_processor, coordinate_convertor
from src.geometry import basic_geometries
from PIL import Image, ImageColor, ImageFont, ImageDraw
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import cv2
import logging
import glog
from src.utils.config_manager import get_image_renderer_config

config = get_image_renderer_config()


def create_and_mount_camera(
    scene,
    pose: sapien.Pose,
    near: float = config.default_near,
    far: float = config.default_far,
    width: float = config.width,
    height: float = config.height,
    fovy: float = config.default_fovy,
    camera_name: str = "camera",
):

    camera_mount_actor = scene.create_actor_builder().build_kinematic(
        name=f"{camera_name}_mount"
    )
    camera = scene.add_mounted_camera(
        name=camera_name,
        mount=camera_mount_actor,
        pose=pose,
        width=width,
        height=height,
        fovy=fovy,
        near=near,
        far=far,
    )
    return camera


def render_image(scene, camera):

    scene.step()
    scene.update_render()
    camera.take_picture()
    rgba = camera.get_picture("Color")
    rgba_img = (rgba * 255).clip(0, 255).astype("uint8")
    img = Image.fromarray(rgba_img)

    return img


def solve_camera_pose_for_4_points(
    points,
    camera_xyz,
    width,
    height,
    focus_ratio,
    fovy_range=config.default_fovy_range,
    view="human_full",
):

    point_center = np.mean(points, axis=0)

    def calculate_projection_error_top(params):
        fovy, yaw = params
        camera_pose = np.array([camera_xyz[0], camera_xyz[1], camera_xyz[2]])
        camera_rotation = transforms3d.euler.euler2mat(
            0, -np.deg2rad(90), yaw, axes="sxyz"
        )

        target_points = [
            np.array([-1 / 2, 1 / 2]) * focus_ratio + np.array([1 / 2, 1 / 2]),
            np.array([1 / 2, 1 / 2]) * focus_ratio + np.array([1 / 2, 1 / 2]),
            np.array([-1 / 2, -1 / 2]) * focus_ratio + np.array([1 / 2, 1 / 2]),
            np.array([1 / 2, -1 / 2]) * focus_ratio + np.array([1 / 2, 1 / 2]),
        ]

        fovx = 2 * np.arctan(np.tan(fovy / 2) * width / height)

        image_points = [
            coordinate_convertor.world_to_image(
                points[i], camera_pose, camera_rotation, fovx, width, height
            )
            for i in range(4)
        ]
        image_points /= np.array([width, height])

        total_error = np.sum(
            [np.linalg.norm(image_points[i] - target_points[i]) for i in range(4)]
        )

        return total_error

    def calculate_projection_error(params):
        fovy, roll, pitch = params
        camera_pose = np.array([camera_xyz[0], camera_xyz[1], camera_xyz[2]])

        fixed_yaw = np.arctan2(
            point_center[1] - camera_xyz[1], point_center[0] - camera_xyz[0]
        )

        camera_rotation = transforms3d.euler.euler2mat(
            roll, pitch, fixed_yaw, axes="sxyz"
        )

        target_points = [
            np.array([-1 / 2, 1 / 2]) * focus_ratio + np.array([1 / 2, 1 / 2]),
            np.array([1 / 2, 1 / 2]) * focus_ratio + np.array([1 / 2, 1 / 2]),
            np.array([-1 / 2, -1 / 2]) * focus_ratio + np.array([1 / 2, 1 / 2]),
            np.array([1 / 2, -1 / 2]) * focus_ratio + np.array([1 / 2, 1 / 2]),
        ]

        fovx = 2 * np.arctan(np.tan(fovy / 2) * width / height)

        image_points = [
            coordinate_convertor.world_to_image(
                points[i],
                camera_pose,
                camera_rotation,
                fovx,
                width,
                height,
            )
            for i in range(4)
        ]
        image_points /= np.array([width, height])

        total_error = np.sum(
            [np.linalg.norm(image_points[i] - target_points[i]) for i in range(4)]
        )

        return total_error

    if "human" in view:
        roll_init = np.pi / 6
        pitch_init = 0
        fovy_init = (fovy_range[0] + fovy_range[1]) / 2

        bounds = [
            (fovy_range[0], fovy_range[1]),
            config.roll_range,  # a small range for roll
            (0, np.pi / 2),
        ]
        result = minimize(
            calculate_projection_error,
            x0=[fovy_init, roll_init, pitch_init],
            method="COBYLA",
            bounds=bounds,
            options={
                "ftol": config.default_scipy_minimize_ftol,
                "maxiter": config.default_scipy_minimize_maxiter,
            },
        )
        total_error = calculate_projection_error(result.x)
        return result.x, total_error
    else:
        fovy_init = (fovy_range[0] + fovy_range[1]) / 2
        yaw_init = 0
        bounds = [
            (fovy_range[0], fovy_range[1]),
            (0, np.pi * 2 - config.EPS),
        ]

        result = minimize(
            calculate_projection_error_top,
            x0=[fovy_init, yaw_init],
            method="COBYLA",
            bounds=bounds,
            options={
                "ftol": config.default_scipy_minimize_ftol,
                "maxiter": config.default_scipy_minimize_maxiter,
            },
        )
        total_error = calculate_projection_error_top(result.x)
        return result.x, total_error


# In the future, this function may be moved to scene_graph.py
def auto_get_optimal_camera_pose_for_object(
    view="top_full",  # 'top_focus', 'top_full', 'human_full', 'human_focus'
    camera_xy=config.default_camera_xy,
    z_range=config.z_range,
    object_bbox=None,
    platform_rect=None,
    width=config.width,
    height=config.height,
    fovy_range=config.default_fovy_range,
    focus_ratio=config.default_focus_ratio,
):

    key_points = None
    if "top_focus" in view:
        key_points = [object_bbox[0], object_bbox[3], object_bbox[1], object_bbox[2]]
    elif view == "human_full" or view == "top_full":
        key_points = [
            platform_rect[0],
            platform_rect[3],
            platform_rect[1],
            platform_rect[2],
        ]
    elif view == "human_focus":
        object_mid_line = (object_bbox[1][:2] + object_bbox[2][:2]) / 2 - (
            object_bbox[0][:2] + object_bbox[3][:2]
        ) / 2
        camera_to_object_line = (
            object_bbox[0][:2] + object_bbox[3][:2]
        ) / 2 - camera_xy
        if np.cross(camera_to_object_line, object_mid_line) > 0:
            key_points = [
                object_bbox[1],
                object_bbox[3],
                object_bbox[5],
                object_bbox[7],
            ]
        else:
            key_points = [
                object_bbox[0],
                object_bbox[2],
                object_bbox[4],
                object_bbox[6],
            ]
    else:
        raise ValueError("Invalid view type")

    optimal_error = 1e9
    optimal_z, optimal_roll, optimal_pitch, optimal_fovy = None, None, None, None
    optimal_yaw = None

    if "human" in view:
        #     import ipdb
        #     ipdb.set_trace()
        for z in np.linspace(z_range[0], z_range[1], 10):
            camera_xyz = [camera_xy[0], camera_xy[1], z]
            [fovy, roll, pitch], error = solve_camera_pose_for_4_points(
                key_points,
                camera_xyz,
                width,
                height,
                focus_ratio,
                fovy_range=fovy_range,
                view=view,
            )
            if error < optimal_error:
                optimal_error = error
                optimal_z, optimal_roll, optimal_pitch, optimal_fovy = (
                    z,
                    roll,
                    pitch,
                    fovy,
                )

        key_point_center = np.mean(key_points, axis=0)
        fixed_yaw = np.arctan2(
            key_point_center[1] - camera_xy[1], key_point_center[0] - camera_xy[0]
        )
        optimal_yaw = fixed_yaw
    else:

        for z in np.linspace(z_range[0], z_range[1], 10):
            camera_xyz = [camera_xy[0], camera_xy[1], z]
            [fovy, yaw], error = solve_camera_pose_for_4_points(
                key_points,
                camera_xyz,
                width,
                height,
                focus_ratio,
                fovy_range=fovy_range,
                view=view,
            )
            if error < optimal_error:
                optimal_error = error
                optimal_z, optimal_roll, optimal_pitch, optimal_fovy = (
                    z,
                    0,
                    np.deg2rad(90),
                    fovy,
                )
                optimal_yaw = yaw
        pass

    fovy_32 = 2 * np.arctan(np.tan(optimal_fovy / 2) * 2)

    optimal_pose = sapien.Pose(
        p=[camera_xy[0], camera_xy[1], optimal_z],
        q=transforms3d.euler.euler2quat(
            optimal_roll, optimal_pitch, optimal_yaw, axes="sxyz"
        ),
    )

    return optimal_pose, optimal_fovy


def clip_polygon_to_screen(points, width, height):
    """Clip polygon using Sutherland-Hodgman algorithm"""

    def clip_against_line(poly_points, x1, y1, x2, y2):
        if poly_points is None:
            return []
        """Clip against one boundary line"""
        result = []
        for i in range(len(poly_points)):
            p1 = poly_points[i]
            p2 = poly_points[(i + 1) % len(poly_points)]

            # Calculate which side of the boundary the points are on
            pos1 = (x2 - x1) * (p1[1] - y1) - (y2 - y1) * (p1[0] - x1)
            pos2 = (x2 - x1) * (p2[1] - y1) - (y2 - y1) * (p2[0] - x1)

            # Both points are inside
            if pos1 >= 0 and pos2 >= 0:
                result.append(p2)
            # First point is outside, second point is inside
            elif pos1 < 0 and pos2 >= 0:
                intersection = compute_intersection(p1, p2, [x1, y1], [x2, y2])
                result.append(intersection)
                result.append(p2)
            # First point is inside, second point is outside
            elif pos1 >= 0 and pos2 < 0:
                intersection = compute_intersection(p1, p2, [x1, y1], [x2, y2])
                result.append(intersection)

        return result

    def compute_intersection(p1, p2, p3, p4):
        """Calculate intersection point of two line segments"""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-8:
            return p1

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        return [x1 + t * (x2 - x1), y1 + t * (y2 - y1)]

    # Clip against four boundaries sequentially
    clipped = points
    clipped = clip_against_line(clipped, 0, 0, width, 0)  # Top boundary
    if not clipped:
        return []
    clipped = clip_against_line(clipped, width, 0, width, height)  # Right boundary
    if not clipped:
        return []
    clipped = clip_against_line(clipped, width, height, 0, height)  # Bottom boundary
    if not clipped:
        return []
    clipped = clip_against_line(clipped, 0, height, 0, 0)  # Left boundary

    return clipped


def draw_freespace_on_image(img, points, color, width, height, draw=True):
    """Draw clipped freespace using cv2"""
    # Convert to OpenCV format
    img_array = np.array(img)
    if img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)

    # Clip polygon
    clipped_points = clip_polygon_to_screen(points, width, height)
    if not clipped_points:
        return img

    # Create mask
    mask = np.zeros_like(img_array)

    # Convert points to integer coordinates
    points_array = np.array(clipped_points, dtype=np.int32)
    points_array = points_array.reshape((-1, 1, 2))

    # Fill polygon
    if draw:
        cv2.fillPoly(mask, [points_array], color[::-1])  # BGR order

    # Blend original image and fill
    img_array = cv2.addWeighted(img_array, 1, mask, 0.5, 0)

    # Convert back to PIL format
    if img.mode == "RGBA":
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGBA)
    else:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

    return Image.fromarray(img_array)


def get_high_contrast_colors(random_seed=2025):
    """

    Return a list of high contrast colors for visualization

    The colors are shuffled based on the random seed to ensure

    """

    deep_colors = config.high_contrast_color_list
    import random

    random.seed(random_seed)
    random.shuffle(deep_colors)  # Randomly shuffle color order

    # More colors can be added as needed
    return deep_colors


def draw_optimized_cuboid_and_guideline(
    img,
    cuboid_image_points,
    label_position,
    obj_id,
    obj_color=None,
    draw_cuboid=False,
):
    """
    Optimized drawing of bounding box and guide line

    Args:
        img: image
        cuboid_image_points: bounding box vertices
        label_position: label position
        obj_id: object ID
        obj_color: object specific color (if None, use default color)

    draw_cuboid: whether to draw the cuboid
    draw_cuboid is disabled by default, because in experiements, the cuboid is often being misunderstood as an object to be grasped.

    """

    # 2. Draw a semi-transparent bounding box
    # Create polygon vertex list for filling
    polygons = []

    # Bottom face
    bottom_face = np.array(cuboid_image_points[0:4]).astype(np.int32)
    polygons.append(bottom_face)

    # Top face
    top_face = np.array(cuboid_image_points[4:8]).astype(np.int32)
    polygons.append(top_face)

    # Side faces
    for i in range(4):
        side_face = np.array(
            [
                cuboid_image_points[i],
                cuboid_image_points[(i + 1) % 4],
                cuboid_image_points[(i + 1) % 4 + 4],
                cuboid_image_points[i + 4],
            ]
        ).astype(np.int32)
        polygons.append(side_face)

    if draw_cuboid:
        # 1. Select a specific color for each object
        if obj_color is None:
            # Use a fixed color mapping to ensure the same ID always gets the same color
            colors = get_high_contrast_colors()
            obj_color = colors[obj_id % len(colors)]
        overlay = img.copy()
        for poly in polygons:
            cv2.fillPoly(
                overlay, [poly], (obj_color[0], obj_color[1], obj_color[2], 50)
            )

        alpha = 0.3
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ]

        for edge in edges:
            pt1 = tuple(map(int, cuboid_image_points[edge[0]]))
            pt2 = tuple(map(int, cuboid_image_points[edge[1]]))

            img = draw_dashed_line(img, pt1, pt2, obj_color, 1, 5, 3)

    # 4. Add anchor point at the center of the top face
    anchor_point = np.mean(cuboid_image_points[4:8], axis=0).astype(int)
    img_array = np.array(img)
    cv2.circle(img_array, tuple(anchor_point), 4, obj_color, -1)
    img = Image.fromarray(img_array)

    # 5. Draw a curved guide line from the anchor point to the label
    img = draw_curved_guide_line(img, anchor_point, label_position, obj_color)

    return img, anchor_point


def draw_dashed_line(img, pt1, pt2, color, thickness=1, dash_length=10, gap_length=5):
    """Draw a dashed line"""
    # Convert PIL image to OpenCV format
    img_array = np.array(img)

    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5
    if dist <= 0:
        return img

    # Calculate direction vector
    dx = (pt2[0] - pt1[0]) / dist
    dy = (pt2[1] - pt1[1]) / dist

    # Draw dashed line
    gap = True
    curr_pos = 0
    while curr_pos < dist:
        curr_pt1 = (
            int(pt1[0] + dx * curr_pos + 0.5),
            int(pt1[1] + dy * curr_pos + 0.5),
        )

        # Determine segment length
        seg_length = gap_length if gap else dash_length
        next_pos = min(curr_pos + seg_length, dist)

        curr_pt2 = (
            int(pt1[0] + dx * next_pos + 0.5),
            int(pt1[1] + dy * next_pos + 0.5),
        )

        # Only draw the solid part
        if not gap:
            # Ensure coordinates are within image boundaries
            if (
                0 <= curr_pt1[0] < img_array.shape[1]
                and 0 <= curr_pt1[1] < img_array.shape[0]
                and 0 <= curr_pt2[0] < img_array.shape[1]
                and 0 <= curr_pt2[1] < img_array.shape[0]
            ):
                cv2.line(img_array, curr_pt1, curr_pt2, color, thickness)

        # Toggle state
        gap = not gap
        curr_pos = next_pos

    # Convert back to PIL image
    img = Image.fromarray(img_array)
    return img


def draw_curved_guide_line(img, start_pt, end_pt, color, thickness=4, num_segments=20):
    """Draw a curved guide line"""
    # Calculate control point
    # Use Bezier curve with control point between start and end points, slightly offset to one side
    mid_x = (start_pt[0] + end_pt[0]) / 2
    mid_y = (start_pt[1] + end_pt[1]) / 2
    img_array = np.array(img)

    # Add some offset to create curve effect
    offset_x = (end_pt[1] - start_pt[1]) * 0.2
    offset_y = (start_pt[0] - end_pt[0]) * 0.2

    control_pt = (int(mid_x + offset_x), int(mid_y + offset_y))

    # Draw Bezier curve
    prev_pt = start_pt
    for i in range(1, num_segments + 1):
        t = i / num_segments
        # Quadratic Bezier curve
        x = int(
            (1 - t) ** 2 * start_pt[0]
            + 2 * (1 - t) * t * control_pt[0]
            + t**2 * end_pt[0]
            + 0.5
        )
        y = int(
            (1 - t) ** 2 * start_pt[1]
            + 2 * (1 - t) * t * control_pt[1]
            + t**2 * end_pt[1]
            + 0.5
        )

        curr_pt = (x, y)

        # Line thickness from thick to thin, making the starting point more prominent
        curr_thickness = max(thickness, int(thickness * (1 - t * 0.7)))
        cv2.line(img_array, prev_pt, curr_pt, color, curr_thickness)

        prev_pt = curr_pt

    img = Image.fromarray(img_array)
    return img


def add_guide_line_labels(
    img,
    might_mark_object_cuboid_list,
    camera,
    width,
    height,
    colors,
    rectangle_grey=False,
):
    def optimize_sequence(seq, min_diff, min_val, max_val):
        n = len(seq)
        new_seq = np.array(seq)
        for _ in range(5):
            for i in range(n - 1):
                if new_seq[i + 1] - new_seq[i] < min_diff:
                    new_seq[i + 1] = new_seq[i] + min_diff
            shift = max(0, new_seq[-1] - max_val)
            if shift > 0:
                new_seq -= shift
            for i in range(n - 1, 0, -1):
                if new_seq[i] - new_seq[i - 1] < min_diff:
                    new_seq[i - 1] = new_seq[i] - min_diff
            shift = max(0, min_val - new_seq[0])
            if shift > 0:
                new_seq += shift

        return new_seq.tolist()

    """Main function: Draw bounding boxes and guide lines using optimized method"""
    # Store used label positions
    used_positions = []

    # Store guide line start points for each object for subsequent optimization
    anchor_points = []
    label_info = []

    # Step 1: Draw all bounding boxes first (without guide lines)
    number_idx = 0
    need_optimize_label_idx_list = []
    need_optimize_label_list = []
    for cuboid_idx, might_mark_cuboid in enumerate(might_mark_object_cuboid_list):
        if len(might_mark_cuboid) == 0:
            continue

        # Calculate bounding box vertices positions in image
        cuboid_image_points = [
            coordinate_convertor.world_to_image(
                point,
                camera.get_global_pose().p,
                transforms3d.quaternions.quat2mat(camera.get_global_pose().q),
                camera.fovx,
                width,
                height,
            )
            for point in might_mark_cuboid
        ]

        # Select color
        base_color = colors[cuboid_idx % len(colors)]

        # Calculate bounding box center point (used for label position determination)
        center_2d = np.mean(cuboid_image_points, axis=0)

        # Select best region for label placement
        region = select_best_region(center_2d, width, height)

        # Find suitable label position
        label_position, _ = find_label_position(
            region, used_positions, center_2d, 50, width, height
        )

        if _:
            need_optimize_label_idx_list.append(number_idx)
        used_positions.append(label_position)

        # Save information for step 2
        label_info.append(
            {
                "id": number_idx + 1,
                "cuboid_image_points": cuboid_image_points,
                "label_position": label_position,
                "color": base_color,
            }
        )

        number_idx += 1

    need_optimize_label_list = [
        label_info[i]["label_position"] for i in need_optimize_label_idx_list
    ]
    leveled_list = []
    leveled_id_list = []
    for h, label in enumerate(need_optimize_label_list):
        for i in range(len(leveled_list)):
            if np.abs(leveled_list[i][0][1] - label[1]) < 5:
                leveled_list[i].append(label)
                leveled_id_list[i].append(need_optimize_label_idx_list[h])

                break
            elif i == len(leveled_list) - 1:
                leveled_list.append([label])
    for i, level_list in enumerate(leveled_list):
        seq = [label[0] for label in level_list]
        optimized_seq = optimize_sequence(seq, 50, 0, width)
        for j, label in enumerate(level_list):
            id = leveled_id_list[i][j]
            label_info[id]["label_position"] = (optimized_seq[j], label[1])

    # Step 2: Optimize guide line paths and draw all elements
    # Sort first, draw closer objects first (reduce crossings)
    label_info.sort(key=lambda x: calculate_depth(x["cuboid_image_points"]))

    font = ImageFont.truetype("msmincho.ttc", int(config.font_size))
    draw = ImageDraw.Draw(img)

    for info in label_info:
        obj_id = info["id"]
        cuboid_points = info["cuboid_image_points"]
        label_position = info["label_position"]
        color = info["color"]
        import ipdb

        # Draw optimized bounding box and guide line
        img, anchor_pt = draw_optimized_cuboid_and_guideline(
            img, cuboid_points, label_position, obj_id, color
        )
        # ipdb.set_trace()
        # Draw label background (black circle)
        draw_circle = ImageDraw.Draw(img)
        draw_circle.ellipse(
            [
                (
                    label_position[0] - config.font_size * 2 // 3,
                    label_position[1] - config.font_size * 2 // 3,
                ),
                (
                    label_position[0] + config.font_size * 2 // 3,
                    label_position[1] + config.font_size * 2 // 3,
                ),
            ],
            fill=color,
        )
        # ipdb.set_trace()
        # Draw number text
        draw = ImageDraw.Draw(img)
        draw.text(
            (
                int(label_position[0] - config.font_size // 2),
                int(label_position[1] - config.font_size // 2),
            ),
            str(obj_id),
            font=font,
            fill="white",
        )
    # ipdb.set_trace()

    return img


def calculate_depth(cuboid_points):
    """Simple calculation of bounding box depth (for sorting)"""
    # Use average of Z coordinates as depth estimate
    return np.mean([p[2] for p in cuboid_points]) if len(cuboid_points[0]) > 2 else 0


def select_best_region(center_2d, image_width, image_height):
    """Select the best region for label placement"""
    x, y = center_2d

    # Calculate distances to four edges
    dist_to_top = y
    dist_to_bottom = image_height - y
    dist_to_left = x
    dist_to_right = image_width - x

    # Find the edge with minimum distance
    min_dist = min(dist_to_top, dist_to_bottom, dist_to_left, dist_to_right)

    if min_dist == dist_to_top:
        return "top"
    elif min_dist == dist_to_bottom:
        return "bottom"
    elif min_dist == dist_to_left:
        return "left"
    else:
        return "right"


def find_label_position(
    region, used_positions, center_2d, margin, image_width, image_height
):
    """Find a position in the specified region that doesn't overlap with other labels, prioritizing top/bottom positions"""
    x, y = center_2d

    # If original region is left or right sides, try to reassign to top/bottom
    if region == "left" or region == "right":
        # Determine if closer to top or bottom
        if y < image_height / 2:
            # Closer to top
            new_region = "top"
            initial_pos = (x, margin)
        else:
            # Closer to bottom
            new_region = "bottom"
            initial_pos = (x, image_height - margin)
    else:
        # Keep original top/bottom position
        if region == "top":
            initial_pos = (x, margin)
        else:  # bottom
            initial_pos = (x, image_height - margin)

    # Check if overlaps with existing labels
    position = initial_pos
    step = 20  # Label movement step
    max_tries = 30  # Maximum number of attempts

    # First try to find position in top/bottom
    for _ in range(max_tries):
        if not is_position_overlapping(position, used_positions):
            return position, 1

        # Move horizontally (for top/bottom regions)
        move_direction = 1 if np.random.random() > 0.5 else -1
        position = (position[0] + step * move_direction, position[1])

        # Ensure not going out of image bounds
        if position[0] < margin:
            position = (margin, position[1])
        elif position[0] > image_width - margin:
            position = (image_width - margin, position[1])

    # If unable to find position in top/bottom, try larger range of top/bottom positions
    # Create a gradient for top/bottom movement
    vertical_steps = [(0, step), (0, -step)]  # Move down  # Move up

    position = initial_pos
    for _ in range(max_tries):
        # Randomly choose to move up or down
        v_step = vertical_steps[np.random.randint(0, 2)]
        position = (position[0], position[1] + v_step[1])

        # Keep within certain distance from top/bottom boundaries
        if position[1] < margin:
            position = (position[0], margin)
        elif position[1] > image_height - margin:
            position = (position[0], image_height - margin)

        if not is_position_overlapping(position, used_positions):
            return position, 1

    # If all above methods fail, finally consider left/right sides
    if region == "left":
        side_pos = (margin, y)
    elif region == "right":
        side_pos = (image_width - margin, y)
    else:
        # If really can't find position, return initial position
        return initial_pos, 0

    # Try to avoid other labels on left/right sides
    position = side_pos
    for _ in range(max_tries):
        if not is_position_overlapping(position, used_positions):
            return position, 0

        # Move vertically
        move_direction = 1 if np.random.random() > 0.5 else -1
        position = (position[0], position[1] + step * move_direction)

        # Ensure not going out of image bounds
        if position[1] < margin:
            position = (position[0], margin)
        elif position[1] > image_height - margin:
            position = (position[0], image_height - margin)

    # If all attempts fail, return initial position
    return initial_pos, 0


def is_position_overlapping(position, used_positions, threshold=40):
    """Check if position overlaps with used positions"""

    for used_pos in used_positions:
        distance = np.sqrt(
            (position[0] - used_pos[0]) ** 2 + (position[1] - used_pos[1]) ** 2
        )
        if distance < threshold:
            return True
    return False


def draw_line_on_image(img, pt1, pt2, color, thickness=1):
    """Draw line segment on image"""
    # Choose appropriate method based on image library used
    # If using PIL
    draw = ImageDraw.Draw(img)
    draw.line([pt1, pt2], fill=color, width=thickness)

    # If using OpenCV
    # img_array = np.array(img)
    #  cv2.line(img_array, pt1, pt2, color, thickness)

    return img


def draw_circle_on_image(img, center, radius, color, thickness=-1):
    """Draw circle on image"""
    # Choose appropriate method based on image library used
    # If using PIL
    draw = ImageDraw.Draw(img)
    top_left = (center[0] - radius, center[1] - radius)
    bottom_right = (center[0] + radius, center[1] + radius)
    draw.ellipse(
        [top_left, bottom_right],
        fill=color if thickness == -1 else None,
        outline=color if thickness != -1 else None,
    )

    # If using OpenCV
    # img_array = np.array(img)
    # cv2.circle(img, center, radius, color, thickness)

    return img


def auto_render_image_refactored(
    scene,
    name="some_item",
    transparent_item_list=[],
    pose=None,
    fovy=np.deg2rad(75),
    number_font_size=config.number_font_size,
    width=config.width,
    height=config.height,
    might_mark_object_cuboid_list=[],
    might_mark_freespace_list=[],
    rectangle_grey=False,
    save_path="image.png",
    trans_visiblity=config.trans_visiblity,
):
    """
    Auto render image for a scene with specified parameters.

    Args:
        scene: The scene to render.
        name: The name of the item to render. Not necessarily used.
        transparent_item_list: List of items to be rendered with transparency.
        pose: The pose of the camera.
        fovy: The field of view in y-axis in radians.
        width: The width of the rendered image.
        height: The height of the rendered image.
        might_mark_object_cuboid_list: List of cuboids to be marked in the image. Currently we only draw a bounding box around the cuboid.
        might_mark_freespace_list: List of free spaces to be marked in the image.
        rectangle_grey: Whether to draw the rectangle in grey color.
        save_path: The path to save the rendered image.
        trans_visiblity: The visibility of the transparent items in the scene, 0 is invisible, 1 is fully visible.

    Returns:
        img: The rendered image.
        Note that the image must be convert to RGB format before saving, or the transparency will have bug.

    """

    for entity in scene.entities:
        if entity.name not in transparent_item_list:
            continue
        for component in entity.get_components():
            if isinstance(component, sapien.pysapien.render.RenderBodyComponent):
                component.visibility = trans_visiblity
                scene.step()
                scene.update_render()

    camera = create_and_mount_camera(
        scene,
        pose=pose,
        near=0.1,
        far=1000,
        width=width,
        height=height,
        fovy=fovy,
        camera_name="camera",
    )
    img = render_image(scene, camera)

    circled_numbers = [
        "①",
        "②",
        "③",
        "④",
        "⑤",
        "⑥",
        "⑦",
        "⑧",
        "⑨",
        "⑩",
        "⑪",
    ]

    colors = get_high_contrast_colors()
    number_idx = 0
    font = ImageFont.truetype("msmincho.ttc", number_font_size, encoding="unic")
    img = add_guide_line_labels(
        img,
        might_mark_object_cuboid_list,
        camera,
        width,
        height,
        colors,
        rectangle_grey,
    )

    for might_mark_freespace in might_mark_freespace_list:
        freespace_image_points = coordinate_convertor.world_rectangle_to_image_polygon(
            might_mark_freespace,
            camera.get_global_pose().p,
            transforms3d.quaternions.quat2mat(camera.get_global_pose().q),
            camera.fovx,
            width,
            height,
        )

        color = colors[number_idx % len(colors)]
        if freespace_image_points is None:
            continue
        img = draw_freespace_on_image(
            img, freespace_image_points, color, width, height, draw=True
        )

        freespace_image_points = [
            (
                max(0, min(width - 1, int(point[0]))),
                max(0, min(height - 1, int(point[1]))),
            )
            for point in freespace_image_points
        ]

        mid_freespace = np.mean(freespace_image_points, axis=0)
        bold_pixel = [
            (-number_font_size // 2, -number_font_size // 2),
            (-number_font_size // 2 + 1, -number_font_size // 2),
        ]

        draw = ImageDraw.Draw(img)
        for bold_offset in bold_pixel:
            draw.text(
                tuple(mid_freespace + np.array(bold_offset)),
                circled_numbers[number_idx % len(circled_numbers)],
                font=font,
                fill="black",
            )
        number_idx += 1

    img = img.convert("RGB")

    img.save(save_path)
    camera.disable()

    for entity in scene.entities:
        if entity.name not in transparent_item_list:
            continue
        for component in entity.get_components():
            if isinstance(component, sapien.pysapien.render.RenderBodyComponent):
                component.visibility = 1.0

    return img


def main():
    pass


if __name__ == "__main__":
    main()

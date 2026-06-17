#!/usr/bin/env python3
"""Eye-in-hand calibration from static robot poses and a 6x6 tag36h11 AprilGrid."""

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]

HAND_EYE_METHODS = {
    "tsai": cv2.CALIB_HAND_EYE_TSAI,
    "park": cv2.CALIB_HAND_EYE_PARK,
    "horaud": cv2.CALIB_HAND_EYE_HORAUD,
    "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
    "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def quaternion_xyzw_to_rotation(q):
    x, y, z, w = [float(v) for v in q]
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        raise ValueError("quaternion norm is too small")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def axis_rotation(axis, angle):
    c = math.cos(angle)
    s = math.sin(angle)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
    if axis == "z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    raise ValueError(f"unsupported rotation axis {axis}")


def euler_to_rotation(angles, order):
    rotation = np.eye(3, dtype=np.float64)
    for axis, angle in zip(order, angles):
        rotation = rotation @ axis_rotation(axis, float(angle))
    return rotation


def rotation_vector_to_rotation(vector):
    rotation, _ = cv2.Rodrigues(np.asarray(vector, dtype=np.float64).reshape(3, 1))
    return rotation


def invert_transform(rotation, translation):
    inv_rotation = rotation.T
    inv_translation = -inv_rotation @ translation.reshape(3, 1)
    return inv_rotation, inv_translation.reshape(3)


def transform_to_matrix(rotation, translation):
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation.reshape(3)
    return matrix


def matrix_to_rt(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    return matrix[:3, :3], matrix[:3, 3]


def rotation_angle_deg(rotation):
    rvec, _ = cv2.Rodrigues(rotation)
    return math.degrees(float(np.linalg.norm(rvec)))


def rotation_to_quaternion_xyzw(rotation):
    rvec, _ = cv2.Rodrigues(rotation)
    theta = float(np.linalg.norm(rvec))
    if theta < 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    axis = rvec.reshape(3) / theta
    return (axis * math.sin(theta / 2.0)).tolist() + [math.cos(theta / 2.0)]


def first_image_size(image_dir):
    for path in sorted(Path(image_dir).iterdir()):
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is not None:
                height, width = image.shape[:2]
                return [width, height]
    return None


def read_camera_file(camera_path):
    with open(camera_path, "r", encoding="utf-8") as file:
        if str(camera_path).lower().endswith(".json"):
            return json.load(file)
        return yaml.safe_load(file)


def camera_from_entry(data, camera_name):
    camera = data[camera_name]
    intrinsics = camera["intrinsics"]
    if isinstance(intrinsics, dict):
        key = sorted(intrinsics)[0]
        item = intrinsics[key]
        fx, fy, cx, cy = [float(item[name]) for name in ["fx", "fy", "ppx", "ppy"]]
        resolution = [int(v) for v in key.split("x")]
    else:
        fx, fy, cx, cy = [float(v) for v in intrinsics]
        resolution = camera.get("resolution")
        if resolution is None and camera.get("shape"):
            shape = camera["shape"]
            resolution = [int(shape[1]), int(shape[0])]
    return camera, fx, fy, cx, cy, resolution


def load_camera(camera_path, camera_name, image_size=None, crop_y=0.0):
    data = read_camera_file(camera_path)
    camera, fx, fy, cx, cy, resolution = camera_from_entry(data, camera_name)
    if crop_y:
        cy -= float(crop_y)
        print(f"[INFO] applying vertical crop offset: cy = cy - {float(crop_y):.3f}")

    scale = [1.0, 1.0]
    if image_size and resolution and list(image_size) != list(resolution):
        expected_after_crop = [resolution[0], int(round(resolution[1] - 2.0 * float(crop_y)))]
        if crop_y and list(image_size) == expected_after_crop:
            print(f"[INFO] image size {image_size} matches cropped source resolution {expected_after_crop}")
        else:
            scale = [image_size[0] / resolution[0], image_size[1] / resolution[1]]
            print(
                "[WARN] image size {} differs from camera resolution {}; scaling intrinsics by "
                "sx={:.6f}, sy={:.6f}".format(image_size, resolution, scale[0], scale[1])
            )
            fx, cx = fx * scale[0], cx * scale[0]
            fy, cy = fy * scale[1], cy * scale[1]
    camera_matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    return {
        "name": camera_name,
        "camera_model": camera.get("camera_model"),
        "distortion_model": camera.get("distortion_model", "none"),
        "camera_matrix": camera_matrix,
        "dist_coeffs": np.array(camera.get("distortion_coeffs", []), dtype=np.float64).reshape(-1, 1),
        "yaml_resolution": resolution,
        "image_size": image_size,
        "crop_y": float(crop_y),
        "intrinsics_scale": scale,
    }


def mean_raw_pose_rotation(summary_json, entry, source):
    pose_file = Path(summary_json).parent / entry["file"]
    with open(pose_file, "r", encoding="utf-8") as file:
        data = json.load(file)
    raw_vectors = np.array([sample["raw_pose"][3:6] for sample in data.get("samples", [])], dtype=np.float64)
    if raw_vectors.size == 0:
        raw_vectors = np.array([data["samples"][0]["raw_pose"][3:6]], dtype=np.float64)
    raw_mean = np.mean(raw_vectors, axis=0)
    if source == "raw_rotvec":
        return rotation_vector_to_rotation(raw_mean)
    if source == "raw_euler_xyz":
        return euler_to_rotation(raw_mean, "xyz")
    if source == "raw_euler_zyx":
        return euler_to_rotation(raw_mean, "zyx")
    if source == "raw_rpy":
        return axis_rotation("z", raw_mean[2]) @ axis_rotation("y", raw_mean[1]) @ axis_rotation("x", raw_mean[0])
    raise ValueError(f"unsupported robot orientation source {source}")


def load_robot_poses(summary_json, pose_direction, orientation_source="quaternion"):
    with open(summary_json, "r", encoding="utf-8") as file:
        data = json.load(file)
    poses = {}
    for item in data.get("captures", []):
        if orientation_source == "quaternion":
            rotation = quaternion_xyzw_to_rotation(item["quaternion_xyzw"])
        else:
            rotation = mean_raw_pose_rotation(summary_json, item, orientation_source)
        translation = np.array(item["position_m"], dtype=np.float64)
        if pose_direction == "gripper_to_base":
            rotation, translation = invert_transform(rotation, translation)
        poses[int(item["sample_id"])] = {
            "rotation_gripper_to_base": rotation,
            "translation_gripper_to_base": translation,
        }
    return poses


def list_images(image_dir):
    images = {}
    for path in sorted(Path(image_dir).iterdir()):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            continue
        try:
            images[int(path.stem)] = path
        except ValueError:
            pass
    return images


def aprilgrid_points(tag_rows, tag_cols, tag_size, tag_spacing):
    gap = tag_size * tag_spacing
    step = tag_size + gap
    points = {}
    for row in range(tag_rows):
        for col in range(tag_cols):
            tag_id = row * tag_cols + col
            x = col * step
            y = row * step
            points[tag_id] = np.array(
                [[x, y, 0], [x + tag_size, y, 0], [x + tag_size, y + tag_size, 0], [x, y + tag_size, 0]],
                dtype=np.float64,
            )
    return points


def preprocess_for_detection(image, mode, scale):
    proc = image
    if mode == "equalize":
        proc = cv2.equalizeHist(proc)
    elif mode == "clahe":
        proc = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(proc)
    elif mode == "gamma":
        lut = np.array(
            [min(255, ((idx / 255.0) ** 0.55) * 255) for idx in range(256)],
            dtype=np.uint8,
        )
        proc = cv2.LUT(proc, lut)
    elif mode == "unsharp":
        blur = cv2.GaussianBlur(proc, (0, 0), 1.2)
        proc = cv2.addWeighted(proc, 1.8, blur, -0.8, 0)
    elif mode == "eq_unsharp":
        proc = cv2.equalizeHist(proc)
        blur = cv2.GaussianBlur(proc, (0, 0), 1.2)
        proc = cv2.addWeighted(proc, 1.8, blur, -0.8, 0)
    elif mode == "clahe_unsharp":
        proc = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(proc)
        blur = cv2.GaussianBlur(proc, (0, 0), 1.2)
        proc = cv2.addWeighted(proc, 1.8, blur, -0.8, 0)
    if scale != 1.0:
        proc = cv2.resize(proc, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return proc


def detection_profiles(preprocess, detect_scale):
    if preprocess != "best":
        return [(preprocess, detect_scale)]
    return [
        ("equalize", 4.0),
        ("eq_unsharp", 4.0),
    ]


def detect_tag36h11_pupil(image, timeout_sec=10.0):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as temp:
        cv2.imwrite(temp.name, image)
        code = (
            "import cv2,json,sys,numpy as np\n"
            "from pupil_apriltags import Detector\n"
            "img=cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)\n"
            "det=Detector(families='tag36h11', nthreads=1, quad_decimate=1.0, "
            "quad_sigma=0.0, refine_edges=1, decode_sharpening=0.25)\n"
            "best={}\n"
            "for d in det.detect(img):\n"
            "    if d.tag_id not in best or d.decision_margin > best[d.tag_id].decision_margin:\n"
            "        best[d.tag_id]=d\n"
            "print(json.dumps([{'id':int(k),'corners':v.corners.tolist(),"
            "'decision_margin':float(v.decision_margin)} for k,v in best.items()]))\n"
        )
        try:
            proc = subprocess.run(
                [sys.executable, "-c", code, temp.name],
                text=True,
                capture_output=True,
                timeout=float(timeout_sec),
            )
        except subprocess.TimeoutExpired:
            return []
    if proc.returncode != 0:
        return []
    try:
        raw = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return []
    return raw


def detect_tag36h11_pupil_batch(images, timeout_sec=15.0):
    temp_files = []
    try:
        for image in images:
            temp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            temp.close()
            cv2.imwrite(temp.name, image)
            temp_files.append(temp.name)
        code = (
            "import cv2,json,sys\n"
            "from pupil_apriltags import Detector\n"
            "det=Detector(families='tag36h11', nthreads=1, quad_decimate=1.0, "
            "quad_sigma=0.0, refine_edges=1, decode_sharpening=0.25)\n"
            "all_results=[]\n"
            "for path in sys.argv[1:]:\n"
            "    img=cv2.imread(path, cv2.IMREAD_GRAYSCALE)\n"
            "    best={}\n"
            "    if img is not None:\n"
            "        for d in det.detect(img):\n"
            "            if d.tag_id not in best or d.decision_margin > best[d.tag_id].decision_margin:\n"
            "                best[d.tag_id]=d\n"
            "    all_results.append([{'id':int(k),'corners':v.corners.tolist(),"
            "'decision_margin':float(v.decision_margin)} for k,v in best.items()])\n"
            "print(json.dumps(all_results))\n"
        )
        try:
            proc = subprocess.run(
                [sys.executable, "-c", code, *temp_files],
                text=True,
                capture_output=True,
                timeout=float(timeout_sec),
            )
        except subprocess.TimeoutExpired:
            return [[] for _ in images]
        if proc.returncode != 0:
            return [[] for _ in images]
        try:
            return json.loads(proc.stdout)
        except json.JSONDecodeError:
            return [[] for _ in images]
    finally:
        for path in temp_files:
            try:
                Path(path).unlink()
            except OSError:
                pass


def detect_tag36h11_opencv(image):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    params = cv2.aruco.DetectorParameters()
    if hasattr(cv2.aruco, "CORNER_REFINE_APRILTAG"):
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    detector = cv2.aruco.ArucoDetector(dictionary, params)
    corners, ids, _ = detector.detectMarkers(image)
    if ids is None:
        return []
    return [{"id": int(tag_id[0]), "corners": corner.reshape(4, 2).tolist()} for corner, tag_id in zip(corners, ids)]


def distort_undistorted_points(points, source_camera_matrix, source_dist_coeffs, undistorted_camera_matrix):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
    normalized = (np.linalg.inv(undistorted_camera_matrix) @ homogeneous.T).T[:, :2]
    distorted = cv2.fisheye.distortPoints(
        normalized.reshape(-1, 1, 2),
        source_camera_matrix,
        source_dist_coeffs.reshape(4, 1),
    )
    return distorted.reshape(-1, 2)


def build_detection_variants(image, camera, preprocess, detect_scale, undistort_balances):
    variants = []

    def add_variant(name, variant_image, mapper_factory):
        for profile_preprocess, profile_scale in detection_profiles(preprocess, detect_scale):
            proc = preprocess_for_detection(variant_image, profile_preprocess, profile_scale)
            variants.append(
                {
                    "name": f"{name}:{profile_preprocess}@{profile_scale:g}",
                    "image": proc,
                    "scale": profile_scale,
                    "mapper": mapper_factory(profile_scale),
                }
            )

    add_variant(
        "raw",
        image,
        lambda scale: lambda corners: np.asarray(corners, dtype=np.float64).reshape(4, 2) / scale,
    )

    if camera["distortion_model"] == "equidistant" and undistort_balances:
        height, width = image.shape[:2]
        for balance in undistort_balances:
            new_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                camera["camera_matrix"],
                camera["dist_coeffs"].reshape(4, 1),
                (width, height),
                np.eye(3),
                balance=float(balance),
                new_size=(width, height),
            )
            undistorted = cv2.fisheye.undistortImage(
                image,
                camera["camera_matrix"],
                D=camera["dist_coeffs"].reshape(4, 1),
                Knew=new_camera_matrix,
                new_size=(width, height),
            )

            def mapper_factory(new_k):
                def mapper(scale):
                    return lambda corners: distort_undistorted_points(
                        np.asarray(corners, dtype=np.float64).reshape(4, 2) / scale,
                        camera["camera_matrix"],
                        camera["dist_coeffs"],
                        new_k,
                    )
                return mapper

            add_variant(f"undistort_balance_{float(balance):.2f}", undistorted, mapper_factory(new_camera_matrix))
    return variants


def detect_markers_for_variants(variants, backend, timeout_sec):
    if backend in {"auto", "pupil"}:
        pupil_results = detect_tag36h11_pupil_batch(
            [variant["image"] for variant in variants],
            timeout_sec=timeout_sec,
        )
    else:
        pupil_results = [[] for _ in variants]

    results = []
    for variant, pupil_markers in zip(variants, pupil_results):
        if backend == "opencv":
            raw_markers = detect_tag36h11_opencv(variant["image"])
        elif backend == "pupil":
            raw_markers = pupil_markers
        else:
            raw_markers = pupil_markers or detect_tag36h11_opencv(variant["image"])

        markers = []
        for marker in raw_markers:
            markers.append(
                {
                    "id": int(marker["id"]),
                    "corners": variant["mapper"](marker["corners"]),
                    "decision_margin": float(marker.get("decision_margin", 0.0)),
                }
            )
        results.append({"name": variant["name"], "markers": markers})
    return results


def detect_markers(image, backend, preprocess, detect_scale, camera, undistort_balances, timeout_sec):
    variants = build_detection_variants(image, camera, preprocess, detect_scale, undistort_balances)
    variant_results = detect_markers_for_variants(variants, backend, timeout_sec)
    all_markers = []
    for result in variant_results:
        for marker in result["markers"]:
            marker = dict(marker)
            marker["strategy"] = result["name"]
            all_markers.append(marker)
    return all_markers, variant_results


def valid_marker_points(markers, grid_points, args):
    best_by_id = {}
    for marker in markers:
        tag_id = int(marker["id"])
        if tag_id < args.start_id or tag_id > args.end_id or tag_id not in grid_points:
            continue
        previous = best_by_id.get(tag_id)
        if previous is None or marker.get("decision_margin", 0.0) > previous.get("decision_margin", 0.0):
            best_by_id[tag_id] = marker

    object_points = []
    image_points = []
    tag_ids = []
    strategies = []
    for tag_id, marker in sorted(best_by_id.items()):
        object_points.extend(grid_points[tag_id])
        image_points.extend(np.asarray(marker["corners"], dtype=np.float64).reshape(4, 2))
        tag_ids.append(tag_id)
        strategies.append(marker.get("strategy", "unknown"))
    return object_points, image_points, tag_ids, strategies


def parse_undistort_balances(value):
    if not value:
        return []
    balances = []
    for item in str(value).split(","):
        item = item.strip()
        if item:
            balances.append(float(item))
    return balances


def choose_best_board_pose(image, grid_points, camera, args):
    markers, variant_results = detect_markers(
        image,
        args.detector_backend,
        args.preprocess,
        args.detect_scale,
        camera,
        parse_undistort_balances(args.undistort_balances),
        args.detector_timeout_sec,
    )

    candidates = []
    candidate_results = list(variant_results)
    if args.allow_union:
        candidate_results.append({"name": "union", "markers": markers})

    for result in candidate_results:
        object_points, image_points, tag_ids, strategies = valid_marker_points(result["markers"], grid_points, args)
        if len(tag_ids) < args.min_tags:
            continue
        try:
            rotation, translation, errors = solve_target_pose(object_points, image_points, camera)
        except Exception:
            continue
        candidates.append(
            {
                "strategy": result["name"],
                "rotation_target_to_camera": rotation,
                "translation_target_to_camera": translation,
                "tag_count": len(tag_ids),
                "corner_count": len(image_points),
                "tag_ids": sorted(tag_ids),
                "tag_strategies": strategies,
                "mean_reprojection_error_px": float(np.mean(errors)),
                "max_reprojection_error_px": float(np.max(errors)),
            }
        )

    if not candidates:
        counts = {
            result["name"]: len(valid_marker_points(result["markers"], grid_points, args)[2])
            for result in variant_results
        }
        best_count = max(counts.values(), default=0)
        return None, f"only_{best_count}_valid_tags", counts

    candidates.sort(
        key=lambda item: (
            item["tag_count"],
            -item["mean_reprojection_error_px"],
            -item["max_reprojection_error_px"],
        ),
        reverse=True,
    )
    return candidates[0], None, {
        result["name"]: len(valid_marker_points(result["markers"], grid_points, args)[2])
        for result in variant_results
    }


def detect_markers_legacy(image, backend, preprocess, detect_scale):
    proc = preprocess_for_detection(image, preprocess, detect_scale)
    if backend == "opencv":
        markers = detect_tag36h11_opencv(proc)
    elif backend == "pupil":
        markers = detect_tag36h11_pupil(proc)
    else:
        markers = detect_tag36h11_pupil(proc)
        if not markers:
            markers = detect_tag36h11_opencv(proc)
    for marker in markers:
        marker["corners"] = (np.asarray(marker["corners"], dtype=np.float64) / detect_scale).tolist()
    return markers


def project_points(object_points, rvec, tvec, camera):
    object_points = np.asarray(object_points, dtype=np.float64).reshape(-1, 1, 3)
    if camera["distortion_model"] == "equidistant":
        projected, _ = cv2.fisheye.projectPoints(
            object_points,
            rvec,
            tvec.reshape(3, 1),
            camera["camera_matrix"],
            camera["dist_coeffs"].reshape(4, 1),
        )
    else:
        projected, _ = cv2.projectPoints(
            object_points,
            rvec,
            tvec.reshape(3, 1),
            camera["camera_matrix"],
            camera["dist_coeffs"],
        )
    return projected.reshape(-1, 2)


def solve_target_pose(object_points, image_points, camera):
    object_points = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    image_points = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    if camera["distortion_model"] == "equidistant":
        normalized = cv2.fisheye.undistortPoints(
            image_points.reshape(-1, 1, 2),
            camera["camera_matrix"],
            camera["dist_coeffs"].reshape(4, 1),
        ).reshape(-1, 2)
        ok, rvec, tvec = cv2.solvePnP(object_points, normalized, np.eye(3), None, flags=cv2.SOLVEPNP_ITERATIVE)
    else:
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera["camera_matrix"],
            camera["dist_coeffs"],
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    if not ok:
        raise RuntimeError("solvePnP failed")
    rotation, _ = cv2.Rodrigues(rvec)
    projected = project_points(object_points, rvec, tvec, camera)
    errors = np.linalg.norm(projected - image_points, axis=1)
    return rotation, tvec.reshape(3), errors


def estimate_board_pose(image_path, grid_points, camera, args):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None, "image_read_failed"
    target, reason, strategy_counts = choose_best_board_pose(image, grid_points, camera, args)
    if target is None:
        return None, f"{reason}; strategies={strategy_counts}"
    target["strategy_counts"] = strategy_counts
    return target, None


def calibrate(samples, method):
    return cv2.calibrateHandEye(
        [s["robot"]["rotation_gripper_to_base"] for s in samples],
        [s["robot"]["translation_gripper_to_base"].reshape(3, 1) for s in samples],
        [s["target"]["rotation_target_to_camera"] for s in samples],
        [s["target"]["translation_target_to_camera"].reshape(3, 1) for s in samples],
        method=HAND_EYE_METHODS[method],
    )


def compute_board_consistency(samples, rotation_cam_to_gripper, translation_cam_to_gripper):
    t_gripper_camera = transform_to_matrix(rotation_cam_to_gripper, translation_cam_to_gripper)
    board_poses = []
    for sample in samples:
        robot = sample["robot"]
        target = sample["target"]
        t_base_gripper = transform_to_matrix(
            robot["rotation_gripper_to_base"],
            robot["translation_gripper_to_base"],
        )
        t_camera_grid = transform_to_matrix(
            target["rotation_target_to_camera"],
            target["translation_target_to_camera"],
        )
        board_poses.append(t_base_gripper @ t_gripper_camera @ t_camera_grid)

    translations = np.array([pose[:3, 3] for pose in board_poses], dtype=np.float64)
    mean_translation = np.mean(translations, axis=0)
    translation_errors = np.linalg.norm(translations - mean_translation, axis=1)

    mean_rvec = np.mean([cv2.Rodrigues(pose[:3, :3])[0].reshape(3) for pose in board_poses], axis=0)
    mean_rotation, _ = cv2.Rodrigues(mean_rvec.reshape(3, 1))
    rotation_errors = np.array(
        [rotation_angle_deg(mean_rotation.T @ pose[:3, :3]) for pose in board_poses],
        dtype=np.float64,
    )

    return {
        "mean_T_base_grid": transform_to_matrix(mean_rotation, mean_translation).tolist(),
        "translation_std_m_xyz": np.std(translations, axis=0).tolist(),
        "translation_error_mean_m": float(np.mean(translation_errors)),
        "translation_error_max_m": float(np.max(translation_errors)),
        "rotation_error_mean_deg": float(np.mean(rotation_errors)),
        "rotation_error_max_deg": float(np.max(rotation_errors)),
        "per_sample": [
            {
                "sample_id": sample["sample_id"],
                "translation_error_m": float(translation_errors[idx]),
                "rotation_error_deg": float(rotation_errors[idx]),
            }
            for idx, sample in enumerate(samples)
        ],
    }


def build_result(args, camera, samples, skipped, rotation_cam_to_gripper, translation_cam_to_gripper):
    rotation_gripper_to_cam, translation_gripper_to_cam = invert_transform(rotation_cam_to_gripper, translation_cam_to_gripper)
    reproj_mean = [s["target"]["mean_reprojection_error_px"] for s in samples]
    reproj_max = [s["target"]["max_reprojection_error_px"] for s in samples]
    tag_counts = [s["target"]["tag_count"] for s in samples]
    board_consistency = compute_board_consistency(samples, rotation_cam_to_gripper, translation_cam_to_gripper)
    return {
        "calibration_type": "eye_in_hand",
        "method": args.method,
        "target": {
            "target_type": "aprilgrid",
            "tag_family": "tag36h11",
            "tagRows": args.tag_rows,
            "tagCols": args.tag_cols,
            "tagSize": args.tag_size,
            "tagSpacing": args.tag_spacing,
            "physical_spacing_m": args.tag_size * args.tag_spacing,
            "start_id": args.start_id,
            "end_id": args.end_id,
        },
        "inputs": {
            "pose_summary": str(args.pose_summary),
            "image_dir": str(args.image_dir),
            "camera_file": str(args.camera_yaml),
            "camera_name": args.camera_name,
            "robot_pose_direction": args.robot_pose_direction,
            "robot_orientation_source": args.robot_orientation_source,
        },
        "camera": {
            "camera_model": camera["camera_model"],
            "distortion_model": camera["distortion_model"],
            "intrinsics": camera["camera_matrix"].tolist(),
            "distortion_coeffs": camera["dist_coeffs"].reshape(-1).tolist(),
            "yaml_resolution": camera["yaml_resolution"],
            "image_size": camera["image_size"],
            "crop_y": camera["crop_y"],
            "intrinsics_scale": camera["intrinsics_scale"],
        },
        "result": {
            "T_cam_to_gripper": transform_to_matrix(rotation_cam_to_gripper, translation_cam_to_gripper).tolist(),
            "T_gripper_to_cam": transform_to_matrix(rotation_gripper_to_cam, translation_gripper_to_cam).tolist(),
            "translation_cam_to_gripper_m": translation_cam_to_gripper.reshape(3).tolist(),
            "quaternion_cam_to_gripper_xyzw": rotation_to_quaternion_xyzw(rotation_cam_to_gripper),
        },
        "diagnostics": {
            "used_sample_count": len(samples),
            "skipped_sample_count": len(skipped),
            "tag_count_mean": float(np.mean(tag_counts)),
            "tag_count_min": int(np.min(tag_counts)),
            "tag_count_max": int(np.max(tag_counts)),
            "mean_reprojection_error_px": float(np.mean(reproj_mean)),
            "max_reprojection_error_px": float(np.max(reproj_max)),
            "board_consistency": board_consistency,
            "samples": [
                {
                    "sample_id": s["sample_id"],
                    "image": str(s["image"]),
                    "tag_count": s["target"]["tag_count"],
                    "tag_ids": s["target"]["tag_ids"],
                    "strategy": s["target"]["strategy"],
                    "strategy_counts": s["target"]["strategy_counts"],
                    "mean_reprojection_error_px": s["target"]["mean_reprojection_error_px"],
                    "max_reprojection_error_px": s["target"]["max_reprojection_error_px"],
                }
                for s in samples
            ],
            "skipped": skipped,
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-summary", type=Path, default=REPO_ROOT / "data/calibration/static_pose_samples_0615/static_pose_summary.json")
    parser.add_argument("--image-dir", type=Path, default=REPO_ROOT / "data/calibration/image_0615")
    parser.add_argument(
        "--camera-yaml",
        type=Path,
        default=REPO_ROOT / "data/calibration/camera_cam0_640x400_intrinsics.yaml",
        help="Kalibr camchain yaml or device calibration.json",
    )
    parser.add_argument("--camera-name", default="cam0")
    parser.add_argument("--crop-y", type=float, default=0.0, help="pixels cropped from the top before saving images")
    parser.add_argument("--tag-rows", type=int, default=6)
    parser.add_argument("--tag-cols", type=int, default=6)
    parser.add_argument("--tag-size", type=float, default=0.055, help="meters; board says Size=5.5cm")
    parser.add_argument("--tag-spacing", type=float, default=0.3, help="spacing/tag_size; 1.65cm / 5.5cm = 0.3")
    parser.add_argument("--start-id", type=int, default=0)
    parser.add_argument("--end-id", type=int, default=35)
    parser.add_argument("--min-tags", type=int, default=6)
    parser.add_argument("--detector-backend", choices=["auto", "pupil", "opencv"], default="pupil")
    parser.add_argument(
        "--preprocess",
        choices=["best", "none", "equalize", "clahe", "gamma", "unsharp", "eq_unsharp", "clahe_unsharp"],
        default="best",
        help="best tries multiple enhancement profiles and keeps the strongest tag set",
    )
    parser.add_argument("--detect-scale", type=float, default=4.0)
    parser.add_argument("--detector-timeout-sec", type=float, default=20.0)
    parser.add_argument(
        "--allow-union",
        action="store_true",
        help="also try a union of detections from all preprocessing strategies; higher recall but easier to admit false detections",
    )
    parser.add_argument(
        "--undistort-balances",
        default="0.0",
        help="comma-separated fisheye undistort balances to try for tag detection; empty disables",
    )
    parser.add_argument("--robot-pose-direction", choices=["base_to_gripper", "gripper_to_base"], default="base_to_gripper")
    parser.add_argument(
        "--robot-orientation-source",
        choices=["quaternion", "raw_rotvec", "raw_euler_xyz", "raw_euler_zyx", "raw_rpy"],
        default="raw_rpy",
        help="quaternion uses summary values; raw_* recomputes orientation from static_pose_XXX.json raw_pose[3:6]",
    )
    parser.add_argument("--method", choices=sorted(HAND_EYE_METHODS), default="tsai")
    parser.add_argument("--max-mean-reprojection-error", type=float, default=5.0)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "data/calibration/handeye_0615/handeye_result.yaml")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    images = list_images(args.image_dir)
    camera = load_camera(args.camera_yaml, args.camera_name, first_image_size(args.image_dir), args.crop_y)
    robot_poses = load_robot_poses(args.pose_summary, args.robot_pose_direction, args.robot_orientation_source)
    grid_points = aprilgrid_points(args.tag_rows, args.tag_cols, args.tag_size, args.tag_spacing)

    samples = []
    skipped = []
    for sample_id in sorted(set(robot_poses) & set(images)):
        try:
            target, reason = estimate_board_pose(images[sample_id], grid_points, camera, args)
        except Exception as exc:
            target, reason = None, f"board_pose_failed:{exc}"
        if target is None:
            skipped.append({"sample_id": sample_id, "image": str(images[sample_id]), "reason": reason})
            continue
        samples.append({"sample_id": sample_id, "image": images[sample_id], "robot": robot_poses[sample_id], "target": target})
        print(
            "[OK] sample {:03d}: tags={}, strategy={}, ids={}, mean reproj={:.3f}px".format(
                sample_id,
                target["tag_count"],
                target["strategy"],
                target["tag_ids"],
                target["mean_reprojection_error_px"],
            )
        )

    for sample_id in sorted(set(robot_poses) - set(images)):
        skipped.append({"sample_id": sample_id, "reason": "missing_image"})

    print(f"[INFO] usable samples: {len(samples)}, skipped: {len(skipped)}")
    if args.dry_run:
        return 0
    if len(samples) < 3:
        raise RuntimeError("at least 3 usable samples are required for hand-eye calibration")

    mean_reproj = float(np.mean([s["target"]["mean_reprojection_error_px"] for s in samples]))
    if mean_reproj > args.max_mean_reprojection_error:
        raise RuntimeError(
            "mean board reprojection error is {:.3f}px, above threshold {:.3f}px".format(
                mean_reproj, args.max_mean_reprojection_error
            )
        )

    rotation, translation = calibrate(samples, args.method)
    result = build_result(args, camera, samples, skipped, rotation, translation.reshape(3))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as file:
        yaml.safe_dump(result, file, allow_unicode=True, sort_keys=False)
    print(f"[RESULT] saved {args.output}")
    print("[RESULT] translation_cam_to_gripper_m =", result["result"]["translation_cam_to_gripper_m"])
    print("[RESULT] quaternion_cam_to_gripper_xyzw =", result["result"]["quaternion_cam_to_gripper_xyzw"])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)

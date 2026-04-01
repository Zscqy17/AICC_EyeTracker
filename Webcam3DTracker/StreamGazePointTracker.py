from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial.transform import Rotation as Rscipy


NOSE_INDICES = [
    4, 45, 275, 220, 440, 1, 5, 51, 281, 44, 274, 241,
    461, 125, 354, 218, 438, 195, 167, 393, 165, 391, 3, 248,
]
LEFT_IRIS_INDEX = 468
RIGHT_IRIS_INDEX = 473


@dataclass
class GazeTrackingResult:
    face_detected: bool
    eyes_calibrated: bool
    center_calibrated: bool
    screen_point: Optional[tuple[int, int]]
    normalized_point: Optional[tuple[float, float]]
    raw_yaw_deg: Optional[float]
    raw_pitch_deg: Optional[float]
    gaze_vector: Optional[np.ndarray]
    frame_size: tuple[int, int]
    message: str


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 1e-9 else vector


def _compute_scale(points_3d: np.ndarray) -> float:
    point_count = len(points_3d)
    total_distance = 0.0
    pair_count = 0
    for first_index in range(point_count):
        for second_index in range(first_index + 1, point_count):
            total_distance += np.linalg.norm(points_3d[first_index] - points_3d[second_index])
            pair_count += 1
    return total_distance / pair_count if pair_count > 0 else 1.0


class StreamGazePointTracker:
    """Track gaze points from grayscale video frames.

    The tracker expects a single-channel frame stream, typically 1280x720 at high frame rate.
    It does not open cameras or windows. The caller pushes frames via process_frame and receives
    the current gaze point after eye calibration and center calibration are completed.
    """

    def __init__(
        self,
        screen_size: tuple[int, int],
        expected_frame_size: tuple[int, int] = (720, 1280),
        filter_length: int = 10,
        yaw_degrees: float = 15.0,
        pitch_degrees: float = 5.0,
        base_eye_radius: float = 20.0,
        enforce_frame_size: bool = True,
    ) -> None:
        self.screen_width, self.screen_height = screen_size
        self.expected_frame_size = expected_frame_size
        self.filter_length = filter_length
        self.yaw_degrees = yaw_degrees
        self.pitch_degrees = pitch_degrees
        self.base_eye_radius = base_eye_radius
        self.enforce_frame_size = enforce_frame_size

        try:
            self.mp_face_mesh = mp.solutions.face_mesh
        except AttributeError as exc:
            raise RuntimeError(
                "MediaPipe Solutions API is unavailable. Install the pinned dependency set from requirements.txt."
            ) from exc

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.r_ref_nose = [None]
        self.combined_gaze_directions: deque[np.ndarray] = deque(maxlen=filter_length)

        self.left_sphere_locked = False
        self.left_sphere_local_offset: Optional[np.ndarray] = None
        self.left_calibration_nose_scale: Optional[float] = None

        self.right_sphere_locked = False
        self.right_sphere_local_offset: Optional[np.ndarray] = None
        self.right_calibration_nose_scale: Optional[float] = None

        self.calibration_offset_yaw = 0.0
        self.calibration_offset_pitch = 0.0
        self.center_calibrated = False

    def close(self) -> None:
        self.face_mesh.close()

    def reset_calibration(self) -> None:
        self.r_ref_nose = [None]
        self.combined_gaze_directions.clear()
        self.left_sphere_locked = False
        self.left_sphere_local_offset = None
        self.left_calibration_nose_scale = None
        self.right_sphere_locked = False
        self.right_sphere_local_offset = None
        self.right_calibration_nose_scale = None
        self.calibration_offset_yaw = 0.0
        self.calibration_offset_pitch = 0.0
        self.center_calibrated = False

    def process_frame(
        self,
        frame_gray: np.ndarray,
        calibrate_eyes: bool = False,
        calibrate_center: bool = False,
    ) -> GazeTrackingResult:
        gray = self._validate_frame(frame_gray)
        frame_height, frame_width = gray.shape

        frame_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        results = self.face_mesh.process(frame_rgb)
        if not results.multi_face_landmarks:
            return GazeTrackingResult(
                face_detected=False,
                eyes_calibrated=self._eyes_calibrated,
                center_calibrated=self.center_calibrated,
                screen_point=None,
                normalized_point=None,
                raw_yaw_deg=None,
                raw_pitch_deg=None,
                gaze_vector=None,
                frame_size=(frame_width, frame_height),
                message="No face detected.",
            )

        face_landmarks = results.multi_face_landmarks[0].landmark
        head_center, rotation_matrix, nose_points_3d = self._compute_head_pose(face_landmarks, frame_width, frame_height)

        left_iris = face_landmarks[LEFT_IRIS_INDEX]
        right_iris = face_landmarks[RIGHT_IRIS_INDEX]
        iris_3d_left = np.array([left_iris.x * frame_width, left_iris.y * frame_height, left_iris.z * frame_width], dtype=float)
        iris_3d_right = np.array([right_iris.x * frame_width, right_iris.y * frame_height, right_iris.z * frame_width], dtype=float)

        if calibrate_eyes:
            self._calibrate_eyes(head_center, rotation_matrix, nose_points_3d, iris_3d_left, iris_3d_right)

        if not self._eyes_calibrated:
            return GazeTrackingResult(
                face_detected=True,
                eyes_calibrated=False,
                center_calibrated=self.center_calibrated,
                screen_point=None,
                normalized_point=None,
                raw_yaw_deg=None,
                raw_pitch_deg=None,
                gaze_vector=None,
                frame_size=(frame_width, frame_height),
                message="Face detected. Call process_frame(..., calibrate_eyes=True) once while the user looks at screen center.",
            )

        sphere_world_l, sphere_world_r = self._compute_eye_spheres(head_center, rotation_matrix, nose_points_3d)
        left_gaze_dir = _normalize(iris_3d_left - sphere_world_l)
        right_gaze_dir = _normalize(iris_3d_right - sphere_world_r)
        raw_combined_direction = _normalize((left_gaze_dir + right_gaze_dir) / 2.0)

        self.combined_gaze_directions.append(raw_combined_direction)
        combined_direction = _normalize(np.mean(self.combined_gaze_directions, axis=0))

        raw_yaw_deg, raw_pitch_deg = self._raw_gaze_angles(combined_direction)
        if calibrate_center:
            self.calibration_offset_yaw = -raw_yaw_deg
            self.calibration_offset_pitch = -raw_pitch_deg
            self.center_calibrated = True

        screen_x, screen_y = self._screen_coordinates(raw_yaw_deg, raw_pitch_deg)
        normalized_x = screen_x / max(self.screen_width, 1)
        normalized_y = screen_y / max(self.screen_height, 1)

        message = "Tracking."
        if not self.center_calibrated:
            message = "Eyes calibrated. Call process_frame(..., calibrate_center=True) once while the user looks at screen center."

        return GazeTrackingResult(
            face_detected=True,
            eyes_calibrated=True,
            center_calibrated=self.center_calibrated,
            screen_point=(screen_x, screen_y),
            normalized_point=(normalized_x, normalized_y),
            raw_yaw_deg=raw_yaw_deg,
            raw_pitch_deg=raw_pitch_deg,
            gaze_vector=combined_direction,
            frame_size=(frame_width, frame_height),
            message=message,
        )

    @property
    def _eyes_calibrated(self) -> bool:
        return self.left_sphere_locked and self.right_sphere_locked

    def _validate_frame(self, frame_gray: np.ndarray) -> np.ndarray:
        if frame_gray is None:
            raise ValueError("frame_gray is required.")

        frame = np.asarray(frame_gray)
        if frame.ndim == 3 and frame.shape[2] == 1:
            frame = frame[:, :, 0]

        if frame.ndim != 2:
            raise ValueError("Expected a single-channel frame with shape (H, W) or (H, W, 1).")

        if self.enforce_frame_size and frame.shape != self.expected_frame_size:
            raise ValueError(
                f"Expected frame size {self.expected_frame_size}, got {tuple(frame.shape)}."
            )

        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        return frame

    def _compute_head_pose(
        self,
        face_landmarks,
        frame_width: int,
        frame_height: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        points_3d = np.array(
            [
                [
                    face_landmarks[index].x * frame_width,
                    face_landmarks[index].y * frame_height,
                    face_landmarks[index].z * frame_width,
                ]
                for index in NOSE_INDICES
            ],
            dtype=float,
        )

        center = np.mean(points_3d, axis=0)
        centered = points_3d - center
        covariance = np.cov(centered.T)
        eigen_values, eigen_vectors = np.linalg.eigh(covariance)
        eigen_vectors = eigen_vectors[:, np.argsort(-eigen_values)]

        if np.linalg.det(eigen_vectors) < 0:
            eigen_vectors[:, 2] *= -1

        rotation = Rscipy.from_matrix(eigen_vectors)
        roll, pitch, yaw = rotation.as_euler("zyx", degrees=False)
        rotation_matrix = Rscipy.from_euler("zyx", [roll, pitch, yaw]).as_matrix()

        if self.r_ref_nose[0] is None:
            self.r_ref_nose[0] = rotation_matrix.copy()
        else:
            reference = self.r_ref_nose[0]
            for axis_index in range(3):
                if np.dot(rotation_matrix[:, axis_index], reference[:, axis_index]) < 0:
                    rotation_matrix[:, axis_index] *= -1

        return center, rotation_matrix, points_3d

    def _calibrate_eyes(
        self,
        head_center: np.ndarray,
        rotation_matrix: np.ndarray,
        nose_points_3d: np.ndarray,
        iris_3d_left: np.ndarray,
        iris_3d_right: np.ndarray,
    ) -> None:
        current_nose_scale = _compute_scale(nose_points_3d)
        camera_dir_world = np.array([0.0, 0.0, 1.0], dtype=float)
        camera_dir_local = rotation_matrix.T @ camera_dir_world

        self.left_sphere_local_offset = rotation_matrix.T @ (iris_3d_left - head_center)
        self.left_sphere_local_offset += self.base_eye_radius * camera_dir_local
        self.left_calibration_nose_scale = current_nose_scale
        self.left_sphere_locked = True

        self.right_sphere_local_offset = rotation_matrix.T @ (iris_3d_right - head_center)
        self.right_sphere_local_offset += self.base_eye_radius * camera_dir_local
        self.right_calibration_nose_scale = current_nose_scale
        self.right_sphere_locked = True

        self.combined_gaze_directions.clear()
        self.calibration_offset_yaw = 0.0
        self.calibration_offset_pitch = 0.0
        self.center_calibrated = False

    def _compute_eye_spheres(
        self,
        head_center: np.ndarray,
        rotation_matrix: np.ndarray,
        nose_points_3d: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        current_nose_scale = _compute_scale(nose_points_3d)

        left_scale = current_nose_scale / self.left_calibration_nose_scale if self.left_calibration_nose_scale else 1.0
        right_scale = current_nose_scale / self.right_calibration_nose_scale if self.right_calibration_nose_scale else 1.0

        sphere_world_l = head_center + rotation_matrix @ (self.left_sphere_local_offset * left_scale)
        sphere_world_r = head_center + rotation_matrix @ (self.right_sphere_local_offset * right_scale)
        return sphere_world_l, sphere_world_r

    def _raw_gaze_angles(self, combined_gaze_direction: np.ndarray) -> tuple[float, float]:
        reference_forward = np.array([0.0, 0.0, -1.0], dtype=float)
        avg_direction = _normalize(combined_gaze_direction)

        xz_projection = _normalize(np.array([avg_direction[0], 0.0, avg_direction[2]], dtype=float))
        yaw_rad = math.acos(np.clip(np.dot(reference_forward, xz_projection), -1.0, 1.0))
        if avg_direction[0] < 0:
            yaw_rad = -yaw_rad

        yz_projection = _normalize(np.array([0.0, avg_direction[1], avg_direction[2]], dtype=float))
        pitch_rad = math.acos(np.clip(np.dot(reference_forward, yz_projection), -1.0, 1.0))
        if avg_direction[1] > 0:
            pitch_rad = -pitch_rad

        yaw_deg = np.degrees(yaw_rad)
        pitch_deg = np.degrees(pitch_rad)

        if yaw_deg < 0:
            yaw_deg = -yaw_deg
        elif yaw_deg > 0:
            yaw_deg = -yaw_deg

        return float(yaw_deg), float(pitch_deg)

    def _screen_coordinates(self, raw_yaw_deg: float, raw_pitch_deg: float) -> tuple[int, int]:
        yaw_deg = raw_yaw_deg + self.calibration_offset_yaw
        pitch_deg = raw_pitch_deg + self.calibration_offset_pitch

        screen_x = int(((yaw_deg + self.yaw_degrees) / (2.0 * self.yaw_degrees)) * self.screen_width)
        screen_y = int(((self.pitch_degrees - pitch_deg) / (2.0 * self.pitch_degrees)) * self.screen_height)

        screen_x = max(0, min(screen_x, self.screen_width - 1))
        screen_y = max(0, min(screen_y, self.screen_height - 1))
        return screen_x, screen_y


if __name__ == "__main__":
    tracker = StreamGazePointTracker(screen_size=(1920, 1080))
    sample_frame = np.zeros((720, 1280), dtype=np.uint8)
    result = tracker.process_frame(sample_frame)
    print(result)
    tracker.close()
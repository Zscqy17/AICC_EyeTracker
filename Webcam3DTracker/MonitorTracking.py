import cv2
import numpy as np
import os
import mediapipe as mp
import sys
import time
import math
from scipy.spatial.transform import Rotation as Rscipy
from collections import deque
import threading

try:
    import pyautogui
except Exception:
    pyautogui = None

try:
    import keyboard
except Exception:
    keyboard = None

# Screen and mouse control setup (from old script)
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "0"))
REGRESSION_CALIBRATION_FILE = os.environ.get(
    "REGRESSION_CALIBRATION_FILE",
    os.path.join(PROJECT_DIR, "monitor_tracking_regression.npz"),
)
if pyautogui is not None:
    try:
        MONITOR_WIDTH, MONITOR_HEIGHT = pyautogui.size()
    except Exception:
        pyautogui = None
        MONITOR_WIDTH, MONITOR_HEIGHT = (1920, 1080)
else:
    MONITOR_WIDTH, MONITOR_HEIGHT = (1920, 1080)
CENTER_X = MONITOR_WIDTH // 2
CENTER_Y = MONITOR_HEIGHT // 2
mouse_control_enabled = False
last_mouse_toggle_time = 0.0
global_hotkey_warning_shown = False
filter_length = 10
gaze_length = 350
SCREEN_YAW_DEGREES = 15.0
SCREEN_PITCH_DEGREES = 5.0
CALIBRATION_WINDOW_NAME = "9-Point Calibration"
NINE_POINT_CAPTURE_FRAMES = 18
NINE_POINT_MAX_STD_DEG = 0.75

# --- Orbit camera state for the debug view ---
orbit_yaw   = -151.0          # radians, left/right
orbit_pitch = 00.0          # radians, up/down
orbit_radius = 1500.0       # distance from head center
orbit_fov_deg = 50.0       # horizontal FOV for projection

# --- Debug-view world freeze (pivot fixed after center calibration) ---
debug_world_frozen = False
orbit_pivot_frozen = None  # world-space point the debug camera orbits (monitor center at calib)

# Stored gaze markers on the monitor plane (as (a,b) in plane coords)
# a = 0..1 across width (p0->p1), b = 0..1 down height (p0->p3)
gaze_markers = []

# --- 3D monitor plane state (world space) ---
monitor_corners = None   # list of 4 world points (p0..p3)
monitor_center_w = None  # world center of the plane
monitor_normal_w = None  # world normal
units_per_cm = None      # world units per centimeter (computed at calibration)

# Shared mouse target position
mouse_target = [CENTER_X, CENTER_Y]
mouse_lock = threading.Lock()

# Calibration offsets for screen mapping
calibration_offset_yaw = 0
calibration_offset_pitch = 0

# --- Simple monitor-edge calibration state ---
# 0 = waiting for center, 1 = waiting for left edge, 2 = done
calib_step = 0

# Buffers to store recent gaze data for smoothing
combined_gaze_directions = deque(maxlen=filter_length)
recent_gaze_angles = deque(maxlen=NINE_POINT_CAPTURE_FRAMES)

# --- Interactive 9-point regression calibration state ---
nine_point_active = False
nine_point_targets = []
nine_point_samples = []
nine_point_index = 0
regression_coefficients_x = None
regression_coefficients_y = None
regression_fit_error_px = None
calibration_window_ready = False
nine_point_status_text = ""
regression_loaded_from_disk = False

# reference matrices to fix coordinate flipping issue
# These help keep the axes consistent from frame to frame by stabilizing eigenvector directions
R_ref_nose = [None]
R_ref_forehead = [None]
calibration_nose_scale = None

# Initialize MediaPipe FaceMesh
try:
    mp_face_mesh = mp.solutions.face_mesh
except AttributeError as exc:
    raise RuntimeError(
        "This project requires the MediaPipe Solutions API. Install the pinned dependency set from requirements.txt, which uses mediapipe==0.10.21."
    ) from exc

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def open_camera(camera_index):
    backend_candidates = []
    if sys.platform == "darwin":
        backend_candidates.append(cv2.CAP_AVFOUNDATION)
    elif sys.platform.startswith("win"):
        backend_candidates.append(cv2.CAP_DSHOW)

    backend_candidates.append(None)

    for backend in backend_candidates:
        if backend is None:
            cap = cv2.VideoCapture(camera_index)
        else:
            cap = cv2.VideoCapture(camera_index, backend)

        if cap.isOpened():
            return cap

        cap.release()

    return cv2.VideoCapture(camera_index)

# === Open webcam ===
cap = open_camera(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError(
        f"Unable to open camera index {CAMERA_INDEX}. On macOS, allow camera access for VS Code or Terminal in System Settings > Privacy & Security > Camera."
    )
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# === Nose-only landmark indices (for stable up/down eye sphere tracking) ===
# These landmarks are near the nose and are less affected by lateral head movement
nose_indices = [4, 45, 275, 220, 440, 1, 5, 51, 281, 44, 274, 241, 
                461, 125, 354, 218, 438, 195, 167, 393, 165, 391,
                3, 248]

# ===== NEW: File writing for screen position =====
screen_position_file = os.environ.get(
    "SCREEN_POSITION_FILE",
    os.path.join(PROJECT_DIR, "screen_position.txt")
)

def write_screen_position(x, y):
    """Write screen position to file, overwriting the same line"""
    output_dir = os.path.dirname(screen_position_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(screen_position_file, 'w') as f:
        f.write(f"{x},{y}\n")

def _rot_x(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0],
                     [0, ca, -sa],
                     [0, sa,  ca]], dtype=float)

def _rot_y(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ ca, 0, sa],
                     [  0, 1,  0],
                     [-sa, 0, ca]], dtype=float)

def _normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v

def _focal_px(width, fov_deg):
    # horizontal pinhole focal length
    return 0.5 * width / math.tan(math.radians(fov_deg) * 0.5)


def quadratic_features(raw_yaw_deg, raw_pitch_deg):
    return np.array(
        [
            1.0,
            raw_yaw_deg,
            raw_pitch_deg,
            raw_yaw_deg * raw_pitch_deg,
            raw_yaw_deg * raw_yaw_deg,
            raw_pitch_deg * raw_pitch_deg,
        ],
        dtype=float,
    )


def regression_ready():
    return regression_coefficients_x is not None and regression_coefficients_y is not None


def clear_regression_calibration():
    global regression_coefficients_x, regression_coefficients_y, regression_fit_error_px
    global regression_loaded_from_disk

    regression_coefficients_x = None
    regression_coefficients_y = None
    regression_fit_error_px = None
    regression_loaded_from_disk = False


def save_regression_calibration():
    if not regression_ready():
        return False

    output_dir = os.path.dirname(REGRESSION_CALIBRATION_FILE)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    np.savez(
        REGRESSION_CALIBRATION_FILE,
        format_version=np.array([1], dtype=np.int32),
        regression_coefficients_x=np.asarray(regression_coefficients_x, dtype=float),
        regression_coefficients_y=np.asarray(regression_coefficients_y, dtype=float),
        regression_fit_error_px=np.array(
            [np.nan if regression_fit_error_px is None else regression_fit_error_px],
            dtype=float,
        ),
        screen_width=np.array([MONITOR_WIDTH], dtype=np.int32),
        screen_height=np.array([MONITOR_HEIGHT], dtype=np.int32),
        saved_at=np.array([time.time()], dtype=float),
    )
    print(f"[9-Point Calibration] Saved regression to {REGRESSION_CALIBRATION_FILE}")
    return True


def load_regression_calibration(announce=True):
    global regression_coefficients_x, regression_coefficients_y, regression_fit_error_px
    global regression_loaded_from_disk, nine_point_status_text

    if not os.path.exists(REGRESSION_CALIBRATION_FILE):
        return False

    try:
        calibration_data = np.load(REGRESSION_CALIBRATION_FILE, allow_pickle=False)
        coefficient_x = np.asarray(calibration_data["regression_coefficients_x"], dtype=float).reshape(-1)
        coefficient_y = np.asarray(calibration_data["regression_coefficients_y"], dtype=float).reshape(-1)
        if coefficient_x.shape != (6,) or coefficient_y.shape != (6,):
            raise ValueError("Expected 6 regression coefficients for both axes.")

        fit_error = calibration_data.get("regression_fit_error_px")
        if fit_error is None:
            regression_fit_error_px = None
        else:
            fit_error_value = float(np.asarray(fit_error, dtype=float).reshape(-1)[0])
            regression_fit_error_px = None if math.isnan(fit_error_value) else fit_error_value

        regression_coefficients_x = coefficient_x
        regression_coefficients_y = coefficient_y
        regression_loaded_from_disk = True
        if regression_fit_error_px is not None:
            nine_point_status_text = f"Loaded saved 9-point regression ({regression_fit_error_px:.1f}px RMSE)."
        else:
            nine_point_status_text = "Loaded saved 9-point regression."

        if announce:
            print(f"[9-Point Calibration] Loaded regression from {REGRESSION_CALIBRATION_FILE}")
        return True
    except Exception as exc:
        clear_regression_calibration()
        print(f"[9-Point Calibration] Failed to load saved regression: {exc}")
        return False


def default_nine_point_targets(margin_ratio=0.1):
    if not 0.0 <= margin_ratio < 0.5:
        raise ValueError("margin_ratio must be between 0.0 and 0.5.")

    width_scale = max(MONITOR_WIDTH - 1, 1)
    height_scale = max(MONITOR_HEIGHT - 1, 1)
    x_positions = [margin_ratio, 0.5, 1.0 - margin_ratio]
    y_positions = [margin_ratio, 0.5, 1.0 - margin_ratio]

    return [
        (int(round(x_ratio * width_scale)), int(round(y_ratio * height_scale)))
        for y_ratio in y_positions
        for x_ratio in x_positions
    ]


def close_nine_point_window():
    global calibration_window_ready

    if not calibration_window_ready:
        return

    try:
        cv2.destroyWindow(CALIBRATION_WINDOW_NAME)
    except cv2.error:
        pass

    calibration_window_ready = False


def reset_nine_point_calibration(clear_regression=False):
    global nine_point_active, nine_point_targets, nine_point_samples, nine_point_index
    global nine_point_status_text

    nine_point_active = False
    nine_point_targets = []
    nine_point_samples = []
    nine_point_index = 0
    nine_point_status_text = ""
    recent_gaze_angles.clear()
    close_nine_point_window()

    if clear_regression:
        clear_regression_calibration()


def start_nine_point_calibration():
    global nine_point_active, nine_point_targets, nine_point_samples, nine_point_index
    global nine_point_status_text

    reset_nine_point_calibration(clear_regression=False)
    nine_point_active = True
    nine_point_targets = default_nine_point_targets()
    nine_point_samples = []
    nine_point_index = 0
    nine_point_status_text = "Target 1/9: move your gaze and hold steady."
    print("[9-Point Calibration] Started. Follow the fullscreen targets.")


def fit_nine_point_regression():
    global regression_coefficients_x, regression_coefficients_y, regression_fit_error_px
    global regression_loaded_from_disk
    global nine_point_status_text

    target_count = len(nine_point_targets)
    if target_count == 0 or len(nine_point_samples) < target_count:
        return False

    width_scale = max(MONITOR_WIDTH - 1, 1)
    height_scale = max(MONITOR_HEIGHT - 1, 1)
    design_rows = []
    target_x_values = []
    target_y_values = []

    for target_point, raw_yaw_deg, raw_pitch_deg in nine_point_samples:
        design_rows.append(quadratic_features(raw_yaw_deg, raw_pitch_deg))
        target_x_values.append(target_point[0] / width_scale)
        target_y_values.append(target_point[1] / height_scale)

    design_matrix = np.vstack(design_rows)
    target_x = np.asarray(target_x_values, dtype=float)
    target_y = np.asarray(target_y_values, dtype=float)

    regression_coefficients_x = np.linalg.lstsq(design_matrix, target_x, rcond=None)[0]
    regression_coefficients_y = np.linalg.lstsq(design_matrix, target_y, rcond=None)[0]
    regression_loaded_from_disk = False

    predicted_x = design_matrix @ regression_coefficients_x
    predicted_y = design_matrix @ regression_coefficients_y
    error_x_px = (predicted_x - target_x) * width_scale
    error_y_px = (predicted_y - target_y) * height_scale
    regression_fit_error_px = float(np.sqrt(np.mean(error_x_px * error_x_px + error_y_px * error_y_px)))
    nine_point_status_text = f"9-point regression ready ({regression_fit_error_px:.1f}px RMSE)."
    save_regression_calibration()
    return True


def capture_nine_point_sample():
    global nine_point_active, nine_point_index, nine_point_status_text

    target_count = len(nine_point_targets)
    if not nine_point_active or target_count == 0 or nine_point_index >= target_count:
        return False

    sample_count = len(recent_gaze_angles)
    target_number = min(nine_point_index + 1, target_count)
    if sample_count < NINE_POINT_CAPTURE_FRAMES:
        nine_point_status_text = (
            f"Target {target_number}/{target_count}: hold steady "
            f"{sample_count}/{NINE_POINT_CAPTURE_FRAMES}."
        )
        return False

    sample_array = np.asarray(recent_gaze_angles, dtype=float)
    yaw_std = float(np.std(sample_array[:, 0]))
    pitch_std = float(np.std(sample_array[:, 1]))
    if yaw_std > NINE_POINT_MAX_STD_DEG or pitch_std > NINE_POINT_MAX_STD_DEG:
        nine_point_status_text = (
            f"Target {target_number}/{target_count}: keep steadier before capture."
        )
        return False

    median_yaw_deg, median_pitch_deg = np.median(sample_array, axis=0)
    target_point = nine_point_targets[nine_point_index]
    nine_point_samples.append((target_point, float(median_yaw_deg), float(median_pitch_deg)))
    print(
        f"[9-Point Calibration] Captured target {target_number}/{target_count} "
        f"at ({target_point[0]}, {target_point[1]})."
    )

    nine_point_index += 1
    recent_gaze_angles.clear()

    if nine_point_index >= target_count:
        nine_point_active = False
        close_nine_point_window()
        fit_success = fit_nine_point_regression()
        if fit_success:
            print(f"[9-Point Calibration] Regression fitted. RMSE={regression_fit_error_px:.2f}px")
        else:
            nine_point_status_text = "9-point regression fit failed."
            print("[9-Point Calibration] Regression fit failed.")
        return fit_success

    nine_point_status_text = f"Target {nine_point_index + 1}/{target_count}: move your gaze and hold steady."
    return True


def draw_nine_point_calibration_window():
    global calibration_window_ready

    if not nine_point_active:
        return

    if not calibration_window_ready:
        cv2.namedWindow(CALIBRATION_WINDOW_NAME, cv2.WINDOW_NORMAL)
        try:
            cv2.setWindowProperty(CALIBRATION_WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        except cv2.error:
            pass
        calibration_window_ready = True

    canvas = np.zeros((MONITOR_HEIGHT, MONITOR_WIDTH, 3), dtype=np.uint8)
    canvas[:] = (18, 18, 18)

    target_count = len(nine_point_targets)
    current_target = min(nine_point_index + 1, target_count) if target_count else 0
    for index, (target_x, target_y) in enumerate(nine_point_targets):
        point = (int(target_x), int(target_y))
        if index < len(nine_point_samples):
            cv2.circle(canvas, point, 18, (40, 180, 40), -1, lineType=cv2.LINE_AA)
        elif index == nine_point_index:
            cv2.circle(canvas, point, 26, (0, 80, 255), 2, lineType=cv2.LINE_AA)
            cv2.line(canvas, (point[0] - 34, point[1]), (point[0] + 34, point[1]), (0, 80, 255), 2)
            cv2.line(canvas, (point[0], point[1] - 34), (point[0], point[1] + 34), (0, 80, 255), 2)
        else:
            cv2.circle(canvas, point, 14, (90, 90, 90), 2, lineType=cv2.LINE_AA)

    header = f"9-Point Calibration {current_target}/{target_count}" if target_count else "9-Point Calibration"
    progress = min(len(recent_gaze_angles), NINE_POINT_CAPTURE_FRAMES)
    status_line = nine_point_status_text or "Follow the current target."

    cv2.putText(canvas, header, (48, 64), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (240, 240, 240), 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"Stable frames: {progress}/{NINE_POINT_CAPTURE_FRAMES}",
        (48, 104),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (200, 200, 200),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(canvas, status_line, (48, 144), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "Press ESC to cancel this calibration run.",
        (48, MONITOR_HEIGHT - 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (180, 180, 180),
        2,
        cv2.LINE_AA,
    )

    cv2.imshow(CALIBRATION_WINDOW_NAME, canvas)


load_regression_calibration(announce=True)


def create_monitor_plane(head_center, R_final, face_landmarks, w, h, 
                         forward_hint=None, gaze_origin=None, gaze_dir=None):
    """
    Build a 60cm x 40cm plane 50cm in front of the face, in world units.
    Monitor is oriented horizontally like a real monitor (top edge parallel to global X-axis).
    """
    # 1) Estimate scale from chin<->forehead distance
    try:
        lm_chin = face_landmarks[152]
        lm_fore = face_landmarks[10]
        chin_w = np.array([lm_chin.x * w,  lm_chin.y * h,  lm_chin.z * w], dtype=float)
        fore_w = np.array([lm_fore.x * w,  lm_fore.y * h,  lm_fore.z * w], dtype=float)
        face_h_units = np.linalg.norm(fore_w - chin_w)
        upc = face_h_units / 15.0  # units per cm
    except Exception:
        upc = 5.0
    
    # 2) Monitor geometry in world units
    dist_cm = 50.0

    mon_w_cm, mon_h_cm = 60.0, 40.0
    half_w = (mon_w_cm * 0.5) * upc
    half_h = (mon_h_cm * 0.5) * upc

    # Head forward vector
    head_forward = -R_final[:, 2]
    if forward_hint is not None:
        head_forward = forward_hint / np.linalg.norm(forward_hint)

    # --- NEW: use gaze ray intersection ---
    if gaze_origin is not None and gaze_dir is not None:
        gaze_dir = gaze_dir / np.linalg.norm(gaze_dir)

        # Place the monitor so its center is exactly at some point on the gaze ray
        # For simplicity: choose intersection at 50 cm from head_center along head_forward
        plane_point = head_center + head_forward * (50.0 * upc)
        plane_normal = head_forward

        denom = np.dot(plane_normal, gaze_dir)
        if abs(denom) > 1e-6:
            t = np.dot(plane_normal, plane_point - gaze_origin) / denom
            center_w = gaze_origin + t * gaze_dir
        else:
            # fallback: use fixed distance
            center_w = head_center + head_forward * (50.0 * upc)
    else:
        # fallback: original placement
        center_w = head_center + head_forward * (50.0 * upc)

    # Compute right/up using head orientation
    world_up = np.array([0, -1, 0], dtype=float)
    head_right = np.cross(world_up, head_forward)
    head_right /= np.linalg.norm(head_right)
    head_up = np.cross(head_forward, head_right)
    head_up /= np.linalg.norm(head_up)

    # Corners
    p0 = center_w - head_right * half_w - head_up * half_h
    p1 = center_w + head_right * half_w - head_up * half_h
    p2 = center_w + head_right * half_w + head_up * half_h
    p3 = center_w - head_right * half_w + head_up * half_h

    normal_w = head_forward / (np.linalg.norm(head_forward) + 1e-9)
    return [p0, p1, p2, p3], center_w, normal_w, upc




def is_hotkey_pressed(key_name):
    global keyboard, global_hotkey_warning_shown
    if keyboard is None:
        return False

    try:
        return keyboard.is_pressed(key_name)
    except Exception as exc:
        if not global_hotkey_warning_shown:
            print(f"[Input] Global hotkeys unavailable ({exc}). Use the OpenCV window shortcuts instead.")
            global_hotkey_warning_shown = True
        keyboard = None
        return False

def toggle_mouse_control():
    global mouse_control_enabled, last_mouse_toggle_time
    if pyautogui is None:
        print("[Mouse Control] PyAutoGUI unavailable; mouse control disabled on this machine.")
        return

    now = time.time()
    if now - last_mouse_toggle_time < 0.3:
        return

    last_mouse_toggle_time = now
    mouse_control_enabled = not mouse_control_enabled
    print(f"[Mouse Control] {'Enabled' if mouse_control_enabled else 'Disabled'}")

def update_orbit_from_keypress(key_code):
    """Keyboard orbit controls using the focused OpenCV window."""
    global orbit_yaw, orbit_pitch, orbit_radius
    if key_code < 0:
        return

    key = key_code & 0xFF
    yaw_step   = math.radians(1.5)
    pitch_step = math.radians(1.5)
    zoom_step  = 12.0

    changed = False

    # Rotate
    if key in (ord('j'), ord('J')):  # yaw left
        orbit_yaw -= yaw_step; changed = True
    if key in (ord('l'), ord('L')):  # yaw right
        orbit_yaw += yaw_step; changed = True
    if key in (ord('i'), ord('I')):  # pitch up
        orbit_pitch += pitch_step; changed = True
    if key in (ord('k'), ord('K')):  # pitch down
        orbit_pitch -= pitch_step; changed = True

    # Zoom
    if key == ord('['):  # zoom out
        orbit_radius += zoom_step; changed = True
    if key == ord(']'):  # zoom in
        orbit_radius = max(80.0, orbit_radius - zoom_step); changed = True

    # Reset (prints every frame while held)
    if key in (ord('r'), ord('R')):
        orbit_yaw = 0.0
        orbit_pitch = 0.0
        orbit_radius = 600.0
        changed = True

    # Clamp pitch & radius
    orbit_pitch = max(math.radians(-89), min(math.radians(89), orbit_pitch))
    orbit_radius = max(80.0, orbit_radius)

    if changed:
        print(f"[Orbit Debug] yaw={math.degrees(orbit_yaw):.2f}°, "
              f"pitch={math.degrees(orbit_pitch):.2f}°, "
              f"radius={orbit_radius:.2f}, "
              f"fov={orbit_fov_deg:.1f}°")




def compute_scale(points_3d):
    # Use average pairwise distance for robustness
    n = len(points_3d)
    total = 0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(points_3d[i] - points_3d[j])
            total += dist
            count += 1
    return total / count if count > 0 else 1.0

def draw_gaze(frame, eye_center, iris_center, eye_radius, color, gaze_length):
    # Gaze vector
    gaze_direction = iris_center - eye_center
    gaze_direction /= np.linalg.norm(gaze_direction)
    gaze_endpoint = eye_center + gaze_direction * gaze_length

    cv2.line(frame, tuple(int(v) for v in eye_center[:2]), tuple(int(v) for v in gaze_endpoint[:2]), color, 2)

    # Segment points
    iris_offset = eye_center + gaze_direction * (1.2 * eye_radius)

    # ---- PART 1: back segment (behind iris) ----
    cv2.line(
        frame,
        (int(eye_center[0]), int(eye_center[1])),
        (int(iris_offset[0]), int(iris_offset[1])),
        color,
        1
    )

    # ---- IRIS (occludes part of the ray) ----
    up_dir = np.array([0, -1, 0])
    right_dir = np.cross(gaze_direction, up_dir)
    if np.linalg.norm(right_dir) < 1e-6:
        right_dir = np.array([1, 0, 0])
    up_dir = np.cross(right_dir, gaze_direction)
    up_dir /= np.linalg.norm(up_dir)
    right_dir /= np.linalg.norm(right_dir)
    ellipse_axes = (
        int((eye_radius / 3) * np.linalg.norm(right_dir[:2])),
        int((eye_radius / 3) * np.linalg.norm(up_dir[:2]))
    )
    angle = math.degrees(math.atan2(gaze_direction[1], gaze_direction[0]))

    # ---- PART 2: front segment (on top of iris) ----
    cv2.line(
        frame,
        (int(iris_offset[0]), int(iris_offset[1])),
        (int(gaze_endpoint[0]), int(gaze_endpoint[1])),
        color,
        1
    )

def draw_wireframe_cube(frame, center, R, size=80):
    # Given a center and rotation matrix, draw a cube aligned to that orientation
    right = R[:, 0]
    up = -R[:, 1]
    forward = -R[:, 2]

    hw, hh, hd = size * 1, size * 1, size * 1

    def corner(x_sign, y_sign, z_sign):
        return (center +
                x_sign * hw * right +
                y_sign * hh * up +
                z_sign * hd * forward)

    # 8 corners of the cube
    corners = [corner(x, y, z) for x in [-1, 1] for y in [1, -1] for z in [-1, 1]]
    projected = [(int(pt[0]), int(pt[1])) for pt in corners]

    # Edges connecting the corners
    edges = [
        (0, 1), (1, 3), (3, 2), (2, 0),
        (4, 5), (5, 7), (7, 6), (6, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    for i, j in edges:
        cv2.line(frame, projected[i], projected[j], (255, 128, 0), 2)

def compute_and_draw_coordinate_box(frame, face_landmarks, indices, ref_matrix_container, color=(0, 255, 0), size=80):
    # Extract 3D positions of selected landmarks
    points_3d = np.array([
        [face_landmarks[i].x * w, face_landmarks[i].y * h, face_landmarks[i].z * w]
        for i in indices
    ])

    # Compute the average position as the center of this substructure
    center = np.mean(points_3d, axis=0)

    # Draw the raw 2D landmark points
    for i in indices:
        x, y = int(face_landmarks[i].x * w), int(face_landmarks[i].y * h)
        cv2.circle(frame, (x, y), 3, color, -1)

    # PCA-based orientation: Compute eigenvectors of the covariance matrix
    centered = points_3d - center
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvecs = eigvecs[:, np.argsort(-eigvals)]  # Sort by descending eigenvalue (major axes)

    # Ensure the orientation matrix is right-handed
    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, 2] *= -1

    # Convert to Euler angles and re-construct rotation matrix (optional but clarifies the transform)
    r = Rscipy.from_matrix(eigvecs)
    roll, pitch, yaw = r.as_euler('zyx', degrees=False)
    yaw *= 1
    roll *= 1
    R_final = Rscipy.from_euler('zyx', [roll, pitch, yaw]).as_matrix()

    # === Stabilize rotation with reference matrix to avoid flipping during eigenvector sign change ===
    if ref_matrix_container[0] is None:
        ref_matrix_container[0] = R_final.copy()
    else:
        R_ref = ref_matrix_container[0]
        for i in range(3):
            if np.dot(R_final[:, i], R_ref[:, i]) < 0:
                R_final[:, i] *= -1

    # Draw cube and orientation axes on the image
    draw_wireframe_cube(frame, center, R_final, size)

    # Draw X (green), Y (blue), Z (red) axes
    axis_length = size * 1.2
    axis_dirs = [R_final[:, 0], -R_final[:, 1], -R_final[:, 2]]
    axis_colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]

    for i in range(3):
        end_pt = center + axis_dirs[i] * axis_length
        cv2.line(frame, (int(center[0]), int(center[1])), (int(end_pt[0]), int(end_pt[1])), axis_colors[i], 2)

    return center, R_final, points_3d


def raw_gaze_angles(combined_gaze_direction):
    reference_forward = np.array([0.0, 0.0, -1.0], dtype=float)
    avg_direction = combined_gaze_direction / np.linalg.norm(combined_gaze_direction)

    xz_proj = np.array([avg_direction[0], 0.0, avg_direction[2]], dtype=float)
    xz_proj /= np.linalg.norm(xz_proj)
    yaw_rad = math.acos(np.clip(np.dot(reference_forward, xz_proj), -1.0, 1.0))
    if avg_direction[0] < 0:
        yaw_rad = -yaw_rad

    yz_proj = np.array([0.0, avg_direction[1], avg_direction[2]], dtype=float)
    yz_proj /= np.linalg.norm(yz_proj)
    pitch_rad = math.acos(np.clip(np.dot(reference_forward, yz_proj), -1.0, 1.0))
    if avg_direction[1] > 0:
        pitch_rad = -pitch_rad

    yaw_deg = np.degrees(yaw_rad)
    pitch_deg = np.degrees(pitch_rad)

    if yaw_deg < 0:
        yaw_deg = -yaw_deg
    elif yaw_deg > 0:
        yaw_deg = -yaw_deg

    return float(yaw_deg), float(pitch_deg)


def linear_screen_coordinates(raw_yaw_deg, raw_pitch_deg, offset_yaw, offset_pitch):
    yaw_deg = raw_yaw_deg + offset_yaw
    pitch_deg = raw_pitch_deg + offset_pitch

    screen_x = int(((yaw_deg + SCREEN_YAW_DEGREES) / (2 * SCREEN_YAW_DEGREES)) * MONITOR_WIDTH)
    screen_y = int(((SCREEN_PITCH_DEGREES - pitch_deg) / (2 * SCREEN_PITCH_DEGREES)) * MONITOR_HEIGHT)

    screen_x = max(10, min(screen_x, MONITOR_WIDTH - 10))
    screen_y = max(10, min(screen_y, MONITOR_HEIGHT - 10))
    return screen_x, screen_y

def convert_gaze_to_screen_coordinates(combined_gaze_direction, calibration_offset_yaw, calibration_offset_pitch):
    """
    Convert 3D gaze direction vector to 2D screen coordinates
    This function is adapted from the old script's vector-to-screen mapping logic
    """
    raw_yaw_deg, raw_pitch_deg = raw_gaze_angles(combined_gaze_direction)
    screen_x, screen_y = linear_screen_coordinates(
        raw_yaw_deg,
        raw_pitch_deg,
        calibration_offset_yaw,
        calibration_offset_pitch,
    )
    return screen_x, screen_y, raw_yaw_deg, raw_pitch_deg

def render_debug_view_orbit(
    h, w,
    head_center3d=None,
    sphere_world_l=None, scaled_radius_l=None,
    sphere_world_r=None, scaled_radius_r=None,
    iris3d_l=None, iris3d_r=None,
    left_locked=False, right_locked=False,
    landmarks3d=None,
    combined_dir=None,
    gaze_len=430,
    monitor_corners=None,
    monitor_center=None,
    monitor_normal=None,
    gaze_markers=None,
):
    if head_center3d is None:
        return

    debug = np.zeros((h, w, 3), dtype=np.uint8)

    # --- Choose orbit pivot ---
    head_w = np.asarray(head_center3d, dtype=float)

    # NEW: if we've frozen the world, orbit around the frozen pivot (monitor center at calib)
    global debug_world_frozen, orbit_pivot_frozen
    if debug_world_frozen and orbit_pivot_frozen is not None:
        pivot_w = np.asarray(orbit_pivot_frozen, dtype=float)
    else:
        if monitor_center is not None:
            pivot_w = (head_w + np.asarray(monitor_center, dtype=float)) * 0.5
        else:
            pivot_w = head_w

    # --- Camera pose (orbit around pivot_w) ---
    f_px = _focal_px(w, orbit_fov_deg)
    cam_offset = _rot_y(orbit_yaw) @ (_rot_x(orbit_pitch) @ np.array([0.0, 0.0, orbit_radius]))
    cam_pos = pivot_w + cam_offset

    up_world = np.array([0.0, -1.0, 0.0])   # image-space up is -Y
    fwd = _normalize(pivot_w - cam_pos)     # look at pivot
    right = _normalize(np.cross(fwd, up_world))
    up = _normalize(np.cross(right, fwd))
    V = np.stack([right, up, fwd], axis=0)

    def project_point(P):
        Pw = np.asarray(P, dtype=float)
        Pc = V @ (Pw - cam_pos)
        if Pc[2] <= 1e-3:
            return None
        x = f_px * (Pc[0] / Pc[2]) + w * 0.5
        y = -f_px * (Pc[1] / Pc[2]) + h * 0.5
        if not (np.isfinite(x) and np.isfinite(y)):
            return None
        return (int(x), int(y)), Pc[2]

    # --- helper draws ---
    def draw_poly_3d(pts, color=(0, 200, 255), thickness=2):
        projs = [project_point(p) for p in pts]
        if any(p is None for p in projs): return
        p2 = [p[0] for p in projs]
        for a, b in zip(p2, p2[1:] + [p2[0]]):
            cv2.line(debug, a, b, color, thickness)

    def draw_cross_3d(P, size=12, color=(255, 0, 255), thickness=2):
        res = project_point(P)
        if res is None: return
        (x, y), _ = res
        cv2.line(debug, (x - size, y), (x + size, y), color, thickness)
        cv2.line(debug, (x, y - size), (x, y + size), color, thickness)

    def draw_arrow_3d(P0, P1, color=(0, 200, 255), thickness=3):
        a = project_point(P0); b = project_point(P1)
        if a is None or b is None: return
        p0, p1 = a[0], b[0]
        cv2.line(debug, p0, p1, color, thickness)
        v = np.array([p1[0]-p0[0], p1[1]-p0[1]], dtype=float)
        n = np.linalg.norm(v)
        if n > 1e-3:
            v /= n
            l = np.array([-v[1], v[0]])
            ah = 10
            a1 = (int(p1[0] - v[0]*ah + l[0]*ah*0.6), int(p1[1] - v[1]*ah + l[1]*ah*0.6))
            a2 = (int(p1[0] - v[0]*ah - l[0]*ah*0.6), int(p1[1] - v[1]*ah - l[1]*ah*0.6))
            cv2.line(debug, p1, a1, color, thickness)
            cv2.line(debug, p1, a2, color, thickness)

    # --- Landmarks ---
    if landmarks3d is not None:
        for P in landmarks3d:
            res = project_point(P)
            if res is not None:
                cv2.circle(debug, res[0], 0, (200, 200, 200), -1)

    # --- Head center ---
    draw_cross_3d(head_w, size=12, color=(255, 0, 255), thickness=2)
    hc2d = project_point(head_w)
    if hc2d is not None:
        cv2.putText(debug, "Head Center", (hc2d[0][0] + 12, hc2d[0][1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)

    # --- Pivot visual (small cross + line to head and monitor) ---
    draw_cross_3d(pivot_w, size=8, color=(180, 120, 255), thickness=2)
    if monitor_center is not None:
        mc2d = project_point(monitor_center)
        pv2d = project_point(pivot_w)
        if mc2d is not None and pv2d is not None and hc2d is not None:
            cv2.line(debug, pv2d[0], hc2d[0], (160, 100, 255), 1)
            cv2.line(debug, pv2d[0], mc2d[0], (160, 100, 255), 1)

    # --- Eyes + per-eye gaze (unchanged from your version) ---
    left_dir = None
    right_dir = None

    if left_locked and sphere_world_l is not None:
        res = project_point(sphere_world_l)
        if res is not None:
            (cx, cy), z = res
            r_px = max(2, int((scaled_radius_l if scaled_radius_l else 6) * f_px / max(z, 1e-3)))
            cv2.circle(debug, (cx, cy), r_px, (255, 255, 25), 1)
            if iris3d_l is not None:
                left_dir = np.asarray(iris3d_l) - np.asarray(sphere_world_l)
                p1 = project_point(np.asarray(sphere_world_l) + _normalize(left_dir) * gaze_len)
                if p1 is not None:
                    cv2.line(debug, (cx, cy), p1[0], (155, 155, 25), 1)
    elif iris3d_l is not None:
        res = project_point(iris3d_l)
        if res is not None:
            cv2.circle(debug, res[0], 2, (255, 255, 25), 1)

    if right_locked and sphere_world_r is not None:
        res = project_point(sphere_world_r)
        if res is not None:
            (cx, cy), z = res
            r_px = max(2, int((scaled_radius_r if scaled_radius_r else 6) * f_px / max(z, 1e-3)))
            cv2.circle(debug, (cx, cy), r_px, (25, 255, 255), 1)
            if iris3d_r is not None:
                right_dir = np.asarray(iris3d_r) - np.asarray(sphere_world_r)
                p1 = project_point(np.asarray(sphere_world_r) + _normalize(right_dir) * gaze_len)
                if p1 is not None:
                    cv2.line(debug, (cx, cy), p1[0], (25, 155, 155), 1)
    elif iris3d_r is not None:
        res = project_point(iris3d_r)
        if res is not None:
            cv2.circle(debug, res[0], 2, (25, 255, 255), 1)

    if left_locked and right_locked and sphere_world_l is not None and sphere_world_r is not None:
        origin_mid = (np.asarray(sphere_world_l) + np.asarray(sphere_world_r)) / 2.0
        if combined_dir is None and (left_dir is not None or right_dir is not None):
            parts = []
            if left_dir is not None:  parts.append(_normalize(left_dir))
            if right_dir is not None: parts.append(_normalize(right_dir))
            if parts:
                combined_dir = _normalize(np.mean(parts, axis=0))
        if combined_dir is not None:
            p0 = project_point(origin_mid)
            p1 = project_point(origin_mid + _normalize(combined_dir) * (gaze_len * 1.2))
            if p0 is not None and p1 is not None:
                cv2.line(debug, p0[0], p1[0], (155, 200, 10), 2)

    # --- Monitor plane ---
    if monitor_corners is not None:
        def draw_poly(points, color, thickness):
            projs = [project_point(p) for p in points]
            if any(p is None for p in projs): return
            p2 = [p[0] for p in projs]
            for a, b in zip(p2, p2[1:] + [p2[0]]):
                cv2.line(debug, a, b, color, thickness)
        draw_poly(monitor_corners, (0, 200, 255), 2)
        draw_poly([monitor_corners[0], monitor_corners[2]], (0, 150, 210), 1)
        draw_poly([monitor_corners[1], monitor_corners[3]], (0, 150, 210), 1)
        if monitor_center is not None:
            draw_cross_3d(monitor_center, size=8, color=(0, 200, 255), thickness=2)
            if monitor_normal is not None:
                tip = np.asarray(monitor_center) + np.asarray(monitor_normal) * (20.0 * (units_per_cm or 1.0))
                draw_arrow_3d(monitor_center, tip, color=(0, 220, 255), thickness=2)

    # --- Stored gaze markers on the monitor plane (green circles) ---
    if (gaze_markers and monitor_corners is not None):
        p0, p1, p2, p3 = [np.asarray(p, dtype=float) for p in monitor_corners]
        u = p1 - p0  # width direction
        v = p3 - p0  # height direction
        width_world = float(np.linalg.norm(u))
        if width_world > 1e-9:
            u_hat = u / width_world
            r_world = 0.01 * width_world  # 2% of width
            for (a, b) in gaze_markers:
                Pm = p0 + a * u + b * v
                projP = project_point(Pm)
                projR = project_point(Pm + u_hat * r_world)
                if projP is not None and projR is not None:
                    center_px = projP[0]
                    r_px = int(max(1, np.linalg.norm(np.array(projR[0]) - np.array(center_px))))
                    cv2.circle(debug, center_px, r_px, (0, 255, 0), 1, lineType=cv2.LINE_AA)



    # --- Gaze hit on monitor plane (circle at intersection) ---
    if (monitor_corners is not None and monitor_center is not None and monitor_normal is not None
        and combined_dir is not None
        and sphere_world_l is not None and sphere_world_r is not None):

        # Ray: origin at midpoint between eyes; direction = combined gaze
        O = (np.asarray(sphere_world_l, dtype=float) + np.asarray(sphere_world_r, dtype=float)) * 0.5
        D = _normalize(np.asarray(combined_dir, dtype=float))

        # Plane: through monitor_center with normal = monitor_normal
        C = np.asarray(monitor_center, dtype=float)
        N = _normalize(np.asarray(monitor_normal, dtype=float))

        denom = float(np.dot(N, D))
        if abs(denom) > 1e-6:
            t = float(np.dot(N, (C - O)) / denom)
            if t > 0.0:
                P = O + t * D  # world-space intersection point

                # Inside-quad test using monitor's local axes (top-left p0, top-right p1, bottom-left p3)
                p0, p1, p2, p3 = [np.asarray(p, dtype=float) for p in monitor_corners]
                u = p1 - p0             # horizontal (width) vector
                v = p3 - p0             # vertical (height) vector
                wv = P  - p0

                u_len2 = float(np.dot(u, u))
                v_len2 = float(np.dot(v, v))
                if u_len2 > 1e-9 and v_len2 > 1e-9:
                    a = float(np.dot(wv, u) / u_len2)  # 0..1 across width
                    b = float(np.dot(wv, v) / v_len2)  # 0..1 across height

                    if 0.0 <= a <= 1.0 and 0.0 <= b <= 1.0:
                        # Project center to pixels
                        projP = project_point(P)
                        if projP is not None:
                            center_px = projP[0]

                            # Circle radius = 5% of monitor width (world), projected to pixels
                            width_world = math.sqrt(u_len2)
                            r_world = 0.05 * width_world
                            u_hat = u / max(width_world, 1e-9)

                            projR = project_point(P + u_hat * r_world)
                            if projR is not None:
                                r_px = int(max(1, np.linalg.norm(np.array(projR[0]) - np.array(center_px))))
                                cv2.circle(debug, center_px, r_px, (0, 255, 255), 2, lineType=cv2.LINE_AA)


    status_lines = []
    if nine_point_active and nine_point_targets:
        target_number = min(nine_point_index + 1, len(nine_point_targets))
        status_lines.append(f"9-point target {target_number}/{len(nine_point_targets)}")
        status_lines.append(f"Stable frames {min(len(recent_gaze_angles), NINE_POINT_CAPTURE_FRAMES)}/{NINE_POINT_CAPTURE_FRAMES}")
    elif regression_ready() and regression_fit_error_px is not None:
        status_lines.append(f"9-point regression ready ({regression_fit_error_px:.1f}px RMSE)")
    elif calibration_offset_yaw != 0 or calibration_offset_pitch != 0:
        status_lines.append("Center calibration ready")
    else:
        status_lines.append("Linear mapping only")

    if nine_point_status_text:
        status_lines.append(nine_point_status_text)

    for index, text in enumerate(status_lines[:3]):
        cv2.putText(
            debug,
            text,
            (10, 24 + index * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

    # --- Key command help text in lower-left ---
    help_text = [
        "C = lock eye spheres",
        "S = calibrate screen center",
        "9 = start 9-point calibration",
        "ESC = cancel 9-point run",
        "J = yaw left",
        "L = yaw right",
        "I = pitch up",
        "K = pitch down",
        "[ = zoom out",
        "] = zoom in",
        "R = reset view",
        "X = add marker",
        "q = quit",
        "M = toggle mouse control"
    ]

    font        = cv2.FONT_HERSHEY_SIMPLEX
    font_scale  = 0.5
    thickness   = 1
    line_height = 18  # pixels between lines

    # Start a bit above the bottom-left corner
    y0 = h - (len(help_text) * line_height) - 10
    x0 = 10

    for i, text in enumerate(help_text):
        y = y0 + i * line_height
        cv2.putText(debug, text, (x0, y), font, font_scale, (200, 200, 200), thickness, cv2.LINE_AA)


    cv2.imshow("Head/Eye Debug", debug)



def mouse_mover():
    """Mouse movement thread from old script"""
    while True:
        if mouse_control_enabled and pyautogui is not None:
            with mouse_lock:
                x, y = mouse_target
            pyautogui.moveTo(x, y)
        time.sleep(0.01)  # adjust for responsiveness

# Start mouse movement thread
threading.Thread(target=mouse_mover, daemon=True).start()

# Eye sphere tracking variables (from new script)
left_sphere_locked = False
left_sphere_local_offset = None
left_calibration_nose_scale = None

right_sphere_locked = False
right_sphere_local_offset = None
right_calibration_nose_scale = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    combined_dir = None  # will be filled once you compute a smoothed direction
    face_landmarks = None
    head_center = None
    R_final = None
    nose_points_3d = None
    iris_3d_left = None
    iris_3d_right = None
    sphere_world_l = None
    sphere_world_r = None
    scaled_radius_l = None
    scaled_radius_r = None
    avg_combined_direction = None
    raw_yaw = None
    raw_pitch = None
    screen_x = None
    screen_y = None
    landmarks3d = None

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0].landmark

        # Index for left iris center point (from MediaPipe's iris model)
        left_iris_idx = 468
        right_iris_idx = 473
        left_iris = face_landmarks[left_iris_idx]
        right_iris = face_landmarks[right_iris_idx]

        # Compute and draw stabilized coordinate frame from nose region
        head_center, R_final, nose_points_3d = compute_and_draw_coordinate_box(
            frame,
            face_landmarks,
            nose_indices,
            R_ref_nose,
            color=(0, 255, 0),
            size=80
        )

        # TODO compute this radius using canthus during calibration
        base_radius = 20  # radius at calibration distance

        x_iris_l = int(left_iris.x * w)
        y_iris_l = int(left_iris.y * h)
        # === LEFT EYE visualization ===
        if not left_sphere_locked:
            cv2.circle(frame, (x_iris_l, y_iris_l), 10, (255, 25, 25), 2)
        else:
            current_nose_scale = compute_scale(nose_points_3d)
            scale_ratio = current_nose_scale / left_calibration_nose_scale if left_calibration_nose_scale else 1.0
            scaled_offset = left_sphere_local_offset * scale_ratio
            sphere_world_l = head_center + R_final @ scaled_offset
            x_sphere_l, y_sphere_l = int(sphere_world_l[0]), int(sphere_world_l[1])
            scaled_radius_l = int(base_radius * scale_ratio)
            cv2.circle(frame, (x_sphere_l, y_sphere_l), scaled_radius_l, (255, 255, 25), 2)

        x_iris_r = int(right_iris.x * w)
        y_iris_r = int(right_iris.y * h)
        # === RIGHT EYE visualization ===
        if not right_sphere_locked:
            cv2.circle(frame, (x_iris_r, y_iris_r), 10, (25, 255, 25), 2)
        else:
            current_nose_scale = compute_scale(nose_points_3d)
            scale_ratio_r = current_nose_scale / right_calibration_nose_scale if right_calibration_nose_scale else 1.0
            scaled_offset_r = right_sphere_local_offset * scale_ratio_r
            sphere_world_r = head_center + R_final @ scaled_offset_r
            x_sphere_r, y_sphere_r = int(sphere_world_r[0]), int(sphere_world_r[1])
            scaled_radius_r = int(base_radius * scale_ratio_r)
            cv2.circle(frame, (x_sphere_r, y_sphere_r), scaled_radius_r, (25, 255, 255), 2)

        iris_3d_left = np.array([left_iris.x * w, left_iris.y * h, left_iris.z * w])
        iris_3d_right = np.array([right_iris.x * w, right_iris.y * h, right_iris.z * w])
        
        if left_sphere_locked and right_sphere_locked:
            # ==== DRAW LEFT AND RIGHT GAZE ====
            draw_gaze(frame, sphere_world_l, iris_3d_left, scaled_radius_l, (55, 255, 0), 130)   
            draw_gaze(frame, sphere_world_r, iris_3d_right, scaled_radius_r, (55, 255, 0), 130)  

            # ==== COMPUTE COMBINED GAZE DIRECTION FOR SCREEN MAPPING ====
            # Calculate individual gaze directions
            left_gaze_dir = iris_3d_left - sphere_world_l
            left_gaze_dir /= np.linalg.norm(left_gaze_dir)
            
            right_gaze_dir = iris_3d_right - sphere_world_r
            right_gaze_dir /= np.linalg.norm(right_gaze_dir)
            
            # Combine gaze directions (average)
            raw_combined_direction = (left_gaze_dir + right_gaze_dir) / 2
            raw_combined_direction /= np.linalg.norm(raw_combined_direction)

            # Update direction buffer for smoothing
            combined_gaze_directions.append(raw_combined_direction)

            # Smoothed direction
            avg_combined_direction = np.mean(combined_gaze_directions, axis=0)
            avg_combined_direction /= np.linalg.norm(avg_combined_direction)

            combined_dir = avg_combined_direction

            # ==== CONVERT GAZE TO SCREEN COORDINATES ====
            linear_screen_x, linear_screen_y, raw_yaw, raw_pitch = convert_gaze_to_screen_coordinates(
                avg_combined_direction, 
                calibration_offset_yaw, 
                calibration_offset_pitch
            )

            if nine_point_active and nine_point_targets:
                recent_gaze_angles.append((raw_yaw, raw_pitch))
                capture_nine_point_sample()

            if regression_ready():
                screen_x, screen_y = regression_screen_coordinates(raw_yaw, raw_pitch)
            else:
                screen_x, screen_y = linear_screen_x, linear_screen_y

            # Update mouse target
            if mouse_control_enabled:
                with mouse_lock:
                    mouse_target[0] = screen_x
                    mouse_target[1] = screen_y

            # ===== NEW: Write screen position to file =====
            write_screen_position(screen_x, screen_y)

            # Draw combined gaze ray for visualization
            combined_origin = (sphere_world_l + sphere_world_r) / 2
            combined_target = combined_origin + avg_combined_direction * gaze_length
            cv2.line(
                frame,
                (int(combined_origin[0]), int(combined_origin[1])),
                (int(combined_target[0]), int(combined_target[1])),
                (255, 255, 10), 3
            )

            # Center multiple lines of text
            mapping_text = "Mapping: 9-point regression"
            if regression_ready() and regression_loaded_from_disk:
                mapping_text = "Mapping: saved 9-point regression"
            elif not regression_ready():
                mapping_text = "Mapping: linear calibration"

            texts = [
                f"Screen: ({screen_x}, {screen_y})",
                mapping_text,
            ]

            if nine_point_active and nine_point_targets:
                target_number = min(nine_point_index + 1, len(nine_point_targets))
                texts.append(
                    f"9-point: target {target_number}/{len(nine_point_targets)} "
                    f"| {min(len(recent_gaze_angles), NINE_POINT_CAPTURE_FRAMES)}/{NINE_POINT_CAPTURE_FRAMES} stable frames"
                )
            elif regression_ready() and regression_fit_error_px is not None:
                texts.append(f"9-point RMSE: {regression_fit_error_px:.1f}px")

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            line_spacing = 30

            for i, text in enumerate(texts):
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                center_x = (w - text_width) // 2
                line_y = 30 + i * line_spacing
                if text.startswith("Screen:"):
                    color = (0, 255, 0)
                elif text.startswith("Mapping:"):
                    color = (255, 220, 0)
                else:
                    color = (0, 220, 255)
                cv2.putText(frame, text, (center_x, line_y), font, font_scale, color, thickness)

        # Draw all landmark points in white
        for idx, lm in enumerate(face_landmarks):
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 0, (255, 255, 255), -1)

        # Build 3D landmarks in your existing scale (x*w, y*h, z*w)
        lm = results.multi_face_landmarks[0].landmark
        landmarks3d = np.array([[p.x * w, p.y * h, p.z * w] for p in lm], dtype=float)

        render_debug_view_orbit(
            h, w,
            head_center3d=head_center,
            sphere_world_l=sphere_world_l if left_sphere_locked else None,
            scaled_radius_l=scaled_radius_l if left_sphere_locked else None,
            sphere_world_r=sphere_world_r if right_sphere_locked else None,
            scaled_radius_r=scaled_radius_r if right_sphere_locked else None,
            iris3d_l=iris_3d_left,
            iris3d_r=iris_3d_right,
            left_locked=left_sphere_locked,
            right_locked=right_sphere_locked,
            landmarks3d=landmarks3d,
            combined_dir=avg_combined_direction,
            gaze_len=5230,
            monitor_corners=monitor_corners,
            monitor_center=monitor_center_w,
            monitor_normal=monitor_normal_w,
            gaze_markers=gaze_markers
        )
    elif nine_point_active and nine_point_targets:
        recent_gaze_angles.clear()
        target_number = min(nine_point_index + 1, len(nine_point_targets))
        nine_point_status_text = f"Target {target_number}/{len(nine_point_targets)}: face lost, look back at the target."

    if nine_point_active:
        draw_nine_point_calibration_window()
    elif calibration_window_ready:
        close_nine_point_window()

    cv2.imshow("Integrated Eye Tracking", frame)

    # Handle keyboard input
    if is_hotkey_pressed('f7'):
        toggle_mouse_control()

    key = cv2.waitKeyEx(1)
    update_orbit_from_keypress(key)
    key_char = key & 0xFF if key != -1 else -1

    if key_char == 27 and nine_point_active:
        reset_nine_point_calibration(clear_regression=False)
        print("[9-Point Calibration] Cancelled.")
    elif key_char in (ord('m'), ord('M')):
        toggle_mouse_control()
    elif key_char in (ord('q'), ord('Q')):
        break
    elif key_char in (ord('9'),):
        if not (left_sphere_locked and right_sphere_locked):
            print("[9-Point Calibration] Lock both eye spheres with 'c' before starting 9-point calibration.")
        else:
            start_nine_point_calibration()
    elif key_char in (ord('c'), ord('C')) and face_landmarks is not None and not (left_sphere_locked and right_sphere_locked):
        current_nose_scale = compute_scale(nose_points_3d)
        # Lock LEFT eye
        left_sphere_local_offset = R_final.T @ (iris_3d_left - head_center)
        camera_dir_world = np.array([0, 0, 1])
        camera_dir_local = R_final.T @ camera_dir_world
        left_sphere_local_offset += base_radius * camera_dir_local
        left_calibration_nose_scale = current_nose_scale
        left_sphere_locked = True

        # Lock RIGHT eye
        right_sphere_local_offset = R_final.T @ (iris_3d_right - head_center)
        right_sphere_local_offset += base_radius * camera_dir_local  # use same camera_dir_local
        right_calibration_nose_scale = current_nose_scale
        right_sphere_locked = True

        # === Create 3D monitor plane at calibration ===
        # Compute instantaneous sphere positions at calibration distance (scale=1)
        sphere_world_l_calib = head_center + R_final @ left_sphere_local_offset
        sphere_world_r_calib = head_center + R_final @ right_sphere_local_offset

        # Estimate a forward gaze direction from the two eyes
        left_dir  = iris_3d_left  - sphere_world_l_calib
        right_dir = iris_3d_right - sphere_world_r_calib
        # Normalize (guard zero)
        if np.linalg.norm(left_dir)  > 1e-9: left_dir  /= np.linalg.norm(left_dir)
        if np.linalg.norm(right_dir) > 1e-9: right_dir /= np.linalg.norm(right_dir)
        forward_hint = (left_dir + right_dir) * 0.5
        if np.linalg.norm(forward_hint) > 1e-9:
            forward_hint /= np.linalg.norm(forward_hint)
        else:
            forward_hint = None  # fallback to head frame

        gaze_origin = (sphere_world_l_calib + sphere_world_r_calib) / 2
        gaze_dir = forward_hint  # already normalized

        monitor_corners, monitor_center_w, monitor_normal_w, units_per_cm = create_monitor_plane(
            head_center, R_final, face_landmarks, w, h,
            forward_hint=forward_hint,
            gaze_origin=gaze_origin,
            gaze_dir=gaze_dir
        )

        # Freeze the debug world's orbit pivot at the calibrated monitor center
        #global debug_world_frozen, orbit_pivot_frozen
        debug_world_frozen = True
        orbit_pivot_frozen = monitor_center_w.copy()
        combined_gaze_directions.clear()
        reset_nine_point_calibration(clear_regression=True)
        load_regression_calibration(announce=True)
        print("[Debug View] World pivot frozen at monitor center.")

        print(f"[Monitor] units_per_cm={units_per_cm:.3f}, center={monitor_center_w}, normal={monitor_normal_w}")


        print("[Both Spheres Locked] Eye sphere calibration complete.")
    elif key_char in (ord('s'), ord('S')) and left_sphere_locked and right_sphere_locked and avg_combined_direction is not None:
        # Screen calibration - user should look at center of screen when pressing 's'
        # Get current gaze direction
        left_gaze_dir = iris_3d_left - sphere_world_l
        left_gaze_dir /= np.linalg.norm(left_gaze_dir)
        right_gaze_dir = iris_3d_right - sphere_world_r
        right_gaze_dir /= np.linalg.norm(right_gaze_dir)
        current_combined_direction = (left_gaze_dir + right_gaze_dir) / 2
        current_combined_direction /= np.linalg.norm(current_combined_direction)
        
        # Calculate what the raw angles would be without calibration
        _, _, raw_yaw, raw_pitch = convert_gaze_to_screen_coordinates(
            current_combined_direction, 0, 0  # no calibration offset
        )
        
        # Set calibration offsets to center the gaze
        calibration_offset_yaw = 0 - raw_yaw
        calibration_offset_pitch = 0 - raw_pitch
        
        print(f"[Screen Calibrated] Offset Yaw: {calibration_offset_yaw:.2f}, Offset Pitch: {calibration_offset_pitch:.2f}")
    elif key_char in (ord('x'), ord('X')) and nose_points_3d is not None and head_center is not None and R_final is not None:
        # Drop a marker at the current gaze∩monitor point
        if (monitor_corners is not None and monitor_center_w is not None and monitor_normal_w is not None
            and left_sphere_locked and right_sphere_locked):
            # Recompute current eye-sphere positions (scale-aware)
            current_nose_scale = compute_scale(nose_points_3d)
            scale_ratio_l = current_nose_scale / left_calibration_nose_scale if left_calibration_nose_scale else 1.0
            scale_ratio_r = current_nose_scale / right_calibration_nose_scale if right_calibration_nose_scale else 1.0
            sphere_world_l_now = head_center + R_final @ (left_sphere_local_offset * scale_ratio_l)
            sphere_world_r_now = head_center + R_final @ (right_sphere_local_offset * scale_ratio_r)

            # Combined gaze direction (use smoothed if available; otherwise instantaneous)
            if 'avg_combined_direction' in locals() and avg_combined_direction is not None:
                D = _normalize(np.asarray(avg_combined_direction, dtype=float))
            else:
                lg = iris_3d_left  - sphere_world_l_now
                rg = iris_3d_right - sphere_world_r_now
                if np.linalg.norm(lg) < 1e-9 or np.linalg.norm(rg) < 1e-9:
                    print("[Marker] Gaze direction invalid; try again.")
                    D = None
                else:
                    lg /= np.linalg.norm(lg)
                    rg /= np.linalg.norm(rg)
                    D = _normalize(lg + rg)

            if D is not None:
                O = (sphere_world_l_now + sphere_world_r_now) * 0.5
                C = np.asarray(monitor_center_w, dtype=float)
                N = _normalize(np.asarray(monitor_normal_w, dtype=float))
                denom = float(np.dot(N, D))
                if abs(denom) < 1e-6:
                    print("[Marker] Gaze ray parallel to monitor; no marker.")
                else:
                    t = float(np.dot(N, (C - O)) / denom)
                    if t <= 0.0:
                        print("[Marker] Intersection behind/at eye; no marker.")
                    else:
                        P = O + t * D  # world-space intersection
                        # Map P to monitor local (a,b), then store if inside the quad
                        p0, p1, p2, p3 = [np.asarray(p, dtype=float) for p in monitor_corners]
                        u = p1 - p0
                        v = p3 - p0
                        u_len2 = float(np.dot(u, u))
                        v_len2 = float(np.dot(v, v))
                        if u_len2 > 1e-9 and v_len2 > 1e-9:
                            wv = P - p0
                            a = float(np.dot(wv, u) / u_len2)
                            b = float(np.dot(wv, v) / v_len2)
                            if 0.0 <= a <= 1.0 and 0.0 <= b <= 1.0:
                                gaze_markers.append((a, b))
                                print(f"[Marker] Added at a={a:.3f}, b={b:.3f}")
                            else:
                                print("[Marker] Gaze not on monitor; no marker.")
                        else:
                            print("[Marker] Monitor dimensions degenerate; no marker.")
        else:
            print("[Marker] Monitor/gaze not ready; complete center calibration first.")


cap.release()

close_nine_point_window()
cv2.destroyAllWindows()

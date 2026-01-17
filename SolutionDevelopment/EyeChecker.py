import cv2
import numpy as np
import mediapipe as mp
import time
import math
from scipy.spatial.transform import Rotation as Rscipy
import json
from datetime import datetime
from ultralytics import YOLO

# Screen setup
MONITOR_WIDTH, MONITOR_HEIGHT = 1920, 1080

# Sensitivity factors
VERTICAL_SENSITIVITY_FACTOR = 2.5
HORIZONTAL_SENSITIVITY_FACTOR = 1.5

# FPS settings - set to 20 FPS
TARGET_FPS = 20

# Base allowed deviation limits (in degrees) - будут адаптироваться
BASE_ALLOWED_HEAD_VERTICAL_UP = 1.5
BASE_ALLOWED_HEAD_VERTICAL_DOWN = 1.1
BASE_ALLOWED_HEAD_HORIZONTAL_ANGLE = 4

BASE_ALLOWED_GAZE_VERTICAL_UP = 1.5
BASE_ALLOWED_GAZE_VERTICAL_DOWN = 1.1
BASE_ALLOWED_GAZE_HORIZONTAL_ANGLE = 4

# Current allowed deviations (will be adapted dynamically)
ALLOWED_HEAD_VERTICAL_UP = BASE_ALLOWED_HEAD_VERTICAL_UP
ALLOWED_HEAD_VERTICAL_DOWN = BASE_ALLOWED_HEAD_VERTICAL_DOWN
ALLOWED_HEAD_HORIZONTAL_ANGLE = BASE_ALLOWED_HEAD_HORIZONTAL_ANGLE

ALLOWED_GAZE_VERTICAL_UP = BASE_ALLOWED_GAZE_VERTICAL_UP
ALLOWED_GAZE_VERTICAL_DOWN = BASE_ALLOWED_GAZE_VERTICAL_DOWN
ALLOWED_GAZE_HORIZONTAL_ANGLE = BASE_ALLOWED_GAZE_HORIZONTAL_ANGLE

# Warning thresholds (degrees beyond allowed)
WARNING_THRESHOLD = 8
DANGER_THRESHOLD = 15

# Smoothing factor for lerp
SMOOTHING_FACTOR = 0.35

# New parameters for tracking deviation duration
WARNING_FRAME_LIMIT = 30
DANGER_FRAME_LIMIT = 15
NORMAL_RESET_FRAMES = 60  # Frames required in normal zone to reset counters

# Frame counters for each parameter
warning_frame_counters = {
    "head_vertical": 0,
    "head_horizontal": 0,
    "gaze_vertical": 0,
    "gaze_horizontal": 0
}

danger_frame_counters = {
    "head_vertical": 0,
    "head_horizontal": 0,
    "gaze_vertical": 0,
    "gaze_horizontal": 0
}

normal_frame_counters = {
    "head_vertical": 0,
    "head_horizontal": 0,
    "gaze_vertical": 0,
    "gaze_horizontal": 0
}

warning_limit_exceeded = {
    "head_vertical": False,
    "head_horizontal": False,
    "gaze_vertical": False,
    "gaze_horizontal": False
}

danger_limit_exceeded = {
    "head_vertical": False,
    "head_horizontal": False,
    "gaze_vertical": False,
    "gaze_horizontal": False
}

# Adaptive calibration parameters
ADAPTIVE_CALIBRATION_ENABLED = True  # Включить адаптивную калибровку
FACE_SIZE_SMOOTHING_FACTOR = 0.1  # Сглаживание для размера лица
ADAPTATION_FACTOR_SMOOTHING = 0.2  # Сглаживание коэффициента адаптации

# Face size tracking
calibration_face_size = None  # Размер лица при калибровке
current_face_size = None  # Текущий размер лица
smoothed_face_size = None  # Сглаженный размер лица
adaptation_factor = 1.0  # Коэффициент адаптации (1.0 = нормальное расстояние)
smoothed_adaptation_factor = 1.0  # Сглаженный коэффициент адаптации

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Load YOLO11 model
print("Loading YOLO11 model...")
yolo_model = YOLO('yolo11s.pt')
yolo_verification_enabled = True  
print("YOLO11 model loaded successfully")

# Class names for YOLO (COCO dataset)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Open webcam
cap = cv2.VideoCapture(0)

# Set camera FPS to 20
cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

# Get actual frame dimensions
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Get actual camera FPS
actual_fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Camera FPS: {actual_fps} (target: {TARGET_FPS})")

if actual_fps != TARGET_FPS:
    print(f"Warning: Camera doesn't support {TARGET_FPS} FPS. Actual: {actual_fps}")
    print("Using software frame rate control...")

# Nose points for stable tracking
nose_indices = [4, 45, 275, 220, 440, 1, 5, 51, 281, 44, 274, 241, 
                461, 125, 354, 218, 438, 195, 167, 393, 165, 391, 3, 248]

# Face boundary points for face size calculation (примерные точки контура лица)
FACE_BOUNDARY_INDICES = [    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 
    361, 288, 397, 365, 379, 378, 400, 377, 
    152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 
    234, 127, 162, 21, 54, 103, 67, 109,]

# For coordinate stabilization
R_ref_nose = [None]

# Eye sphere lock status
left_sphere_locked = False
right_sphere_locked = False
left_sphere_local_offset = None
right_sphere_local_offset = None
left_calibration_nose_scale = None
right_calibration_nose_scale = None

# Calibration state
calibration_completed = False
calibration_vector = None

# Calibration for head direction
head_calibration_completed = False
head_calibration_vector = None

# Current values for analysis
current_gaze_vector = None
head_direction_vector = None

# Smoothed gaze vector
smoothed_gaze_vector = None

# FPS control variables
frame_interval = 1.0 / TARGET_FPS
last_frame_time = time.time()
frame_count = 0
fps = 0
last_fps_update = time.time()

# Store current state for each parameter (0 = normal, 1 = warning, 2 = danger)
current_state = {
    "head_vertical": 0,
    "head_horizontal": 0,
    "gaze_vertical": 0,
    "gaze_horizontal": 0
}

# YOLO frame skip counter
yolo_frame_skip = 0
YOLO_FRAME_INTERVAL = 5

# Variables for YOLO results
yolo_enabled = True
person_count = 0
phone_detected = False
gaze_tracking_enabled = True

def _normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v

def lerp_scalar(v1, v2, t):
    """Linear interpolation for scalar values"""
    return v1 + t * (v2 - v1)

def lerp_vector(v1, v2, t):
    if v1 is None:
        return v2
    if v2 is None:
        return v1
    return v1 + t * (v2 - v1)

def compute_scale(points_3d):
    n = len(points_3d)
    total = 0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(points_3d[i] - points_3d[j])
            total += dist
            count += 1
    return total / count if count > 0 else 1.0

def compute_face_size(face_landmarks, indices=FACE_BOUNDARY_INDICES):
    """Вычисляет размер лица на основе ключевых точек"""
    if face_landmarks is None or len(face_landmarks) == 0:
        return 0
    
    # Собираем координаты граничных точек лица
    points = []
    for idx in indices:
        if idx < len(face_landmarks):
            point = face_landmarks[idx]
            points.append((point.x * w, point.y * h))
    
    if len(points) < 3:
        return 0
    
    # Вычисляем площадь выпуклой оболочки точек лица
    # Используем bounding box как приближение размера лица
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    
    width = max(x_coords) - min(x_coords)
    height = max(y_coords) - min(y_coords)
    
    # Возвращаем площадь bounding box
    return width * height

def update_adaptation_factor():
    """Обновляет коэффициент адаптации на основе размера лица"""
    global adaptation_factor, smoothed_adaptation_factor, calibration_face_size, smoothed_face_size
    
    if calibration_face_size is None or current_face_size is None:
        adaptation_factor = 1.0
        return
    
    if calibration_face_size <= 0 or current_face_size <= 0:
        adaptation_factor = 1.0
        return
    
    # Вычисляем отношение размера лица к калибровочному
    # Если лицо ближе (больше), ratio > 1, если дальше (меньше), ratio < 1
    size_ratio = current_face_size / calibration_face_size
    
    # Применяем нелинейное преобразование: изменение на 50% размера лица = изменение на 15% угла
    # Формула: factor = 1 + (size_ratio - 1) * 0.3 (0.3 = 15%/50%)
    if size_ratio > 1:
        # Лицо ближе - увеличиваем допустимые углы
        factor_change = (size_ratio - 1) * 0.3
        adaptation_factor = 1 + factor_change
    else:
        # Лицо дальше - уменьшаем допустимые углы
        factor_change = (1 - size_ratio) * 0.3
        adaptation_factor = 1 - factor_change
    
    # Ограничиваем коэффициент адаптации разумными пределами
    adaptation_factor = max(0.5, min(2.0, adaptation_factor))
    
    # Сглаживаем коэффициент адаптации
    if smoothed_adaptation_factor is None:
        smoothed_adaptation_factor = adaptation_factor
    else:
        smoothed_adaptation_factor = lerp_scalar(smoothed_adaptation_factor, adaptation_factor, ADAPTATION_FACTOR_SMOOTHING)

def update_allowed_deviations():
    """Обновляет допустимые отклонения на основе коэффициента адаптации"""
    global ALLOWED_HEAD_VERTICAL_UP, ALLOWED_HEAD_VERTICAL_DOWN, ALLOWED_HEAD_HORIZONTAL_ANGLE
    global ALLOWED_GAZE_VERTICAL_UP, ALLOWED_GAZE_VERTICAL_DOWN, ALLOWED_GAZE_HORIZONTAL_ANGLE
    
    if not ADAPTIVE_CALIBRATION_ENABLED:
        return
    
    # Обновляем коэффициенты адаптации
    update_adaptation_factor()
    
    # Применяем адаптивный коэффициент к допустимым отклонениям
    ALLOWED_HEAD_VERTICAL_UP = BASE_ALLOWED_HEAD_VERTICAL_UP * smoothed_adaptation_factor
    ALLOWED_HEAD_VERTICAL_DOWN = BASE_ALLOWED_HEAD_VERTICAL_DOWN * smoothed_adaptation_factor
    ALLOWED_HEAD_HORIZONTAL_ANGLE = BASE_ALLOWED_HEAD_HORIZONTAL_ANGLE * smoothed_adaptation_factor
    
    ALLOWED_GAZE_VERTICAL_UP = BASE_ALLOWED_GAZE_VERTICAL_UP * smoothed_adaptation_factor
    ALLOWED_GAZE_VERTICAL_DOWN = BASE_ALLOWED_GAZE_VERTICAL_DOWN * smoothed_adaptation_factor
    ALLOWED_GAZE_HORIZONTAL_ANGLE = BASE_ALLOWED_GAZE_HORIZONTAL_ANGLE * smoothed_adaptation_factor

def compute_and_draw_coordinate_box(frame, face_landmarks, indices, ref_matrix_container, color=(0, 255, 0), size=80):
    points_3d = np.array([
        [face_landmarks[i].x * w, face_landmarks[i].y * h, face_landmarks[i].z * w]
        for i in indices
    ])

    center = np.mean(points_3d, axis=0)
    
    for i in indices:
        x, y = int(face_landmarks[i].x * w), int(face_landmarks[i].y * h)
        cv2.circle(frame, (x, y), 3, color, -1)

    centered = points_3d - center
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvecs = eigvecs[:, np.argsort(-eigvals)]
    
    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, 2] *= -1

    r = Rscipy.from_matrix(eigvecs)
    roll, pitch, yaw = r.as_euler('zyx', degrees=False)
    R_final = Rscipy.from_euler('zyx', [roll, pitch, yaw]).as_matrix()

    if ref_matrix_container[0] is None:
        ref_matrix_container[0] = R_final.copy()
    else:
        R_ref = ref_matrix_container[0]
        for i in range(3):
            if np.dot(R_final[:, i], R_ref[:, i]) < 0:
                R_final[:, i] *= -1

    return center, R_final, points_3d

def calculate_gaze_direction(iris_3d_left, iris_3d_right, sphere_world_l, sphere_world_r):
    left_gaze_dir = iris_3d_left - sphere_world_l
    right_gaze_dir = iris_3d_right - sphere_world_r
    
    left_gaze_dir[1] *= VERTICAL_SENSITIVITY_FACTOR
    right_gaze_dir[1] *= VERTICAL_SENSITIVITY_FACTOR
    
    left_gaze_dir[0] *= HORIZONTAL_SENSITIVITY_FACTOR
    right_gaze_dir[0] *= HORIZONTAL_SENSITIVITY_FACTOR
    
    left_gaze_dir = _normalize(left_gaze_dir)
    right_gaze_dir = _normalize(right_gaze_dir)
    
    combined_direction = (left_gaze_dir + right_gaze_dir) / 2
    combined_direction = _normalize(combined_direction)
    
    return combined_direction

def calculate_head_direction(R_final):
    head_forward = -R_final[:, 2]
    return _normalize(head_forward)

def calculate_angle_between_vectors(v1, v2):
    if v1 is None or v2 is None:
        return 0
    
    dot_product = np.dot(v1, v2)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    angle_rad = math.acos(dot_product)
    angle_deg = math.degrees(angle_rad)
    
    return angle_deg

def calculate_vertical_deviation(gaze_vector, calibration_vector):
    if gaze_vector is None or calibration_vector is None:
        return 0
    
    gaze_vertical = np.array([0, gaze_vector[1], gaze_vector[2]])
    calib_vertical = np.array([0, calibration_vector[1], calibration_vector[2]])
    
    gaze_vertical_norm = _normalize(gaze_vertical)
    calib_vertical_norm = _normalize(calib_vertical)
    
    dot_product = np.dot(gaze_vertical_norm, calib_vertical_norm)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    vertical_angle_rad = math.acos(dot_product)
    vertical_angle_deg = math.degrees(vertical_angle_rad)
    
    if gaze_vector[1] > calibration_vector[1]:
        return vertical_angle_deg
    else:
        return -vertical_angle_deg

def calculate_horizontal_deviation(gaze_vector, calibration_vector):
    if gaze_vector is None or calibration_vector is None:
        return 0
    
    gaze_horizontal = np.array([gaze_vector[0], 0, gaze_vector[2]])
    calib_horizontal = np.array([calibration_vector[0], 0, calibration_vector[2]])
    
    gaze_horizontal_norm = _normalize(gaze_horizontal)
    calib_horizontal_norm = _normalize(calib_horizontal)
    
    dot_product = np.dot(gaze_horizontal_norm, calib_horizontal_norm)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    horizontal_angle_rad = math.acos(dot_product)
    horizontal_angle_deg = math.degrees(horizontal_angle_rad)
    
    if gaze_vector[0] > calibration_vector[0]:
        return horizontal_angle_deg
    else:
        return -horizontal_angle_deg

def draw_face_boundary(frame, face_landmarks):
    """Draw face boundary points for visualization"""
    if face_landmarks is None:
        return
    
    for idx in FACE_BOUNDARY_INDICES:
        if idx < len(face_landmarks):
            x = int(face_landmarks[idx].x * w)
            y = int(face_landmarks[idx].y * h)
            cv2.circle(frame, (x, y), 2, (255, 0, 255), -1)

def get_deviation_level(deviation, allowed_up, allowed_down=None):
    if allowed_down is None:
        allowed_down = allowed_up
    
    if deviation >= 0:
        allowed = allowed_up
    else:
        allowed = allowed_down
    
    if abs(deviation) <= allowed:
        return 0
    
    excess = abs(deviation) - allowed
    
    if excess >= DANGER_THRESHOLD:
        return 2
    elif excess >= WARNING_THRESHOLD:
        return 1
    else:
        return 0

def analyze_deviations(head_vertical_dev, head_horizontal_dev, gaze_vertical_dev, gaze_horizontal_dev):
    global current_state, warning_frame_counters, danger_frame_counters, normal_frame_counters
    global warning_limit_exceeded, danger_limit_exceeded
    
    new_state = {
        "head_vertical": get_deviation_level(head_vertical_dev, ALLOWED_HEAD_VERTICAL_UP, ALLOWED_HEAD_VERTICAL_DOWN),
        "head_horizontal": get_deviation_level(head_horizontal_dev, ALLOWED_HEAD_HORIZONTAL_ANGLE),
        "gaze_vertical": get_deviation_level(gaze_vertical_dev, ALLOWED_GAZE_VERTICAL_UP, ALLOWED_GAZE_VERTICAL_DOWN),
        "gaze_horizontal": get_deviation_level(gaze_horizontal_dev, ALLOWED_GAZE_HORIZONTAL_ANGLE)
    }
    
    # Check each parameter
    for param in ["head_vertical", "head_horizontal", "gaze_vertical", "gaze_horizontal"]:
        old_level = current_state[param]
        new_level = new_state[param]
        
        # Update counters based on new state
        if new_level == 0:
            # Normal zone
            normal_frame_counters[param] += 1
            
            # Reset warning/danger counters only after NORMAL_RESET_FRAMES frames in normal zone
            if normal_frame_counters[param] >= NORMAL_RESET_FRAMES:
                warning_frame_counters[param] = 0
                danger_frame_counters[param] = 0
                warning_limit_exceeded[param] = False
                danger_limit_exceeded[param] = False
                normal_frame_counters[param] = 0
                
        elif new_level == 1:
            # Warning zone
            warning_frame_counters[param] += 1
            normal_frame_counters[param] = 0  # Reset normal counter
        elif new_level == 2:
            # Danger zone
            warning_frame_counters[param] += 1
            danger_frame_counters[param] += 1
            normal_frame_counters[param] = 0  # Reset normal counter
        
        # Update current state
        current_state[param] = new_level
    
    # Check for frame limit exceedances
    for param in ["head_vertical", "head_horizontal", "gaze_vertical", "gaze_horizontal"]:
        param_name = param.replace("_", " ").title()
        current_level = current_state[param]
        
        # Check warning limit
        if current_level == 1 and warning_frame_counters[param] >= WARNING_FRAME_LIMIT:
            if not warning_limit_exceeded[param]:
                warning_limit_exceeded[param] = True
                print(f"[Warning] {param_name}"
                      f"({warning_frame_counters[param]} frames, limit: {WARNING_FRAME_LIMIT})")
        
        # Check danger limit
        if current_level == 2 and danger_frame_counters[param] >= DANGER_FRAME_LIMIT:
            if not danger_limit_exceeded[param]:
                danger_limit_exceeded[param] = True
                print(f"[Danger] {param_name}"
                      f"({danger_frame_counters[param]} frames, limit: {DANGER_FRAME_LIMIT})")
        
        # Reset flags if state improved
        if current_level < 1 and warning_limit_exceeded[param]:
            warning_limit_exceeded[param] = False
        
        if current_level < 2 and danger_limit_exceeded[param]:
            danger_limit_exceeded[param] = False

def draw_info(frame, head_vertical_dev, head_horizontal_dev, gaze_vertical_dev, gaze_horizontal_dev):
    h_frame, w_frame = frame.shape[:2]
    
    status_bar = np.zeros((50, w_frame, 3), dtype=np.uint8)

    
    frame[:50, :] = status_bar
    
    # Обновляем адаптивные отклонения
    update_allowed_deviations()
    
    info_y = 80
    info_lines = [
        # f"System: REAL-TIME MONITORING {'(ADAPTIVE MODE)' if ADAPTIVE_CALIBRATION_ENABLED else ''}",
        # f"Camera FPS: {fps:.1f} (Target: {TARGET_FPS})",
        # f"Time: {datetime.now().strftime('%H:%M:%S')}",
        # f"YOLO: {'ENABLED' if yolo_enabled else 'DISABLED'}",
        # f"People: {person_count}, Phone: {'YES' if phone_detected else 'NO'}",
        # f"Gaze tracking: {'ENABLED' if gaze_tracking_enabled else 'DISABLED'}",
        # "",
        # f"--- ADAPTIVE PARAMETERS ---" if ADAPTIVE_CALIBRATION_ENABLED else f"--- FIXED PARAMETERS ---",
        # f"Face size: {current_face_size:.0f} px" if current_face_size else "Face size: N/A",
        # f"Calibration face size: {calibration_face_size:.0f} px" if calibration_face_size else "Calibration face size: N/A",
        # f"Adaptation factor: {smoothed_adaptation_factor:.2f}x" if ADAPTIVE_CALIBRATION_ENABLED else "Adaptation factor: 1.00x (disabled)",
        # "",
        # f"--- HEAD DEVIATIONS ---",
        # f"Vertical: {head_vertical_dev:+.1f}° (up: +{ALLOWED_HEAD_VERTICAL_UP:.1f}°, down: -{ALLOWED_HEAD_VERTICAL_DOWN:.1f}°)",
        # f"Horizontal: {head_horizontal_dev:+.1f}° (allowed: ±{ALLOWED_HEAD_HORIZONTAL_ANGLE:.1f}°)",
        # f"Warning counter: {warning_frame_counters['head_vertical']}/{warning_frame_counters['head_horizontal']} (limit: {WARNING_FRAME_LIMIT})",
        # f"Danger counter: {danger_frame_counters['head_vertical']}/{danger_frame_counters['head_horizontal']} (limit: {DANGER_FRAME_LIMIT})",
        # f"Normal counter: {normal_frame_counters['head_vertical']}/{normal_frame_counters['head_horizontal']} (reset: {NORMAL_RESET_FRAMES})",
        # "",
        f"--- GAZE DEVIATIONS ---",
        f"Vertical: {gaze_vertical_dev:+.1f}° (up: +{ALLOWED_GAZE_VERTICAL_UP:.1f}°, down: -{ALLOWED_GAZE_VERTICAL_DOWN:.1f}°)",
        f"Horizontal: {gaze_horizontal_dev:+.1f}° (allowed: ±{ALLOWED_GAZE_HORIZONTAL_ANGLE:.1f}°)",
        f"Warning counter: {warning_frame_counters['gaze_vertical']}/{warning_frame_counters['gaze_horizontal']} (limit: {WARNING_FRAME_LIMIT})",
        f"Danger counter: {danger_frame_counters['gaze_vertical']}/{danger_frame_counters['gaze_horizontal']} (limit: {DANGER_FRAME_LIMIT})",
        f"Normal counter: {normal_frame_counters['gaze_vertical']}/{normal_frame_counters['gaze_horizontal']} (reset: {NORMAL_RESET_FRAMES})",
    ]
    
    # Add settings information
    info_lines.extend([
        # "",
        # f"--- SETTINGS ---",
        # f"Vert. sensitivity: x{VERTICAL_SENSITIVITY_FACTOR}",
        # f"Horiz. sensitivity: x{HORIZONTAL_SENSITIVITY_FACTOR}",
        # f"Smoothing factor: {SMOOTHING_FACTOR}",
        # f"Adaptive calibration: {'ENABLED' if ADAPTIVE_CALIBRATION_ENABLED else 'DISABLED'}",
        # f"WARNING: >{WARNING_THRESHOLD}° beyond allowed",
        # f"DANGER: >{DANGER_THRESHOLD}° beyond allowed",
        # f"Warning frame limit: {WARNING_FRAME_LIMIT} frames ({WARNING_FRAME_LIMIT/TARGET_FPS:.1f} sec)",
        # f"Danger frame limit: {DANGER_FRAME_LIMIT} frames ({DANGER_FRAME_LIMIT/TARGET_FPS:.1f} sec)",
        # f"Normal reset frames: {NORMAL_RESET_FRAMES} frames ({NORMAL_RESET_FRAMES/TARGET_FPS:.1f} sec)"
    ])
    
    for i, line in enumerate(info_lines):
        y_pos = info_y + i * 25
        color = (255, 255, 255)
        
        if "Vertical:" in line or "Horizontal:" in line:
            if "HEAD" in line:
                if "Vertical:" in line:
                    dev = head_vertical_dev
                    allowed_up = ALLOWED_HEAD_VERTICAL_UP
                    allowed_down = ALLOWED_HEAD_VERTICAL_DOWN
                else:
                    dev = head_horizontal_dev
                    allowed_up = allowed_down = ALLOWED_HEAD_HORIZONTAL_ANGLE
            elif "GAZE" in line:
                if "Vertical:" in line:
                    dev = gaze_vertical_dev
                    allowed_up = ALLOWED_GAZE_VERTICAL_UP
                    allowed_down = ALLOWED_GAZE_VERTICAL_DOWN
                else:
                    dev = gaze_horizontal_dev
                    allowed_up = allowed_down = ALLOWED_GAZE_HORIZONTAL_ANGLE
            else:
                dev = 0
                allowed_up = allowed_down = 0
            
            if dev >= 0:
                allowed = allowed_up
            else:
                allowed = allowed_down
            
            excess = abs(dev) - allowed
            if excess >= DANGER_THRESHOLD:
                color = (0, 0, 255)
            elif excess >= WARNING_THRESHOLD:
                color = (0, 165, 255)
            elif abs(dev) > allowed:
                color = (0, 255, 255)
        
        elif "Warning counter:" in line:
            if "HEAD" in line:
                vert_counter = warning_frame_counters['head_vertical']
                horiz_counter = warning_frame_counters['head_horizontal']
            elif "GAZE" in line:
                vert_counter = warning_frame_counters['gaze_vertical']
                horiz_counter = warning_frame_counters['gaze_horizontal']
            else:
                vert_counter = horiz_counter = 0
            
            if vert_counter >= WARNING_FRAME_LIMIT or horiz_counter >= WARNING_FRAME_LIMIT:
                color = (0, 165, 255)
        
        elif "Danger counter:" in line:
            if "HEAD" in line:
                vert_counter = danger_frame_counters['head_vertical']
                horiz_counter = danger_frame_counters['head_horizontal']
            elif "GAZE" in line:
                vert_counter = danger_frame_counters['gaze_vertical']
                horiz_counter = danger_frame_counters['gaze_horizontal']
            else:
                vert_counter = horiz_counter = 0
            
            if vert_counter >= DANGER_FRAME_LIMIT or horiz_counter >= DANGER_FRAME_LIMIT:
                color = (0, 0, 255)
        
        cv2.putText(frame, line, (20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def run_yolo_detection(frame):
    global person_count, phone_detected, gaze_tracking_enabled, yolo_verification_enabled  # Добавьте yolo_verification_enabled
    
    results = yolo_model(frame, verbose=False)
    
    person_count = 0
    phone_detected = False
    
    if results and len(results) > 0:
        result = results[0]
        
        if result.boxes is not None:
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            for class_id in class_ids:
                if class_id == 0:
                    person_count += 1
                elif class_id == 67:
                    phone_detected = True
            
            # Изменённая логика: если проверка YOLO отключена, всегда включаем трекинг
            gaze_tracking_enabled = (not yolo_verification_enabled) or (person_count == 1 and not phone_detected)
            
            if result.boxes is not None and len(result.boxes) > 0:
                for i, box in enumerate(result.boxes):
                    if box.conf[0] > 0.5:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        confidence = float(box.conf[0].cpu().numpy())
                        
                        if class_id < len(COCO_CLASSES):
                            class_name = COCO_CLASSES[class_id]
                        else:
                            class_name = f"Class {class_id}"
                        
                        if class_id == 0:
                            color = (0, 255, 0)
                        elif class_id == 67:
                            color = (0, 0, 255)
                        else:
                            color = (255, 0, 0)
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        label = f"{class_name}: {confidence:.2f}"
                        cv2.putText(frame, label, (x1, y1 - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return gaze_tracking_enabled

print("=== Enhanced Gaze and Head Direction Monitoring System ===")
print("=== WITH ADAPTIVE CALIBRATION ===")
print("Instructions:")
print("1. Press 'c' to calibrate (look straight at monitor center)")
print("2. System will analyze deviations in real-time")
print(f"3. FPS limited to {TARGET_FPS} for stability")
print("4. Press 'q' to exit")
print(f"5. Vert. sensitivity: {VERTICAL_SENSITIVITY_FACTOR}x")
print(f"6. Horiz. sensitivity: {HORIZONTAL_SENSITIVITY_FACTOR}x")
print(f"7. BASE allowed head deviations: up: +{BASE_ALLOWED_HEAD_VERTICAL_UP}°, down: -{BASE_ALLOWED_HEAD_VERTICAL_DOWN}° vert., ±{BASE_ALLOWED_HEAD_HORIZONTAL_ANGLE}° horiz.")
print(f"8. BASE allowed gaze deviations: up: +{BASE_ALLOWED_GAZE_VERTICAL_UP}°, down: -{BASE_ALLOWED_GAZE_VERTICAL_DOWN}° vert., ±{BASE_ALLOWED_GAZE_HORIZONTAL_ANGLE}° horiz.")
print(f"9. Adaptive calibration: {'ENABLED' if ADAPTIVE_CALIBRATION_ENABLED else 'DISABLED'}")
print("10. System will automatically adjust thresholds based on face distance from camera")
print("11. When face gets closer (50% larger), allowed angles increase by ~15%")
print("12. When face moves away (50% smaller), allowed angles decrease by ~15%")
print(f"14. YOLO monitoring: Enabled (checks every {YOLO_FRAME_INTERVAL} frames)")
print("15. Gaze tracking only when YOLO detects exactly 1 person and no phone")
print(f"16. WARNING: {WARNING_THRESHOLD}°-{DANGER_THRESHOLD}° beyond allowed")
print(f"17. DANGER: more than {DANGER_THRESHOLD}° beyond allowed")
print(f"18. Smoothing factor: {SMOOTHING_FACTOR} (for gaze vector)")
print(f"19. Warning frame limit: {WARNING_FRAME_LIMIT} frames ({WARNING_FRAME_LIMIT/TARGET_FPS:.1f} seconds)")
print(f"20. Danger frame limit: {DANGER_FRAME_LIMIT} frames ({DANGER_FRAME_LIMIT/TARGET_FPS:.1f} seconds)")
print(f"21. Normal reset frames: {NORMAL_RESET_FRAMES} frames ({NORMAL_RESET_FRAMES/TARGET_FPS:.1f} seconds)")
print("\nStarting monitoring system...")
print("System will output messages only when frame limits are exceeded.")
print("Counters reset only after staying in normal zone for 60 frames.")

# Initialize deviation variables
head_vertical_deviation = 0
head_horizontal_deviation = 0
gaze_vertical_deviation = 0
gaze_horizontal_deviation = 0

while cap.isOpened():
    current_time = time.time()
    
    if current_time - last_frame_time < frame_interval:
        time_to_wait = frame_interval - (current_time - last_frame_time)
        if time_to_wait > 0:
            time.sleep(time_to_wait)
        continue
    
    last_frame_time = current_time
    
    frame_count += 1
    if current_time - last_fps_update >= 1.0:
        fps = frame_count / (current_time - last_fps_update)
        frame_count = 0
        last_fps_update = current_time
    
    ret, frame = cap.read()
    if not ret:
        break
    
    yolo_frame_skip += 1
    
    if yolo_enabled and yolo_frame_skip >= YOLO_FRAME_INTERVAL:
        gaze_tracking_enabled = run_yolo_detection(frame)
        yolo_frame_skip = 0
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    current_gaze_vector = None
    head_direction_vector = None
    
    if gaze_tracking_enabled:
        results = face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark
            
            # Вычисляем текущий размер лица
            current_face_size = compute_face_size(face_landmarks)
            
            # Сглаживаем размер лица
            if smoothed_face_size is None:
                smoothed_face_size = current_face_size
            else:
                smoothed_face_size = lerp_scalar(smoothed_face_size, current_face_size, FACE_SIZE_SMOOTHING_FACTOR)
            
            left_iris_idx = 468
            right_iris_idx = 473
            left_iris = face_landmarks[left_iris_idx]
            right_iris = face_landmarks[right_iris_idx]
            
            head_center, R_final, nose_points_3d = compute_and_draw_coordinate_box(
                frame, face_landmarks, nose_indices, R_ref_nose, color=(0, 255, 0), size=80
            )
            
            head_direction_vector = calculate_head_direction(R_final)
            
            iris_3d_left = np.array([left_iris.x * w, left_iris.y * h, left_iris.z * w])
            iris_3d_right = np.array([right_iris.x * w, right_iris.y * h, right_iris.z * w])
            
            x_iris_l, y_iris_l = int(left_iris.x * w), int(left_iris.y * h)
            x_iris_r, y_iris_r = int(right_iris.x * w), int(right_iris.y * h)
            cv2.circle(frame, (x_iris_l, y_iris_l), 8, (25, 255, 255), -1)
            cv2.circle(frame, (x_iris_r, y_iris_r), 8, (25, 255, 255), -1)
            
            if left_sphere_locked and right_sphere_locked:
                current_nose_scale = compute_scale(nose_points_3d)
                
                scale_ratio_l = current_nose_scale / left_calibration_nose_scale if left_calibration_nose_scale else 1.0
                scaled_offset_l = left_sphere_local_offset * scale_ratio_l
                sphere_world_l = head_center + R_final @ scaled_offset_l
                
                scale_ratio_r = current_nose_scale / right_calibration_nose_scale if right_calibration_nose_scale else 1.0
                scaled_offset_r = right_sphere_local_offset * scale_ratio_r
                sphere_world_r = head_center + R_final @ scaled_offset_r
                
                current_gaze_vector = calculate_gaze_direction(
                    iris_3d_left, iris_3d_right, sphere_world_l, sphere_world_r
                )
                
                if current_gaze_vector is not None:
                    if smoothed_gaze_vector is None:
                        smoothed_gaze_vector = current_gaze_vector.copy()
                    else:
                        smoothed_gaze_vector = lerp_vector(smoothed_gaze_vector, current_gaze_vector, SMOOTHING_FACTOR)
                
                gaze_vector_for_display = smoothed_gaze_vector if smoothed_gaze_vector is not None else current_gaze_vector
                
                x_sphere_l, y_sphere_l = int(sphere_world_l[0]), int(sphere_world_l[1])
                x_sphere_r, y_sphere_r = int(sphere_world_r[0]), int(sphere_world_r[1])
                
                cv2.circle(frame, (x_sphere_l, y_sphere_l), 15, (255, 255, 25), 2)
                cv2.circle(frame, (x_sphere_r, y_sphere_r), 15, (25, 255, 255), 2)
                
                if gaze_vector_for_display is not None:
                    gaze_end_l = sphere_world_l + (gaze_vector_for_display * 100)
                    gaze_end_r = sphere_world_r + (gaze_vector_for_display * 100)
                    
                    cv2.line(frame, (x_sphere_l, y_sphere_l), 
                            (int(gaze_end_l[0]), int(gaze_end_l[1])), (55, 255, 0), 2)
                    cv2.line(frame, (x_sphere_r, y_sphere_r), 
                            (int(gaze_end_r[0]), int(gaze_end_r[1])), (55, 255, 0), 2)
                    
                    eye_center = ((sphere_world_l + sphere_world_r) / 2)[:2].astype(int)
                    combined_gaze_end = eye_center + (gaze_vector_for_display[:2] * 150)
                    cv2.arrowedLine(frame, tuple(eye_center), tuple(combined_gaze_end.astype(int)),
                                  (0, 255, 255), 3, tipLength=0.3)
                
                if calibration_completed and head_calibration_completed and smoothed_gaze_vector is not None and head_direction_vector is not None:
                    head_vertical_deviation = calculate_vertical_deviation(head_direction_vector, head_calibration_vector)
                    head_horizontal_deviation = calculate_horizontal_deviation(head_direction_vector, head_calibration_vector)
                    
                    gaze_vertical_deviation = calculate_vertical_deviation(smoothed_gaze_vector, calibration_vector)
                    gaze_horizontal_deviation = calculate_horizontal_deviation(smoothed_gaze_vector, calibration_vector)
                    
                    analyze_deviations(head_vertical_deviation, head_horizontal_deviation, 
                                     gaze_vertical_deviation, gaze_horizontal_deviation)
            
            # Draw face boundary points for visualization
            draw_face_boundary(frame, face_landmarks)
    
    if not gaze_tracking_enabled and yolo_enabled:
        warning_text = f"GAZE TRACKING DISABLED - People: {person_count}, Phone: {'DETECTED' if phone_detected else 'NOT DETECTED'}"
        text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        cv2.putText(frame, warning_text, (text_x, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    draw_info(frame, head_vertical_deviation, head_horizontal_deviation, 
             gaze_vertical_deviation, gaze_horizontal_deviation)
    
    cv2.imshow("Enhanced Gaze and Head Direction Monitoring", frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    
    elif key == ord('c'):
        # Single-button calibration: calibrate eye spheres, gaze center, head direction
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark
            
            # First, calibrate eye spheres if not already done
            if not (left_sphere_locked and right_sphere_locked):
                print("[Calibration] Calibrating eye spheres...")
                current_nose_scale = compute_scale(nose_points_3d)
                
                # Calibrate left eye
                left_sphere_local_offset = R_final.T @ (iris_3d_left - head_center)
                camera_dir_world = np.array([0, 0, 1])
                camera_dir_local = R_final.T @ camera_dir_world
                left_sphere_local_offset += 20 * camera_dir_local
                left_calibration_nose_scale = current_nose_scale
                left_sphere_locked = True
                
                # Calibrate right eye
                right_sphere_local_offset = R_final.T @ (iris_3d_right - head_center)
                right_sphere_local_offset += 20 * camera_dir_local
                right_calibration_nose_scale = current_nose_scale
                right_sphere_locked = True
                
                print("[Calibration] Eye spheres calibrated")
                time.sleep(0.5)
            
            # Calibrate gaze center (assuming user is looking at monitor center)
            current_nose_scale = compute_scale(nose_points_3d)
            scale_ratio_l = current_nose_scale / left_calibration_nose_scale if left_calibration_nose_scale else 1.0
            scale_ratio_r = current_nose_scale / right_calibration_nose_scale if right_calibration_nose_scale else 1.0
            sphere_world_l_calib = head_center + R_final @ (left_sphere_local_offset * scale_ratio_l)
            sphere_world_r_calib = head_center + R_final @ (right_sphere_local_offset * scale_ratio_r)
            
            # Calculate gaze direction at calibration moment
            left_gaze_dir = iris_3d_left - sphere_world_l_calib
            right_gaze_dir = iris_3d_right - sphere_world_r_calib
            
            # Apply sensitivity factors during calibration
            left_gaze_dir[0] *= HORIZONTAL_SENSITIVITY_FACTOR
            left_gaze_dir[1] *= VERTICAL_SENSITIVITY_FACTOR
            
            right_gaze_dir[0] *= HORIZONTAL_SENSITIVITY_FACTOR
            right_gaze_dir[1] *= VERTICAL_SENSITIVITY_FACTOR
            
            left_gaze_dir = _normalize(left_gaze_dir)
            right_gaze_dir = _normalize(right_gaze_dir)
            
            calibration_vector = (left_gaze_dir + right_gaze_dir) / 2
            calibration_vector = _normalize(calibration_vector)
            calibration_completed = True
            
            # Calibrate head direction
            head_calibration_vector = head_direction_vector.copy()
            head_calibration_completed = True
            
            # Calibrate face size
            calibration_face_size = compute_face_size(face_landmarks)
            smoothed_face_size = calibration_face_size
            print(f"[Calibration] Face size calibrated: {calibration_face_size:.0f} px")
            
            # Сбрасываем сглаженный вектор при калибровке
            smoothed_gaze_vector = calibration_vector.copy()
            
            # Reset adaptation factor
            adaptation_factor = 1.0
            smoothed_adaptation_factor = 1.0
            
            # Reset all states to normal after calibration
            for param in current_state:
                current_state[param] = 0
            
            # Сбрасываем все счетчики при калибровке
            for param in warning_frame_counters:
                warning_frame_counters[param] = 0
                danger_frame_counters[param] = 0
                normal_frame_counters[param] = 0
                warning_limit_exceeded[param] = False
                danger_limit_exceeded[param] = False
            
            print(f"[Calibration] Complete!")
            print(f"[Calibration] Gaze vector: {calibration_vector}")
            print(f"[Calibration] Head vector: {head_calibration_vector}")
            print(f"[Calibration] Face size: {calibration_face_size:.0f} px")
            print(f"[Calibration] Vert. sensitivity: {VERTICAL_SENSITIVITY_FACTOR}x")
            print(f"[Calibration] Horiz. sensitivity: {HORIZONTAL_SENSITIVITY_FACTOR}x")
            print(f"[Calibration] Smoothing factor: {SMOOTHING_FACTOR}")
            print(f"[Calibration] Adaptive calibration: {'ENABLED' if ADAPTIVE_CALIBRATION_ENABLED else 'DISABLED'}")
            print(f"[Calibration] Warning frame limit: {WARNING_FRAME_LIMIT} frames")
            print(f"[Calibration] Danger frame limit: {DANGER_FRAME_LIMIT} frames")
            print(f"[Calibration] Normal reset frames: {NORMAL_RESET_FRAMES} frames")
            print("[Calibration] Monitoring started. System will output messages only when frame limits are exceeded.")
    
    elif key == ord('a'):  # Toggle adaptive calibration
        ADAPTIVE_CALIBRATION_ENABLED = not ADAPTIVE_CALIBRATION_ENABLED
        status = "ENABLED" if ADAPTIVE_CALIBRATION_ENABLED else "DISABLED"
        print(f"[Settings] Adaptive calibration {status}")
        
        if not ADAPTIVE_CALIBRATION_ENABLED:
            # Reset to base values when disabling adaptive mode
            ALLOWED_HEAD_VERTICAL_UP = BASE_ALLOWED_HEAD_VERTICAL_UP
            ALLOWED_HEAD_VERTICAL_DOWN = BASE_ALLOWED_HEAD_VERTICAL_DOWN
            ALLOWED_HEAD_HORIZONTAL_ANGLE = BASE_ALLOWED_HEAD_HORIZONTAL_ANGLE
            
            ALLOWED_GAZE_VERTICAL_UP = BASE_ALLOWED_GAZE_VERTICAL_UP
            ALLOWED_GAZE_VERTICAL_DOWN = BASE_ALLOWED_GAZE_VERTICAL_DOWN
            ALLOWED_GAZE_HORIZONTAL_ANGLE = BASE_ALLOWED_GAZE_HORIZONTAL_ANGLE
    
    elif key == ord('y'):  # Toggle YOLO verification
        yolo_verification_enabled = not yolo_verification_enabled
        status = "ENABLED" if yolo_verification_enabled else "DISABLED"
        print(f"[Settings] YOLO verification {status}")
        print(f"[Settings] Gaze tracking will now work {'only with 1 person and no phone' if yolo_verification_enabled else 'regardless of YOLO detection'}")

cap.release()
cv2.destroyAllWindows()
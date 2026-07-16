import cv2
import numpy as np
import mediapipe as mp
import time
import math
from scipy.spatial.transform import Rotation as Rscipy
import base64
import logging
from typing import Dict, Optional, Any
from datetime import datetime
from ultralytics import YOLO

from core.config import settings

logger = logging.getLogger(__name__)

class ProctorService:
    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = None
        
        self.yolo_model = None
        self.yolo_verification_enabled = True
        
        self.MONITOR_WIDTH, self.MONITOR_HEIGHT = 1920, 1080
        self.VERTICAL_SENSITIVITY_FACTOR = 2.5
        self.HORIZONTAL_SENSITIVITY_FACTOR = 1.5
        self.TARGET_FPS = 20
        
        self.BASE_ALLOWED_HEAD_VERTICAL_UP = 1.5
        self.BASE_ALLOWED_HEAD_VERTICAL_DOWN = 1.1
        self.BASE_ALLOWED_HEAD_HORIZONTAL_ANGLE = 4
        self.BASE_ALLOWED_GAZE_VERTICAL_UP = 1.5
        self.BASE_ALLOWED_GAZE_VERTICAL_DOWN = 1.1
        self.BASE_ALLOWED_GAZE_HORIZONTAL_ANGLE = 4
        
        self.ALLOWED_HEAD_VERTICAL_UP = self.BASE_ALLOWED_HEAD_VERTICAL_UP
        self.ALLOWED_HEAD_VERTICAL_DOWN = self.BASE_ALLOWED_HEAD_VERTICAL_DOWN
        self.ALLOWED_HEAD_HORIZONTAL_ANGLE = self.BASE_ALLOWED_HEAD_HORIZONTAL_ANGLE
        self.ALLOWED_GAZE_VERTICAL_UP = self.BASE_ALLOWED_GAZE_VERTICAL_UP
        self.ALLOWED_GAZE_VERTICAL_DOWN = self.BASE_ALLOWED_GAZE_VERTICAL_DOWN
        self.ALLOWED_GAZE_HORIZONTAL_ANGLE = self.BASE_ALLOWED_GAZE_HORIZONTAL_ANGLE
        
        self.WARNING_THRESHOLD = 8
        self.DANGER_THRESHOLD = 15
        
        self.SMOOTHING_FACTOR = 0.35
        
        self.WARNING_FRAME_LIMIT = 30  # 1.5 секунды при 20 FPS
        self.DANGER_FRAME_LIMIT = 15   # 0.75 секунды при 20 FPS
        self.NORMAL_RESET_FRAMES = 60  # 3 секунды при 20 FPS
        
        self.ADAPTIVE_CALIBRATION_ENABLED = True
        self.FACE_SIZE_SMOOTHING_FACTOR = 0.1
        self.ADAPTATION_FACTOR_SMOOTHING = 0.2
        
        self.nose_indices = [4, 45, 275, 220, 440, 1, 5, 51, 281, 44, 274, 241, 
                            461, 125, 354, 218, 438, 195, 167, 393, 165, 391, 3, 248]
        
        self.FACE_BOUNDARY_INDICES = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 
            361, 288, 397, 365, 379, 378, 400, 377, 
            152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 
            234, 127, 162, 21, 54, 103, 67, 109
        ]
        
        self.COCO_CLASSES = [
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
        
        self.YOLO_FRAME_INTERVAL = 5
        
        logger.info("ProctorService инициализирован")
    
    async def initialize_models(self):
        try:
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            logger.info("Загрузка YOLO модели...")
            self.yolo_model = YOLO(settings.YOLO_MODEL_PATH)
            logger.info("Модели загружены успешно")
            
        except Exception as e:
            logger.error(f"Ошибка загрузки моделей: {e}")
            raise
    
    def create_session(self, user_id: int) -> str:
        import uuid
        session_id = str(uuid.uuid4())
        
        self.sessions[session_id] = {
            "id": session_id,
            "user_id": user_id,
            "created_at": datetime.now(),
            "calibration_completed": False,
            "calibration_vector": None,
            "head_calibration_completed": False,
            "head_calibration_vector": None,
            
            "left_sphere_locked": False,
            "right_sphere_locked": False,
            "left_sphere_local_offset": None,
            "right_sphere_local_offset": None,
            "left_calibration_nose_scale": None,
            "right_calibration_nose_scale": None,
            
            "calibration_face_size": None,
            "current_face_size": None,
            "smoothed_face_size": None,
            "adaptation_factor": 1.0,
            "smoothed_adaptation_factor": 1.0,
            
            "current_state": {
                "head_vertical": 0,
                "head_horizontal": 0,
                "gaze_vertical": 0,
                "gaze_horizontal": 0
            },
            "warning_frame_counters": {
                "head_vertical": 0,
                "head_horizontal": 0,
                "gaze_vertical": 0,
                "gaze_horizontal": 0
            },
            "danger_frame_counters": {
                "head_vertical": 0,
                "head_horizontal": 0,
                "gaze_vertical": 0,
                "gaze_horizontal": 0
            },
            "normal_frame_counters": {
                "head_vertical": 0,
                "head_horizontal": 0,
                "gaze_vertical": 0,
                "gaze_horizontal": 0
            },
            "warning_limit_exceeded": {
                "head_vertical": False,
                "head_horizontal": False,
                "gaze_vertical": False,
                "gaze_horizontal": False
            },
            "danger_limit_exceeded": {
                "head_vertical": False,
                "head_horizontal": False,
                "gaze_vertical": False,
                "gaze_horizontal": False
            },
            
            "R_ref_nose": [None],
            "smoothed_gaze_vector": None,
            "last_gaze_vector": None,
            "last_head_vector": None,
            
            "frame_count": 0,
            "last_frame_time": time.time(),
            "fps": 0,
            "last_fps_update": time.time(),
            "yolo_frame_skip": 0,
            
            "person_count": 0,
            "phone_detected": False,
            "gaze_tracking_enabled": True,
            
            "warnings": [],
            "alerts": [],
            "last_status_data": None,
            "graphics_frame": None,
            
            "need_calibration": False,
            "calibration_iris_3d_left": None,
            "calibration_iris_3d_right": None,
            "calibration_head_center": None,
            "calibration_R_final": None,
            "calibration_nose_points_3d": None
        }
        
        logger.info(f"Создана сессия {session_id} для пользователя {user_id}")
        return session_id
    
    def _normalize(self, v):
        v = np.asarray(v, dtype=float)
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else v
    
    def _lerp_vector(self, v1, v2, t):
        if v1 is None:
            return v2
        if v2 is None:
            return v1
        return v1 + t * (v2 - v1)
    
    def _lerp_scalar(self, v1, v2, t):
        return v1 + t * (v2 - v1)
    
    def _compute_scale(self, points_3d):
        n = len(points_3d)
        total = 0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(points_3d[i] - points_3d[j])
                total += dist
                count += 1
        return total / count if count > 0 else 1.0
    
    def _compute_face_size(self, face_landmarks, w, h):
        if face_landmarks is None or len(face_landmarks) == 0:
            return 0
        
        points = []
        for idx in self.FACE_BOUNDARY_INDICES:
            if idx < len(face_landmarks):
                point = face_landmarks[idx]
                points.append((point.x * w, point.y * h))
        
        if len(points) < 3:
            return 0
        
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        width = max(x_coords) - min(x_coords)
        height = max(y_coords) - min(y_coords)
        
        return width * height
    
    def _update_adaptation_factor(self, session):
        if session["calibration_face_size"] is None or session["current_face_size"] is None:
            session["adaptation_factor"] = 1.0
            return
        
        if session["calibration_face_size"] <= 0 or session["current_face_size"] <= 0:
            session["adaptation_factor"] = 1.0
            return
        
        size_ratio = session["current_face_size"] / session["calibration_face_size"]
        
        if size_ratio > 1:
            factor_change = (size_ratio - 1) * 0.3
            session["adaptation_factor"] = 1 + factor_change
        else:
            factor_change = (1 - size_ratio) * 0.3
            session["adaptation_factor"] = 1 - factor_change
        
        session["adaptation_factor"] = max(0.5, min(2.0, session["adaptation_factor"]))
        
        if session["smoothed_adaptation_factor"] is None:
            session["smoothed_adaptation_factor"] = session["adaptation_factor"]
        else:
            session["smoothed_adaptation_factor"] = self._lerp_scalar(
                session["smoothed_adaptation_factor"], 
                session["adaptation_factor"], 
                self.ADAPTATION_FACTOR_SMOOTHING
            )
    
    def _update_allowed_deviations(self, session):
        if not self.ADAPTIVE_CALIBRATION_ENABLED:
            return
        
        self._update_adaptation_factor(session)
        
        factor = session["smoothed_adaptation_factor"]
        
        self.ALLOWED_HEAD_VERTICAL_UP = self.BASE_ALLOWED_HEAD_VERTICAL_UP * factor
        self.ALLOWED_HEAD_VERTICAL_DOWN = self.BASE_ALLOWED_HEAD_VERTICAL_DOWN * factor
        self.ALLOWED_HEAD_HORIZONTAL_ANGLE = self.BASE_ALLOWED_HEAD_HORIZONTAL_ANGLE * factor
        
        self.ALLOWED_GAZE_VERTICAL_UP = self.BASE_ALLOWED_GAZE_VERTICAL_UP * factor
        self.ALLOWED_GAZE_VERTICAL_DOWN = self.BASE_ALLOWED_GAZE_VERTICAL_DOWN * factor
        self.ALLOWED_GAZE_HORIZONTAL_ANGLE = self.BASE_ALLOWED_GAZE_HORIZONTAL_ANGLE * factor
    
    def _compute_and_draw_coordinate_box(self, frame, face_landmarks, indices, ref_matrix_container, w, h, color=(0, 255, 0)):
        points_3d = np.array([
            [face_landmarks[i].x * w, face_landmarks[i].y * h, face_landmarks[i].z * w]
            for i in indices
        ])

        center = np.mean(points_3d, axis=0)
        
        if settings.DEBUG:
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
    
    def _calculate_gaze_direction(self, iris_3d_left, iris_3d_right, sphere_world_l, sphere_world_r):
        left_gaze_dir = iris_3d_left - sphere_world_l
        right_gaze_dir = iris_3d_right - sphere_world_r
        
        left_gaze_dir[1] *= self.VERTICAL_SENSITIVITY_FACTOR
        right_gaze_dir[1] *= self.VERTICAL_SENSITIVITY_FACTOR
        
        left_gaze_dir[0] *= self.HORIZONTAL_SENSITIVITY_FACTOR
        right_gaze_dir[0] *= self.HORIZONTAL_SENSITIVITY_FACTOR
        
        left_gaze_dir = self._normalize(left_gaze_dir)
        right_gaze_dir = self._normalize(right_gaze_dir)
        
        combined_direction = (left_gaze_dir + right_gaze_dir) / 2
        return self._normalize(combined_direction)
    
    def _calculate_head_direction(self, R_final):
        head_forward = -R_final[:, 2]
        return self._normalize(head_forward)
    
    def _calculate_vertical_deviation(self, gaze_vector, calibration_vector):
        if gaze_vector is None or calibration_vector is None:
            return 0
        
        gaze_vertical = np.array([0, gaze_vector[1], gaze_vector[2]])
        calib_vertical = np.array([0, calibration_vector[1], calibration_vector[2]])
        
        gaze_vertical_norm = self._normalize(gaze_vertical)
        calib_vertical_norm = self._normalize(calib_vertical)
        
        dot_product = np.dot(gaze_vertical_norm, calib_vertical_norm)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        vertical_angle_rad = math.acos(dot_product)
        vertical_angle_deg = math.degrees(vertical_angle_rad)
        
        if gaze_vector[1] > calibration_vector[1]:
            return vertical_angle_deg
        else:
            return -vertical_angle_deg
    
    def _calculate_horizontal_deviation(self, gaze_vector, calibration_vector):
        if gaze_vector is None or calibration_vector is None:
            return 0
        
        gaze_horizontal = np.array([gaze_vector[0], 0, gaze_vector[2]])
        calib_horizontal = np.array([calibration_vector[0], 0, calibration_vector[2]])
        
        gaze_horizontal_norm = self._normalize(gaze_horizontal)
        calib_horizontal_norm = self._normalize(calib_horizontal)
        
        dot_product = np.dot(gaze_horizontal_norm, calib_horizontal_norm)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        horizontal_angle_rad = math.acos(dot_product)
        horizontal_angle_deg = math.degrees(horizontal_angle_rad)
        
        if gaze_vector[0] > calibration_vector[0]:
            return horizontal_angle_deg
        else:
            return -horizontal_angle_deg
    
    def _get_deviation_level(self, deviation, allowed_up, allowed_down=None):
        if allowed_down is None:
            allowed_down = allowed_up
        
        if deviation >= 0:
            allowed = allowed_up
        else:
            allowed = allowed_down
        
        if abs(deviation) <= allowed:
            return 0
        
        excess = abs(deviation) - allowed
        
        if excess >= self.DANGER_THRESHOLD:
            return 2
        elif excess >= self.WARNING_THRESHOLD:
            return 1
        else:
            return 0
    
    def _analyze_deviations(self, session, head_vertical_dev, head_horizontal_dev, 
                           gaze_vertical_dev, gaze_horizontal_dev):
        new_state = {
            "head_vertical": self._get_deviation_level(head_vertical_dev, self.ALLOWED_HEAD_VERTICAL_UP, self.ALLOWED_HEAD_VERTICAL_DOWN),
            "head_horizontal": self._get_deviation_level(head_horizontal_dev, self.ALLOWED_HEAD_HORIZONTAL_ANGLE),
            "gaze_vertical": self._get_deviation_level(gaze_vertical_dev, self.ALLOWED_GAZE_VERTICAL_UP, self.ALLOWED_GAZE_VERTICAL_DOWN),
            "gaze_horizontal": self._get_deviation_level(gaze_horizontal_dev, self.ALLOWED_GAZE_HORIZONTAL_ANGLE)
        }
        
        alerts = []
        
        for param in ["head_vertical", "head_horizontal", "gaze_vertical", "gaze_horizontal"]:
            old_level = session["current_state"][param]
            new_level = new_state[param]
            
            if new_level == 0:
                session["normal_frame_counters"][param] += 1
                session["warning_frame_counters"][param] = 0
                session["danger_frame_counters"][param] = 0
                
                if session["normal_frame_counters"][param] >= self.NORMAL_RESET_FRAMES:
                    session["warning_limit_exceeded"][param] = False
                    session["danger_limit_exceeded"][param] = False
                    session["normal_frame_counters"][param] = 0
                    
            elif new_level == 1:
                session["warning_frame_counters"][param] += 1
                session["normal_frame_counters"][param] = 0
                
                if session["warning_frame_counters"][param] >= self.WARNING_FRAME_LIMIT:
                    if not session["warning_limit_exceeded"][param]:
                        session["warning_limit_exceeded"][param] = True
                        alerts.append(f"WARNING: Excessive {param.replace('_', ' ')} deviation")
                
                session["danger_frame_counters"][param] = 0
                session["danger_limit_exceeded"][param] = False
                
            elif new_level == 2:
                session["warning_frame_counters"][param] += 1
                session["danger_frame_counters"][param] += 1
                session["normal_frame_counters"][param] = 0
                
                if session["danger_frame_counters"][param] >= self.DANGER_FRAME_LIMIT:
                    if not session["danger_limit_exceeded"][param]:
                        session["danger_limit_exceeded"][param] = True
                        alerts.append(f"DANGER: Critical {param.replace('_', ' ')} deviation")
            
            session["current_state"][param] = new_level
            
            if new_level == 0:
                if session["warning_limit_exceeded"][param]:
                    session["warning_limit_exceeded"][param] = False
                if session["danger_limit_exceeded"][param]:
                    session["danger_limit_exceeded"][param] = False
        
        return alerts
    
    def _run_yolo_detection(self, frame, session):
        if not self.yolo_model:
            return True
        
        session["yolo_frame_skip"] += 1
        
        if session["yolo_frame_skip"] >= self.YOLO_FRAME_INTERVAL:
            results = self.yolo_model(frame, verbose=False)
            session["yolo_frame_skip"] = 0
            
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
                    
                    session["person_count"] = person_count
                    session["phone_detected"] = phone_detected
                    
                    if self.yolo_verification_enabled:
                        session["gaze_tracking_enabled"] = (person_count == 1 and not phone_detected)
                    else:
                        session["gaze_tracking_enabled"] = True
            
            return session["gaze_tracking_enabled"]
        
        return session.get("gaze_tracking_enabled", True)
    
    async def process_frame(self, frame_data: str, session_id: str, calibration_requested: bool = False) -> Optional[Dict[str, Any]]:
        try:
            if ',' in frame_data:
                frame_data = frame_data.split(',')[1]
            
            img_bytes = base64.b64decode(frame_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                logger.error("Не удалось декодировать кадр")
                return None
            
            if session_id not in self.sessions:
                logger.error(f"Сессия не найдена: {session_id}")
                return None
            
            session = self.sessions[session_id]
            
            if self.face_mesh is None:
                await self.initialize_models()
            
            session["frame_count"] += 1
            current_time = time.time()
            if current_time - session["last_fps_update"] >= 1.0:
                session["fps"] = session["frame_count"] / (current_time - session["last_fps_update"])
                session["frame_count"] = 0
                session["last_fps_update"] = current_time
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            debug_frame = frame.copy() if settings.DEBUG else None
            
            gaze_tracking_enabled = self._run_yolo_detection(frame, session)
            
            results = self.face_mesh.process(frame_rgb)
            head_vertical_dev = 0
            head_horizontal_dev = 0
            gaze_vertical_dev = 0
            gaze_horizontal_dev = 0
            
            calibration_performed = False
            
            if gaze_tracking_enabled and results and results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0].landmark
                h, w = frame.shape[:2]
                
                session["current_face_size"] = self._compute_face_size(face_landmarks, w, h)
                
                if session["smoothed_face_size"] is None:
                    session["smoothed_face_size"] = session["current_face_size"]
                else:
                    session["smoothed_face_size"] = self._lerp_scalar(
                        session["smoothed_face_size"], 
                        session["current_face_size"], 
                        self.FACE_SIZE_SMOOTHING_FACTOR
                    )
                
                left_iris_idx = 468
                right_iris_idx = 473
                left_iris = face_landmarks[left_iris_idx]
                right_iris = face_landmarks[right_iris_idx]
                
                head_center, R_final, nose_points_3d = self._compute_and_draw_coordinate_box(
                    debug_frame if settings.DEBUG else frame,
                    face_landmarks, 
                    self.nose_indices, 
                    session["R_ref_nose"], 
                    w, h
                )
                
                head_direction_vector = self._calculate_head_direction(R_final)
                session["last_head_vector"] = head_direction_vector
                
                iris_3d_left = np.array([left_iris.x * w, left_iris.y * h, left_iris.z * w])
                iris_3d_right = np.array([right_iris.x * w, right_iris.y * h, right_iris.z * w])
                
                if settings.DEBUG:
                    x_iris_l, y_iris_l = int(left_iris.x * w), int(left_iris.y * h)
                    x_iris_r, y_iris_r = int(right_iris.x * w), int(right_iris.y * h)
                    cv2.circle(debug_frame, (x_iris_l, y_iris_l), 8, (25, 255, 255), -1)
                    cv2.circle(debug_frame, (x_iris_r, y_iris_r), 8, (25, 255, 255), -1)
                
                if calibration_requested or session.get("need_calibration", False):
                    calibration_performed = self._perform_calibration(
                        session, face_landmarks, head_center, R_final, 
                        nose_points_3d, iris_3d_left, iris_3d_right, 
                        head_direction_vector, w, h
                    )
                
                if session["left_sphere_locked"] and session["right_sphere_locked"]:
                    current_nose_scale = self._compute_scale(nose_points_3d)
                    
                    scale_ratio_l = current_nose_scale / session["left_calibration_nose_scale"] if session["left_calibration_nose_scale"] else 1.0
                    scaled_offset_l = session["left_sphere_local_offset"] * scale_ratio_l
                    sphere_world_l = head_center + R_final @ scaled_offset_l
                    
                    scale_ratio_r = current_nose_scale / session["right_calibration_nose_scale"] if session["right_calibration_nose_scale"] else 1.0
                    scaled_offset_r = session["right_sphere_local_offset"] * scale_ratio_r
                    sphere_world_r = head_center + R_final @ scaled_offset_r
                    
                    current_gaze_vector = self._calculate_gaze_direction(
                        iris_3d_left, iris_3d_right, sphere_world_l, sphere_world_r
                    )
                    session["last_gaze_vector"] = current_gaze_vector
                    
                    if current_gaze_vector is not None:
                        if session["smoothed_gaze_vector"] is None:
                            session["smoothed_gaze_vector"] = current_gaze_vector.copy()
                        else:
                            session["smoothed_gaze_vector"] = self._lerp_vector(
                                session["smoothed_gaze_vector"], 
                                current_gaze_vector, 
                                self.SMOOTHING_FACTOR
                            )
                    
                    gaze_vector_for_analysis = session["smoothed_gaze_vector"] if session["smoothed_gaze_vector"] is not None else current_gaze_vector
                    
                    if settings.DEBUG:
                        x_sphere_l, y_sphere_l = int(sphere_world_l[0]), int(sphere_world_l[1])
                        x_sphere_r, y_sphere_r = int(sphere_world_r[0]), int(sphere_world_r[1])
                        
                        cv2.circle(debug_frame, (x_sphere_l, y_sphere_l), 15, (255, 255, 25), 2)
                        cv2.circle(debug_frame, (x_sphere_r, y_sphere_r), 15, (25, 255, 255), 2)
                        
                        if gaze_vector_for_analysis is not None:
                            gaze_end_l = sphere_world_l + (gaze_vector_for_analysis * 100)
                            gaze_end_r = sphere_world_r + (gaze_vector_for_analysis * 100)
                            
                            cv2.line(debug_frame, (x_sphere_l, y_sphere_l), 
                                    (int(gaze_end_l[0]), int(gaze_end_l[1])), (55, 255, 0), 2)
                            cv2.line(debug_frame, (x_sphere_r, y_sphere_r), 
                                    (int(gaze_end_r[0]), int(gaze_end_r[1])), (55, 255, 0), 2)
                            
                            eye_center = ((sphere_world_l + sphere_world_r) / 2)[:2].astype(int)
                            combined_gaze_end = eye_center + (gaze_vector_for_analysis[:2] * 150)
                            cv2.arrowedLine(debug_frame, tuple(eye_center), tuple(combined_gaze_end.astype(int)),
                                          (0, 255, 255), 3, tipLength=0.3)
                    
                    if (session["calibration_completed"] and 
                        session["head_calibration_completed"] and 
                        gaze_vector_for_analysis is not None and 
                        head_direction_vector is not None):
                        
                        self._update_allowed_deviations(session)
                        
                        head_vertical_dev = self._calculate_vertical_deviation(
                            head_direction_vector, 
                            session["head_calibration_vector"]
                        )
                        head_horizontal_dev = self._calculate_horizontal_deviation(
                            head_direction_vector, 
                            session["head_calibration_vector"]
                        )
                        
                        gaze_vertical_dev = self._calculate_vertical_deviation(
                            gaze_vector_for_analysis, 
                            session["calibration_vector"]
                        )
                        gaze_horizontal_dev = self._calculate_horizontal_deviation(
                            gaze_vector_for_analysis, 
                            session["calibration_vector"]
                        )
                        
                        alerts = self._analyze_deviations(
                            session, 
                            head_vertical_dev, 
                            head_horizontal_dev,
                            gaze_vertical_dev, 
                            gaze_horizontal_dev
                        )
                        
                        if alerts:
                            for alert in alerts:
                                if alert not in session["alerts"]:  # Предотвращение дублирования
                                    session["alerts"].append(f"{datetime.now().strftime('%H:%M:%S')} - {alert}")
                            session["alerts"] = session["alerts"][-10:]  # Храним последние 10 алертов
            
            status_data = {
                "session_id": session_id,
                "user_id": session["user_id"],
                "frame_count": session["frame_count"],
                "fps": session.get("fps", 0),
                "timestamp": datetime.now().isoformat(),
                "status_changed": False,
                "current_state": session["current_state"].copy(),
                "warning_limit_exceeded": session["warning_limit_exceeded"].copy(),
                "danger_limit_exceeded": session["danger_limit_exceeded"].copy(),
                "deviations": {
                    "gaze_vertical": gaze_vertical_dev,
                    "gaze_horizontal": gaze_horizontal_dev,
                    "head_vertical": head_vertical_dev,
                    "head_horizontal": head_horizontal_dev
                },
                "limits": {
                    "gaze_vertical_up": self.ALLOWED_GAZE_VERTICAL_UP,
                    "gaze_vertical_down": self.ALLOWED_GAZE_VERTICAL_DOWN,
                    "gaze_horizontal": self.ALLOWED_GAZE_HORIZONTAL_ANGLE,
                    "head_vertical_up": self.ALLOWED_HEAD_VERTICAL_UP,
                    "head_vertical_down": self.ALLOWED_HEAD_VERTICAL_DOWN,
                    "head_horizontal": self.ALLOWED_HEAD_HORIZONTAL_ANGLE
                },
                "counters": {
                    "warning": {
                        "head_vertical": session["warning_frame_counters"]["head_vertical"],
                        "head_horizontal": session["warning_frame_counters"]["head_horizontal"],
                        "gaze_vertical": session["warning_frame_counters"]["gaze_vertical"],
                        "gaze_horizontal": session["warning_frame_counters"]["gaze_horizontal"]
                    },
                    "danger": {
                        "head_vertical": session["danger_frame_counters"]["head_vertical"],
                        "head_horizontal": session["danger_frame_counters"]["head_horizontal"],
                        "gaze_vertical": session["danger_frame_counters"]["gaze_vertical"],
                        "gaze_horizontal": session["danger_frame_counters"]["gaze_horizontal"]
                    },
                    "normal": {
                        "head_vertical": session["normal_frame_counters"]["head_vertical"],
                        "head_horizontal": session["normal_frame_counters"]["head_horizontal"],
                        "gaze_vertical": session["normal_frame_counters"]["gaze_vertical"],
                        "gaze_horizontal": session["normal_frame_counters"]["gaze_horizontal"]
                    }
                },
                "yolo_status": {
                    "enabled": session.get("gaze_tracking_enabled", True),
                    "person_count": session.get("person_count", 0),
                    "phone_detected": session.get("phone_detected", False)
                },
                "calibration_status": {
                    "calibration_completed": session["calibration_completed"],
                    "head_calibration_completed": session["head_calibration_completed"],
                    "eye_spheres_locked": session["left_sphere_locked"] and session["right_sphere_locked"],
                    "calibration_performed": calibration_performed
                },
                "alerts": session["alerts"][-5:],  # Последние 5 алертов
                "graphics_frame": None
            }
            
            if session["last_status_data"]:
                old_state = session["last_status_data"]["current_state"]
                new_state = status_data["current_state"]
                
                for param in old_state:
                    if old_state[param] != new_state[param]:
                        status_data["status_changed"] = True
                        break
            
            session["last_status_data"] = status_data
            
            if settings.DEBUG and debug_frame is not None:
                info_lines = [
                    f"Session: {session_id[:8]}...",
                    f"FPS: {status_data['fps']:.1f}",
                    f"Face size: {session.get('current_face_size', 0):.0f}",
                    f"Gaze: V={gaze_vertical_dev:+.1f}°, H={gaze_horizontal_dev:+.1f}°",
                    f"Head: V={head_vertical_dev:+.1f}°, H={head_horizontal_dev:+.1f}°",
                    f"State: H_V={status_data['current_state']['head_vertical']}, "
                    f"H_H={status_data['current_state']['head_horizontal']}, "
                    f"G_V={status_data['current_state']['gaze_vertical']}, "
                    f"G_H={status_data['current_state']['gaze_horizontal']}",
                    f"Warning counters: H_V={session['warning_frame_counters']['head_vertical']}/{self.WARNING_FRAME_LIMIT}, "
                    f"H_H={session['warning_frame_counters']['head_horizontal']}/{self.WARNING_FRAME_LIMIT}",
                    f"Danger counters: H_V={session['danger_frame_counters']['head_vertical']}/{self.DANGER_FRAME_LIMIT}, "
                    f"H_H={session['danger_frame_counters']['head_horizontal']}/{self.DANGER_FRAME_LIMIT}",
                    f"Calibration: {'Complete' if session['calibration_completed'] else 'Pending'}",
                    f"Eye Spheres: {'Locked' if session['left_sphere_locked'] and session['right_sphere_locked'] else 'Unlocked'}"
                ]
                
                for i, line in enumerate(info_lines):
                    cv2.putText(debug_frame, line, (10, 30 + i * 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                _, buffer = cv2.imencode('.jpg', debug_frame)
                status_data["graphics_frame"] = base64.b64encode(buffer).decode('utf-8')
            
            return status_data
            
        except Exception as e:
            logger.error(f"Ошибка обработки кадра: {e}", exc_info=True)
            return None
    
    def _perform_calibration(self, session, face_landmarks, head_center, R_final, 
                           nose_points_3d, iris_3d_left, iris_3d_right, 
                           head_direction_vector, w, h):
        try:
            logger.info(f"Начинаем калибровку для сессии {session['id']}")
            
            if not (session["left_sphere_locked"] and session["right_sphere_locked"]):
                current_nose_scale = self._compute_scale(nose_points_3d)
                
                session["left_sphere_local_offset"] = R_final.T @ (iris_3d_left - head_center)
                camera_dir_world = np.array([0, 0, 1])
                camera_dir_local = R_final.T @ camera_dir_world
                session["left_sphere_local_offset"] += 20 * camera_dir_local
                session["left_calibration_nose_scale"] = current_nose_scale
                session["left_sphere_locked"] = True
                
                session["right_sphere_local_offset"] = R_final.T @ (iris_3d_right - head_center)
                session["right_sphere_local_offset"] += 20 * camera_dir_local
                session["right_calibration_nose_scale"] = current_nose_scale
                session["right_sphere_locked"] = True
                
                logger.info("Сферы глаз откалиброваны")
            
            current_nose_scale = self._compute_scale(nose_points_3d)
            scale_ratio_l = current_nose_scale / session["left_calibration_nose_scale"] if session["left_calibration_nose_scale"] else 1.0
            scale_ratio_r = current_nose_scale / session["right_calibration_nose_scale"] if session["right_calibration_nose_scale"] else 1.0
            
            sphere_world_l_calib = head_center + R_final @ (session["left_sphere_local_offset"] * scale_ratio_l)
            sphere_world_r_calib = head_center + R_final @ (session["right_sphere_local_offset"] * scale_ratio_r)
            
            left_gaze_dir = iris_3d_left - sphere_world_l_calib
            right_gaze_dir = iris_3d_right - sphere_world_r_calib
            
            left_gaze_dir[0] *= self.HORIZONTAL_SENSITIVITY_FACTOR
            left_gaze_dir[1] *= self.VERTICAL_SENSITIVITY_FACTOR
            right_gaze_dir[0] *= self.HORIZONTAL_SENSITIVITY_FACTOR
            right_gaze_dir[1] *= self.VERTICAL_SENSITIVITY_FACTOR
            
            left_gaze_dir = self._normalize(left_gaze_dir)
            right_gaze_dir = self._normalize(right_gaze_dir)
            
            session["calibration_vector"] = (left_gaze_dir + right_gaze_dir) / 2
            session["calibration_vector"] = self._normalize(session["calibration_vector"])
            session["calibration_completed"] = True
            
            session["head_calibration_vector"] = head_direction_vector.copy()
            session["head_calibration_completed"] = True
            
            session["calibration_face_size"] = self._compute_face_size(face_landmarks, w, h)
            session["smoothed_face_size"] = session["calibration_face_size"]
            
            session["smoothed_gaze_vector"] = session["calibration_vector"].copy()
            
            session["adaptation_factor"] = 1.0
            session["smoothed_adaptation_factor"] = 1.0
            
            for param in session["current_state"]:
                session["current_state"][param] = 0
            
            for param in session["warning_frame_counters"]:
                session["warning_frame_counters"][param] = 0
                session["danger_frame_counters"][param] = 0
                session["normal_frame_counters"][param] = 0
                session["warning_limit_exceeded"][param] = False
                session["danger_limit_exceeded"][param] = False
            
            session["alerts"] = []
            
            session["need_calibration"] = False
            
            logger.info(f"Калибровка завершена для сессии {session['id']}")
            logger.info(f"Вектор взгляда: {session['calibration_vector']}")
            logger.info(f"Вектор головы: {session['head_calibration_vector']}")
            logger.info(f"Размер лица: {session['calibration_face_size']:.0f} px")
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при калибровке: {e}")
            return False
    
    def calibrate(self, session_id: str):
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        
        session["need_calibration"] = True
        
        logger.info(f"Запрошена калибровка для сессии {session_id}")
        return True
    
    def set_calibration_data(self, session_id: str, calibration_data: Dict):
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        
        if "gaze_vector" in calibration_data:
            session["calibration_vector"] = np.array(calibration_data["gaze_vector"])
            session["calibration_completed"] = True
        
        if "head_vector" in calibration_data:
            session["head_calibration_vector"] = np.array(calibration_data["head_vector"])
            session["head_calibration_completed"] = True
        
        if "face_size" in calibration_data:
            session["calibration_face_size"] = calibration_data["face_size"]
        
        if "left_sphere_offset" in calibration_data:
            session["left_sphere_local_offset"] = np.array(calibration_data["left_sphere_offset"])
            session["left_sphere_locked"] = True
        
        if "right_sphere_offset" in calibration_data:
            session["right_sphere_local_offset"] = np.array(calibration_data["right_sphere_offset"])
            session["right_sphere_locked"] = True
        
        if "nose_scale" in calibration_data:
            session["left_calibration_nose_scale"] = calibration_data["nose_scale"]
            session["right_calibration_nose_scale"] = calibration_data["nose_scale"]
        
        logger.info(f"Данные калибровки установлены для сессии {session_id}")
        return True
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        return self.sessions.get(session_id)
    
    def delete_session(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Сессия удалена: {session_id}")
            return True
        return False
    
    def cleanup(self):
        if self.face_mesh:
            self.face_mesh.close()
        logger.info("Ресурсы ProctorService очищены")
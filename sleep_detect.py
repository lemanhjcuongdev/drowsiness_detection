import imutils
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
import time
import dlib
import cv2, os, sys
import collections
import random
import face_recognition
import pickle
import math
import threading

################## PHAN DINH NGHIA CLASS, FUNCTION #######################


# Class dinh nghia vi tri 2 mat con nguoi
class FacialLandMarksPosition:
    left_eye_start_index, left_eye_end_index = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    right_eye_start_index, right_eye_end_index = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


# Ham tinh Eye Aspect Ratio (EAR) - thay the model deep learning
def calculate_ear(eye_landmarks):
    """
    Tinh Eye Aspect Ratio (EAR) tu 6 diem landmark cua mat.
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    
    Khi mat mo: EAR cao (~0.25-0.35)
    Khi mat dong: EAR thap (~0.1-0.15)
    """
    # Tinh khoang cach doc (chieu cao mat)
    A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])  # p2 - p6
    B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])  # p3 - p5
    
    # Tinh khoang cach ngang (chieu rong mat)
    C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])  # p1 - p4
    
    # Tinh EAR
    ear = (A + B) / (2.0 * C) if C > 0 else 0
    return ear


# Nguong EAR de xac dinh mat dong hay mo
EAR_THRESHOLD = 0.22  # Duoi nguong nay = mat dong


# Ham tinh goc huong dau (Head Pose Estimation)
def get_head_pose(face_landmarks, frame_shape):
    """
    Tinh goc huong dau su dung solvePnP
    Tra ve: pitch (ngua len/cui xuong), yaw (quay trai/phai), roll (nghieng)
    """
    # 6 diem chuan tren khuon mat (theo 68 landmarks)
    # 30: mui, 8: cam, 36: mat trai, 45: mat phai, 48: khoe mieng trai, 54: khoe mieng phai
    image_points = np.array([
        face_landmarks[30],  # Mui
        face_landmarks[8],   # Cam
        face_landmarks[36],  # Goc mat trai
        face_landmarks[45],  # Goc mat phai
        face_landmarks[48],  # Khoe mieng trai
        face_landmarks[54]   # Khoe mieng phai
    ], dtype=np.float64)

    # Toa do 3D chuan cua khuon mat (mo hinh trung binh)
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Mui
        (0.0, -330.0, -65.0),        # Cam
        (-225.0, 170.0, -135.0),     # Goc mat trai
        (225.0, 170.0, -135.0),      # Goc mat phai
        (-150.0, -150.0, -125.0),    # Khoe mieng trai
        (150.0, -150.0, -125.0)      # Khoe mieng phai
    ], dtype=np.float64)

    # Ma tran camera (xap xi)
    height, width = frame_shape[:2]
    focal_length = width
    center = (width / 2, height / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    # He so bien dang (giu = 0)
    dist_coeffs = np.zeros((4, 1))

    # Giai bai toan PnP
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, 
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    # Chuyen rotation vector thanh rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    # Tinh goc Euler tu rotation matrix
    sy = math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = math.atan2(-rotation_matrix[2, 0], sy)
        z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        x = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y = math.atan2(-rotation_matrix[2, 0], sy)
        z = 0

    # Chuyen tu radian sang do
    pitch = math.degrees(x)  # Ngua len/cui xuong
    yaw = math.degrees(y)    # Quay trai/phai
    roll = math.degrees(z)   # Nghieng

    return pitch, yaw, roll


################ CHUONG TRINH CHINH ##############################

# Nguong phat hien huong dau (mo rong de giam bao sai)
HEAD_UP_THRESHOLD = -25      # Ngua len > 25 do
HEAD_DOWN_THRESHOLD = 25     # Cui xuong > 25 do
HEAD_LEFT_THRESHOLD = -30    # Quay trai > 30 do
HEAD_RIGHT_THRESHOLD = 30    # Quay phai > 30 do

# Load model dlib de phat hien cac diem tren mat nguoi - landmark
facial_landmarks_predictor = 'shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(facial_landmarks_predictor)

# Khong can load model deep learning nua - su dung EAR thay the

# Lay anh tu Webcam
cap = cv2.VideoCapture(0)
scale = 0.5
countClose = 0
countDistracted = 0  # Dem so frame mat tap trung
currState = 0
alarmThreshold = 5
distractedThreshold = 10  # Nguong canh bao mat tap trung

# Calibration offset (nhin thang = 0 do sau khi calibrate)
# Mac dinh cho webcam laptop: Pitch = 180 degree, Yaw = 0 degree la nhin thang
pitch_offset = 180
yaw_offset = 0
is_calibrated = True  # Da co offset mac dinh


while (True):
    c = time.time()
    # Doc anh tu webcam va chuyen thanh RGB
    ret, frame = cap.read()
    if not ret:
        break
        
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize anh con 50% kich thuoc goc
    original_height, original_width = image.shape[:2]
    resized_image = cv2.resize(image, (0, 0), fx=scale, fy=scale)

    # Chuyen sang he mau LAB de lay thanh lan Lightness
    lab = cv2.cvtColor(resized_image, cv2.COLOR_BGR2LAB)
    l, _, _ = cv2.split(lab)
    resized_height, resized_width = l.shape[:2]
    height_ratio, width_ratio = original_height / resized_height, original_width / resized_width

    # Tim kiem khuon mat bang HOG
    face_locations = face_recognition.face_locations(l, model='hog')

    # Trang thai chu y
    attention_status = "ATTENTIVE"
    status_color = (0, 255, 0)  # Mau xanh la
    head_direction = ""

    # Neu tim thay it nhat 1 khuon mat
    if len(face_locations):

        # Lay vi tri khuon mat
        top, right, bottom, left = face_locations[0]
        x1, y1, x2, y2 = left, top, right, bottom
        x1 = int(x1 * width_ratio)
        y1 = int(y1 * height_ratio)
        x2 = int(x2 * width_ratio)
        y2 = int(y2 * height_ratio)

        # Trich xuat vi tri 2 mat

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        shape = predictor(gray, dlib.rectangle(x1, y1, x2, y2))
        face_landmarks = face_utils.shape_to_np(shape)

        # ===== PHAT HIEN HUONG DAU =====
        pitch_raw, yaw_raw, roll = get_head_pose(face_landmarks, frame.shape)
        
        # Ap dung calibration offset
        pitch = pitch_raw - pitch_offset
        yaw = yaw_raw - yaw_offset
        
        # Kiem tra huong dau
        is_distracted = False
        
        if pitch < HEAD_UP_THRESHOLD:
            head_direction = "LOOKING UP"
            is_distracted = True
        elif pitch > HEAD_DOWN_THRESHOLD:
            head_direction = "LOOKING DOWN"
            is_distracted = True
        elif yaw < HEAD_LEFT_THRESHOLD:
            head_direction = "LOOKING LEFT"
            is_distracted = True
        elif yaw > HEAD_RIGHT_THRESHOLD:
            head_direction = "LOOKING RIGHT"
            is_distracted = True
        else:
            head_direction = "FORWARD"

        # ===== PHAT HIEN MAT BANG EAR =====
        left_eye_landmarks = face_landmarks[FacialLandMarksPosition.left_eye_start_index:
                                            FacialLandMarksPosition.left_eye_end_index]
        right_eye_landmarks = face_landmarks[FacialLandMarksPosition.right_eye_start_index:
                                             FacialLandMarksPosition.right_eye_end_index]

        # Tinh EAR cho tung mat
        left_ear = calculate_ear(left_eye_landmarks)
        right_ear = calculate_ear(right_eye_landmarks)
        avg_ear = (left_ear + right_ear) / 2.0

        # Kiem tra mat dong hay mo
        left_eye_open = 'yes' if left_ear > EAR_THRESHOLD else 'no'
        right_eye_open = 'yes' if right_ear > EAR_THRESHOLD else 'no'
        eyes_closed = avg_ear < EAR_THRESHOLD

        # ===== DANH GIA TRANG THAI CHU Y =====
        if eyes_closed:
            attention_status = "EYES CLOSED"
            status_color = (0, 0, 255)  # Do
            countClose += 1
            currState = 1
        elif is_distracted:
            attention_status = f"DISTRACTED - {head_direction}"
            status_color = (0, 165, 255)  # Cam
            countDistracted += 1
            countClose = 0
            currState = 2
        else:
            attention_status = "ATTENTIVE"
            status_color = (0, 255, 0)  # Xanh la
            countClose = 0
            countDistracted = 0
            currState = 0

        # Ve khung mat
        cv2.rectangle(frame, (x1, y1), (x2, y2), status_color, 2)
        
        # Luu thong tin de hien thi sau khi flip
        calib_text = "[CALIBRATED]" if is_calibrated else "[Press C to calibrate]"
        pose_info = f"Pitch: {pitch:.1f} Yaw: {yaw:.1f} {calib_text}"

        print(f'Eyes: L={left_eye_open} R={right_eye_open} EAR:{avg_ear:.2f} | Head: {head_direction} | Pitch:{pitch:.1f} Yaw:{yaw:.1f}')

    else:
        attention_status = "NO FACE DETECTED"
        status_color = (128, 128, 128)
        pose_info = ""

    frame = cv2.flip(frame, 1)
    
    # Tinh toan kich thuoc hien thi theo ti le goc de tranh meo anh
    display_width = 960
    display_height = int(display_width * original_height / original_width)
    frame = cv2.resize(frame, (display_width, display_height))
    
    # Hien thi trang thai chu y
    cv2.putText(frame, attention_status, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2, cv2.LINE_AA)
    
    # Hien thi goc huong dau (sau khi flip de khong bi lat)
    if pose_info:
        cv2.putText(frame, pose_info, (10, display_height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Canh bao nham mat
    if countClose > alarmThreshold:
        cv2.putText(frame, "ALERT: Eyes closed too long!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
    
    # Canh bao mat tap trung
    if countDistracted > distractedThreshold:
        cv2.putText(frame, "ALERT: Not paying attention!", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2, cv2.LINE_AA)

    cv2.imshow('Attention Detection', frame)

    key = cv2.waitKey(1) & 0xFF
    
    # Bam 'c' de calibrate vi tri nhin thang
    if key == ord('c') and len(face_locations) > 0:
        pitch_offset = pitch_raw
        yaw_offset = yaw_raw
        is_calibrated = True
        print(f"=== CALIBRATED! Offset: Pitch={pitch_offset:.1f}, Yaw={yaw_offset:.1f} ===")
    
    # Bam 'q' de thoat
    if key == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
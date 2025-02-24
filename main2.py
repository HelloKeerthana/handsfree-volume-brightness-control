import cv2
import numpy as np 
import mediapipe as mp
from math import hypot
import screen_brightness_control as sbc 
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL

def control_system(): 
    output_device = AudioUtilities.GetSpeakers()
    interface = output_device.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    audio_control = cast(interface, POINTER(IAudioEndpointVolume))
    volume_range = audio_control.GetVolumeRange()
    min_level, max_level, _ = volume_range

    hand_model = mp.solutions.hands
    hand_tracking = hand_model.Hands(
        static_image_mode=False,
        min_tracking_confidence=0.75,
        min_detection_confidence=0.75,
        max_num_hands=2,
        model_complexity=1
    )

    drawer = mp.solutions.drawing_utils
    capture = cv2.VideoCapture(0)
    
    try:
        while capture.isOpened():
            status, frame = capture.read()
            if not status:
                break
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hand_tracking.process(frame_rgb)

            left_hand, right_hand = detect_hands(frame, results, drawer, hand_model)
            
            if left_hand:
                brightness_level = np.interp(compute_distance(frame, left_hand), [50, 220], [0, 100])
                sbc.set_brightness(brightness_level)
            
            if right_hand:
                volume_level = np.interp(compute_distance(frame, right_hand), [50, 220], [min_level, max_level])
                audio_control.SetMasterVolumeLevel(volume_level, None)

            cv2.imshow('Control Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def detect_hands(frame, results, drawer, hand_model):
    left_hand_data = []
    right_hand_data = []
    
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            for idx, landmark in enumerate(hand.landmark):
                h, w, _ = frame.shape
                x, y = int(landmark.x * w), int(landmark.y * h)
                
                if idx in [4, 8]:
                    point_data = [idx, x, y]
                    
                    if hand == results.multi_hand_landmarks[0]:
                        left_hand_data.append(point_data)
                    elif hand == results.multi_hand_landmarks[1]:
                        right_hand_data.append(point_data)
            drawer.draw_landmarks(frame, hand, hand_model.HAND_CONNECTIONS)
    return left_hand_data, right_hand_data


def compute_distance(frame, hand_data):
    if len(hand_data) < 2:
        return None
    
    (x1, y1), (x2, y2) = (hand_data[0][1], hand_data[0][2]), (hand_data[1][1], hand_data[1][2])
    cv2.circle(frame, (x1, y1), 7, (0, 255, 0), cv2.FILLED)
    cv2.circle(frame, (x2, y2), 7, (0, 255, 0), cv2.FILLED)
    midpoint = ((x1 + x2) // 2, (y1 + y2) // 2)
    cv2.circle(frame, midpoint, 7, (0, 255, 0), cv2.FILLED)
    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

    return hypot(x2 - x1, y2 - y1)

if __name__ == "__main__":
    control_system()

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
from collections import deque

# CONFIG 
FPS_SMOOTH = 0.1
BUFFER_LEN = 8                  # how many frames to use for velocity estimation
SWIPE_THRESHOLD = 0.08        # normalized x displacement over buffer -> swipe
SWIPE_COOLDOWN = 0.6            # seconds between left/right swipes
VERTICAL_THRESHOLD = 0.12       # normalized y displacement -> jump/slide
VERT_COOLDOWN = 0.6
OPEN_PALM_THRESHOLD = 0.08      # average distance between finger tips and wrist for open vs closed
# END CONFIG

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

pyautogui.PAUSE = 0  
pyautogui.FAILSAFE = False

# Track wrist positions (normalized 0..1)
pos_buffer = deque(maxlen=BUFFER_LEN)
time_buffer = deque(maxlen=BUFFER_LEN)

last_swipe_time = 0
last_vert_time = 0

def norm_landmark_xy(landmark, w, h):
    return np.array([landmark.x, landmark.y])  # already normalized

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
) as hands:

    prev = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)  
        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        now = time.time()
        dt = now - prev if now - prev > 0 else 1/30
        prev = now

        # Default dx/dy so they always exist (this prevents NameError)
        dx = 0.0
        dy = 0.0

        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0]
            wrist = lm.landmark[0]
            wrist_xy = norm_landmark_xy(wrist, w, h)

            pos_buffer.append(wrist_xy)
            time_buffer.append(now)

            # ---- Compute dx, dy safely ----
            if len(pos_buffer) >= 3:
                dx = pos_buffer[-1][0] - pos_buffer[0][0]
                dy = pos_buffer[-1][1] - pos_buffer[0][1]

                dt_span = time_buffer[-1] - time_buffer[0]
                if dt_span < 1e-6:
                    dt_span = 1e-6

                vx = dx / dt_span
                vy = dy / dt_span
            else:
                vx = vy = 0.0

            # ---- Hand openness measurement ----
            tips_idx = [4, 8, 12, 16, 20]
            dists = []
            for t in tips_idx:
                tip = lm.landmark[t]
                tip_xy = np.array([tip.x, tip.y])
                dists.append(np.linalg.norm(tip_xy - wrist_xy))

            avg_tip_dist = np.mean(dists)

            # ---- Draw skeleton ----
            mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

            # ---- Gesture decisions ----
            action = None
            cur_time = now

            # Swipe L/R
            if abs(dx) > SWIPE_THRESHOLD and (cur_time - last_swipe_time) > SWIPE_COOLDOWN:
                if dx < 0:
                    action = "left"
                    pyautogui.press("left")
                else:
                    action = "right"
                    pyautogui.press("right")
                last_swipe_time = cur_time

            # Jump (upward motion = dy negative)
            if (-dy > VERTICAL_THRESHOLD) and (cur_time - last_vert_time > VERT_COOLDOWN):
                if avg_tip_dist > OPEN_PALM_THRESHOLD:  
                    action = "jump"
                    pyautogui.press("up")
                    last_vert_time = cur_time

            # Slide (downward motion = dy positive)
            if (dy > VERTICAL_THRESHOLD) and (cur_time - last_vert_time > VERT_COOLDOWN):
                if avg_tip_dist < OPEN_PALM_THRESHOLD * 1.1 or wrist_xy[1] > 0.75:
                    action = "slide"
                    pyautogui.press("down")
                    last_vert_time = cur_time

            # ---- Show text ----
            if action:
                cv2.putText(frame, f"Action: {action}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.putText(frame, f'vx={vx:.2f} vy={vy:.2f} avg_tip={avg_tip_dist:.3f}',
                        (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        else:
            # No hand → reset buffer
            pos_buffer.clear()
            time_buffer.clear()

        cv2.imshow("Hand Control Prototype - press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

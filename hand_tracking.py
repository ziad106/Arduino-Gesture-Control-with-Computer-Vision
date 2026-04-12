import cv2
import mediapipe as mp
import mediapipe.tasks as tasks
import os
import serial
import time

vision = tasks.vision

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

def count_fingers(hand_landmarks, hand_label):
    fingers = []

    # Thumb: compare tip (4) x vs MCP (2) x
    # Labels are flipped from MediaPipe, so conditions are swapped
    if hand_label == "Left":  # actually right hand in flipped frame
        if hand_landmarks[4].x > hand_landmarks[2].x:
            fingers.append(1)
    else:  # actually left hand in flipped frame
        if hand_landmarks[4].x < hand_landmarks[2].x:
            fingers.append(1)

    # Four fingers: tip y < PIP y (finger is raised)
    # Index
    if hand_landmarks[8].y < hand_landmarks[6].y:
        fingers.append(1)
    else:
        fingers.append(0)

    # Middle
    if hand_landmarks[12].y < hand_landmarks[10].y:
        fingers.append(1)
    else:
        fingers.append(0)

    # Ring
    if hand_landmarks[16].y < hand_landmarks[14].y:
        fingers.append(1)
    else:
        fingers.append(0)

    # Pinky
    if hand_landmarks[20].y < hand_landmarks[18].y:
        fingers.append(1)
    else:
        fingers.append(0)

    return sum(fingers), fingers

def draw_landmarks(frame, hand_landmarks_list, hand_labels_list, fingers_count_list):
    h, w = frame.shape[:2]
    for idx, hand_landmarks in enumerate(hand_landmarks_list):
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

        # Draw connections
        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, pts[start], pts[end], (0, 200, 255), 2)

        # Draw points
        for pt in pts:
            cv2.circle(frame, pt, 5, (0, 255, 0), -1)

        # Display finger count for this hand
        hand_label = hand_labels_list[idx]
        finger_count = fingers_count_list[idx]
        label_text = f"{hand_label}: {finger_count} fingers"

        x_offset = 20 if hand_label == "Left" else w - 300
        y_offset = 100 + (idx * 40)

        cv2.putText(frame, label_text, (x_offset, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

base_options = tasks.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

landmarker = vision.HandLandmarker.create_from_options(options)

arduino = serial.Serial('/dev/cu.usbmodem141011', 9600)
time.sleep(2)  # Wait for Arduino to reset after serial connection

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open camera")
    print("Fix: System Settings > Privacy & Security > Camera")
    print("     Allow Terminal (or VS Code) camera access, then re-run.")
    landmarker.close()
    exit(1)

print("Hand tracking with finger counting started. Press 'q' to quit.")

frame_count = 0
last_sent = -1

while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)
    frame_count += 1

    timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    if timestamp_ms == 0:
        timestamp_ms = frame_count * 33  # ~30fps fallback

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    if result.hand_landmarks:
        fingers_counts = []
        hand_names = []

        for idx, hand_landmarks in enumerate(result.hand_landmarks):
            # Extract hand label from handedness
            detected_label = result.handedness[idx][0].display_name  # "Left" or "Right"
            # Flip the label because frame is horizontally flipped
            hand_label = "Right" if detected_label == "Left" else "Left"
            hand_names.append(hand_label)
            finger_count, _ = count_fingers(hand_landmarks, hand_label)
            fingers_counts.append(finger_count)

        draw_landmarks(frame, result.hand_landmarks, hand_names, fingers_counts)

        num_hands = len(result.hand_landmarks)
        total_fingers = sum(fingers_counts)
        title = f"Hands: {num_hands} | Total Fingers: {total_fingers}"
        cv2.putText(frame, title, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        if total_fingers != last_sent:
            arduino.write(f"{total_fingers}\n".encode())
            arduino.flush()
            last_sent = total_fingers
    else:
        cv2.putText(frame, "No hand detected", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if last_sent != 0:
            arduino.write(b"0\n")
            arduino.flush()
            last_sent = 0

    cv2.imshow("Hand Tracking - Finger Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

landmarker.close()
cap.release()
arduino.close()
cv2.destroyAllWindows()

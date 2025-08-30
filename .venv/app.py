
import cv2
import time
import numpy as np
import mediapipe as mp

# --- Setup MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Finger landmark indices
# Thumb: 4 (tip), 3 (ip)
# Index: 8 (tip), 6 (pip)
# Middle: 12 (tip), 10 (pip)
# Ring: 16 (tip), 14 (pip)
# Pinky: 20 (tip), 18 (pip)
FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = {8: 6, 12: 10, 16: 14, 20: 18}

def landmarks_to_xy(landmarks, w, h):
    """Convert normalized landmarks to pixel coords as (x, y)."""
    pts = []
    for lm in landmarks.landmark:
        pts.append((int(lm.x * w), int(lm.y * h)))
    return pts

def finger_states(landmarks, hand_label, w, h):
    """
    Returns dict of which fingers are 'up'. Uses simple, robust rules:
    - For index/middle/ring/pinky: tip.y < pip.y  => finger is extended
    - For thumb: compare tip.x vs ip.x with handedness
      (Image is processed unflipped, so this works consistently)
    """
    pts = landmarks_to_xy(landmarks, w, h)
    states = {"thumb": False, "index": False, "middle": False, "ring": False, "pinky": False}

    # Non-thumb fingers
    for tip in [8, 12, 16, 20]:
        pip = FINGER_PIPS[tip]
        states_key = {8: "index", 12: "middle", 16: "ring", 20: "pinky"}[tip]
        states[states_key] = pts[tip][1] < pts[pip][1]  # smaller y => higher => extended

    # Thumb logic (Right hand: tip.x > ip.x; Left hand: tip.x < ip.x)
    thumb_tip = pts[4][0]
    thumb_ip  = pts[3][0]
    if hand_label == "Right":
        states["thumb"] = thumb_tip > thumb_ip
    else:  # "Left"
        states["thumb"] = thumb_tip < thumb_ip

    return states

def classify_gesture(states):
    """Return gesture label from finger states."""
    thumb, idx, mid, ring, pinky = (states["thumb"], states["index"], states["middle"], states["ring"], states["pinky"])

    # Fist: none up
    if not any([thumb, idx, mid, ring, pinky]):
        return "Fist"

    # Open Palm: all up
    if all([thumb, idx, mid, ring, pinky]):
        return "Open Palm"

    # Peace: only index & middle up (allow thumb either way for robustness)
    if idx and mid and not ring and not pinky:
        return "Peace"

    # Thumbs Up: thumb up, others down
    if thumb and not idx and not mid and not ring and not pinky:
        return "Thumbs Up"

    return "Unknown"

def draw_bbox_and_label(img, pts, label, color=(0, 255, 0)):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x1, y1 = max(min(xs)-10, 0), max(min(ys)-10, 0)
    x2, y2 = min(max(xs)+10, img.shape[1]-1), min(max(ys)+10, img.shape[0]-1)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, label, (x1, max(0, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

def main():
    cap = cv2.VideoCapture(0)
    # Try to set a good resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    prev_time = time.time()
    recording = False
    writer = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("WARN: Empty frame from camera.")
                break

            h, w = frame.shape[:2]
            # Process on the ORIGINAL frame (not flipped), for consistent handedness logic
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            overlay = frame.copy()

            if result.multi_hand_landmarks:
                # Handedness corresponds to each set of landmarks by index
                for hand_idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
                    # Draw landmarks without styles (default simple lines and circles)
                    mp_drawing.draw_landmarks(
                        overlay,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

                    # Determine left/right
                    hand_label = "Right"
                    if result.multi_handedness and len(result.multi_handedness) > hand_idx:
                        hand_label = result.multi_handedness[hand_idx].classification[0].label

                    states = finger_states(hand_landmarks, hand_label, w, h)
                    gesture = classify_gesture(states)

                    pts = landmarks_to_xy(hand_landmarks, w, h)
                    draw_bbox_and_label(overlay, pts, f"{gesture}")

            # Calculate FPS
            now = time.time()
            fps = 1.0 / max(1e-6, (now - prev_time))
            prev_time = now
            cv2.putText(overlay, f"FPS: {int(fps)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Flip ONLY for display (nice mirror view)
            display = cv2.flip(overlay, 1)

            # Recording toggle
            if recording and writer is not None:
                writer.write(display)
                cv2.putText(display, "REC", (display.shape[1]-80, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)

            cv2.imshow("Hand Gesture Recognition (press 'r' to record, 'q' to quit)", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                if not recording:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter("demo.mp4", fourcc, 30.0, (display.shape[1], display.shape[0]))
                    recording = True
                    print("Recording started: demo.mp4")
                else:
                    recording = False
                    if writer:
                        writer.release()
                        writer = None
                    print("Recording stopped.")
    finally:
        if writer:
            writer.release()
        hands.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

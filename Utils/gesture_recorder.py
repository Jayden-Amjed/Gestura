import csv
import os
import cv2
import mediapipe as mp


OUTPUT_FILE = "gesture_data.csv"


def normalize_landmarks(hand_landmarks):
    """
    Converts MediaPipe hand landmarks into normalized coordinates.
    Wrist point becomes the origin so hand position on screen matters less.
    """
    landmarks = hand_landmarks.landmark

    base_x = landmarks[0].x
    base_y = landmarks[0].y
    base_z = landmarks[0].z

    normalized = []

    for lm in landmarks:
        normalized.extend([
            lm.x - base_x,
            lm.y - base_y,
            lm.z - base_z
        ])

    return normalized


def save_gesture(label, landmark_data):
    file_exists = os.path.exists(OUTPUT_FILE)

    with open(OUTPUT_FILE, "a", newline="") as file:
        writer = csv.writer(file)

        if not file_exists:
            header = ["label"]
            for i in range(21):
                header += [f"x{i}", f"y{i}", f"z{i}"]
            writer.writerow(header)

        writer.writerow([label] + landmark_data)


def main():
    label = input("Enter gesture label to record, like SET or PRINT: ").strip().upper()

    if not label:
        print("No label entered.")
        return

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open camera.")
        return

    print(f"Recording gesture: {label}")
    print("Press S to save a sample.")
    print("Press Q to quit.")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Could not read camera frame.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            current_landmarks = None

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                current_landmarks = normalize_landmarks(hand_landmarks)

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                cv2.putText(
                    frame,
                    f"Label: {label}",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    "Press S to save sample | Q to quit",
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
            else:
                cv2.putText(
                    frame,
                    "No hand detected",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )

            cv2.imshow("Gestura Gesture Recorder", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                if current_landmarks is not None:
                    save_gesture(label, current_landmarks)
                    print(f"Saved sample for {label}")
                else:
                    print("No hand detected. Sample not saved.")

            elif key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
import csv
import math
import cv2
import mediapipe as mp


DATA_FILE = "gesture_data.csv"


def normalize_landmarks(hand_landmarks):
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


def load_gesture_data():
    samples = []

    with open(DATA_FILE, "r") as file:
        reader = csv.reader(file)
        next(reader)

        for row in reader:
            label = row[0]
            values = [float(value) for value in row[1:]]
            samples.append((label, values))

    return samples


def distance(a, b):
    total = 0

    for x, y in zip(a, b):
        total += (x - y) ** 2

    return math.sqrt(total)


def predict_gesture(current_landmarks, samples):
    best_label = "UNKNOWN"
    best_distance = float("inf")

    for label, saved_landmarks in samples:
        d = distance(current_landmarks, saved_landmarks)

        if d < best_distance:
            best_distance = d
            best_label = label

    return best_label, best_distance


def main():
    samples = load_gesture_data()

    print(f"Loaded {len(samples)} gesture samples.")
    print("Press Q to quit.")

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open camera.")
        return

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

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                current_landmarks = normalize_landmarks(hand_landmarks)

                label, d = predict_gesture(current_landmarks, samples)

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                cv2.putText(
                    frame,
                    f"Gesture: {label}",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Distance: {d:.3f}",
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

            cv2.imshow("Gestura Gesture Recognizer", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
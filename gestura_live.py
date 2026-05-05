import csv
import math
import time
import cv2
import mediapipe as mp


DATA_FILE = "gesture_data.csv"
COOLDOWN_SECONDS = 1.0
UNKNOWN_THRESHOLD = 0.35


VARIABLE_MAP = {
    "ONE": "x",
    "TWO": "y",
    "THREE": "z",
}

NUMBER_MAP = {
    "ONE": 1,
    "TWO": 2,
    "THREE": 3,
    "FOUR": 4,
    "FIVE": 5,
}


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
            lm.z - base_z,
        ])

    return normalized


def load_gesture_data():
    samples = []

    with open(DATA_FILE, "r") as file:
        reader = csv.reader(file)
        next(reader)

        for row in reader:
            label = row[0].strip().upper()
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

    if best_distance > UNKNOWN_THRESHOLD:
        return "UNKNOWN", best_distance

    return best_label, best_distance


class GesturaRuntime:
    def __init__(self):
        self.mode = "NORMAL"
        self.selected_variable = None
        self.number_total = 0
        self.variables = {}
        self.last_input_time = 0
        self.last_accepted = "NONE"
        self.message = "NORMAL MODE: Show SET to begin"

    def can_accept_input(self):
        return time.time() - self.last_input_time >= COOLDOWN_SECONDS

    def accept(self, gesture):
        if gesture == "UNKNOWN":
            return

        if not self.can_accept_input():
            return

        self.last_input_time = time.time()
        self.last_accepted = gesture
        self.process_gesture(gesture)

    def process_gesture(self, gesture):
        if self.mode == "NORMAL":
            self.handle_normal_mode(gesture)

        elif self.mode == "SET_MODE":
            self.handle_set_mode(gesture)

        elif self.mode == "WAIT_FOR_NUMBER":
            self.handle_wait_for_number(gesture)

        elif self.mode == "NUMBER_MODE":
            self.handle_number_mode(gesture)

    def handle_normal_mode(self, gesture):
        if gesture == "SET":
            self.mode = "SET_MODE"
            self.selected_variable = None
            self.number_total = 0
            self.message = "SET MODE: Pick a variable (ONE = x, TWO = y, THREE = z)"
        else:
            self.message = "NORMAL MODE: Show SET to begin"

    def handle_set_mode(self, gesture):
        if gesture in VARIABLE_MAP:
            self.selected_variable = VARIABLE_MAP[gesture]
            self.mode = "WAIT_FOR_NUMBER"
            self.message = f"Selected variable {self.selected_variable}. Show NUMBER to enter value."
        elif gesture == "OK":
            self.reset()
        else:
            self.message = "SET MODE: Pick a variable number"

    def handle_wait_for_number(self, gesture):
        if gesture == "NUMBER":
            self.mode = "NUMBER_MODE"
            self.number_total = 0
            self.message = "NUMBER MODE: Show numbers. OK saves the total."
        elif gesture == "OK":
            self.reset()
        else:
            self.message = "WAITING: Show NUMBER gesture to start value input"

    def handle_number_mode(self, gesture):
        if gesture in NUMBER_MAP:
            self.number_total += NUMBER_MAP[gesture]
            self.message = f"NUMBER MODE: Current total = {self.number_total}. Press OK to save."

        elif gesture == "OK":
            if self.selected_variable is not None:
                self.variables[self.selected_variable] = self.number_total
                self.message = (
                    f"Executed: SET({self.selected_variable}, NUMBER({self.number_total})) OK"
                )
                print(self.message)
                print("Variables:", self.variables)

            self.mode = "NORMAL"
            self.selected_variable = None
            self.number_total = 0

        else:
            self.message = "NUMBER MODE: Show ONE-FIVE or OK"

    def reset(self):
        self.mode = "NORMAL"
        self.selected_variable = None
        self.number_total = 0
        self.message = "Reset. NORMAL MODE: Show SET to begin"

    def get_status_lines(self):
        return [
            f"Mode: {self.mode}",
            f"Last accepted: {self.last_accepted}",
            f"Variable: {self.selected_variable}",
            f"Number total: {self.number_total}",
            f"Variables: {self.variables}",
            self.message,
        ]


def draw_status(frame, lines):
    y = 35

    for line in lines:
        cv2.putText(
            frame,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
        )
        y += 30


def main():
    samples = load_gesture_data()
    runtime = GesturaRuntime()

    print(f"Loaded {len(samples)} gesture samples.")
    print("Press Q to quit.")
    print("Press R to reset.")

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
        min_tracking_confidence=0.7,
    ) as hands:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Could not read camera frame.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            predicted_label = "NO HAND"
            predicted_distance = 0

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                current_landmarks = normalize_landmarks(hand_landmarks)

                predicted_label, predicted_distance = predict_gesture(
                    current_landmarks,
                    samples,
                )

                runtime.accept(predicted_label)

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                )

            status_lines = [
                f"Detected: {predicted_label}",
                f"Distance: {predicted_distance:.3f}",
            ] + runtime.get_status_lines()

            draw_status(frame, status_lines)

            cv2.imshow("Gestura Live Runtime", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("r"):
                runtime.reset()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
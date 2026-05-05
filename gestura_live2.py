import csv
import math
import time
import cv2
import mediapipe as mp
import numpy as np


DATA_FILE = "gesture_data.csv"
COOLDOWN_SECONDS = 1.0
UNKNOWN_THRESHOLD = 0.35

CAMERA_WIDTH = 1100
CAMERA_HEIGHT = 750

VARIABLE_MAP = {
    "ONE": "x",
    "TWO": "y",
    "THREE": "z",
    "FOUR": "w",
    "FIVE": "v",
}

LETTER_GESTURES = {
    "A", "B", "C", "D", "E", "F", "G", "H", "I",
    "J", "K", "L", "M", "N", "O", "P", "Q", "R",
    "S", "T", "U", "V", "W", "X", "Y", "Z"
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
        self.gestura_code_lines = []
        self.string_buffer = ""
        self.output_lines = []

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

        elif self.mode == "PRINT_MODE":
            self.handle_print_mode(gesture)

        elif self.mode == "PRINT_VARIABLE_MODE":
            self.handle_print_variable_mode(gesture)

        elif self.mode == "STRING_MODE":
            self.handle_string_mode(gesture)

    def handle_normal_mode(self, gesture):
        if gesture == "SET":
            self.mode = "SET_MODE"
            self.selected_variable = None
            self.number_total = 0
            self.message = "SET MODE: Pick a variable (ONE = x, TWO = y, THREE = z)"

        elif gesture == "PRINT":
            self.mode = "PRINT_MODE"
            self.message = "PRINT MODE: ONE = variable, TWO = string"

        else:
            self.message = "NORMAL MODE: Show SET or PRINT"

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

    def handle_print_mode(self, gesture):
        if gesture == "ONE":
            self.mode = "PRINT_VARIABLE_MODE"
            self.message = "PRINT VARIABLE MODE: Pick variable (ONE = x, TWO = y, THREE = z, FOUR = w, FIVE = v)"

        elif gesture == "TWO":
            self.mode = "STRING_MODE"
            self.string_buffer = ""
            self.message = "STRING MODE: Use ASL letters. OK prints string."

        elif gesture == "OK":
            self.reset()

        else:
            self.message = "PRINT MODE: ONE = variable, TWO = string"


    def handle_print_variable_mode(self, gesture):
        if gesture in VARIABLE_MAP:
            variable_name = VARIABLE_MAP[gesture]

            if variable_name in self.variables:
                value = self.variables[variable_name]
                output = str(value)

                gestura_line = f"PRINT({variable_name.upper()}) OK"
                python_line = f"print({variable_name})"

                self.gestura_code_lines.append(gestura_line)
                self.gestura_code_lines.append("Python: " + python_line)
                self.output_lines.append(output)

                self.message = f"Output: {output}"
                print(output)

            else:
                self.message = f"Variable {variable_name} does not exist yet."

            self.mode = "NORMAL"

        elif gesture == "OK":
            self.reset()

        else:
            self.message = "PRINT VARIABLE MODE: Pick ONE-FIVE"


    def handle_string_mode(self, gesture):
        if gesture in LETTER_GESTURES:
            self.string_buffer += gesture
            self.message = f"STRING MODE: Current string = {self.string_buffer}. OK prints."

        elif gesture == "OK":
            output = self.string_buffer

            gestura_line = f'PRINT("{output}") OK'
            python_line = f'print("{output}")'

            self.gestura_code_lines.append(gestura_line)
            self.gestura_code_lines.append("Python: " + python_line)
            self.output_lines.append(output)

            self.message = f"Output: {output}"
            print(output)

            self.string_buffer = ""
            self.mode = "NORMAL"

        else:
            self.message = "STRING MODE: Use ASL letters or OK"
    
    def handle_number_mode(self, gesture):
        if gesture in NUMBER_MAP:
            self.number_total += NUMBER_MAP[gesture]
            self.message = f"NUMBER MODE: Current total = {self.number_total}. Press OK to save."

        elif gesture == "OK":
            if self.selected_variable is not None:
                self.variables[self.selected_variable] = self.number_total

                gestura_line = f"SET({self.selected_variable.upper()}, NUMBER({self.number_total})) OK"
                python_line = f"{self.selected_variable} = {self.number_total}"

                self.gestura_code_lines.append(gestura_line)
                self.gestura_code_lines.append("Python: " + python_line)

                self.message = f"Executed: {gestura_line}"
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
        lines = [
            f"Mode: {self.mode}",
            f"Last accepted: {self.last_accepted}",
            f"Variable: {self.selected_variable}",
        ]

        if self.mode == "NUMBER_MODE":
            lines.append(f"Number total: {self.number_total}")

        if self.mode == "STRING_MODE":
            lines.append(f"String: {self.string_buffer}")

        lines.append(self.message)
        return lines


def draw_top_status(frame, lines):
    y = 35
    for line in lines:
        cv2.putText(
            frame,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        y += 32


def draw_bottom_detection(frame, detected, distance_value):
    h, w, _ = frame.shape

    cv2.putText(
        frame,
        f"Detected: {detected}",
        (10, h - 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        frame,
        f"Distance: {distance_value:.3f}",
        (10, h - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )


def draw_code_window(runtime):
    canvas = np.zeros((500, 800, 3), dtype=np.uint8)
    
    cv2.putText(
        canvas,
        f"Output: {runtime.output_lines[-1] if runtime.output_lines else ''}",
        (20, 455),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        canvas,
        "Gestura Code Window",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
    )

    cv2.putText(
        canvas,
        "Live code generated from gestures:",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    y = 140

    if not runtime.gestura_code_lines:
        cv2.putText(
            canvas,
            "No code written yet.",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (180, 180, 180),
            2,
        )
    else:
        for line in runtime.gestura_code_lines[-12:]:
            cv2.putText(
                canvas,
                line,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            y += 35

    y = 420
    cv2.putText(
        canvas,
        f"Variables: {runtime.variables}",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    cv2.imshow("Gestura Code", canvas)


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

    cv2.namedWindow("Gestura Camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Gestura Camera", CAMERA_WIDTH, CAMERA_HEIGHT)

    cv2.namedWindow("Gestura Code", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Gestura Code", 800, 500)

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
            frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT))

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

            draw_top_status(frame, runtime.get_status_lines())
            draw_bottom_detection(frame, predicted_label, predicted_distance)
            draw_code_window(runtime)

            cv2.imshow("Gestura Camera", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("r"):
                runtime.reset()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
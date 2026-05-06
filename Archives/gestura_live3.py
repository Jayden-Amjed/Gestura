import csv
import math
import time
import cv2
import mediapipe as mp
import numpy as np


DATA_FILE = "gesture_data.csv"
COOLDOWN_SECONDS = 3.0
UNKNOWN_THRESHOLD = 0.35

CAMERA_WIDTH = 850
CAMERA_HEIGHT = 750

VARIABLE_MAP = {
    "ONE": "x",
    "TWO": "y",
    "THREE": "z",
    "FOUR": "w",
    "FIVE": "v",
}

NUMBER_MAP = {
    "ONE": 1,
    "TWO": 2,
    "THREE": 3,
    "FOUR": 4,
    "FIVE": 5,
}

LETTER_MAP = {
    "A": "A",
    "FOUR": "B",
    "C": "C",
    "D": "D",
    "E": "E",
    "THREE": "F",
    "G": "G",
    "H": "H",
    "I": "I",
    "J": "J",
    "K": "K",
    "L": "L",
    "M": "M",
    "N": "N",
    "O": "O",
    "P": "P",
    "Q": "Q",
    "R": "R",
    "S": "S",
    "T": "T",
    "TWO": "U",
    "V": "V",
    "W": "W",
    "X": "X",
    "Y": "Y",
    "Z": "Z",
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
        self.string_buffer = ""

        self.last_input_time = 0
        self.last_accepted = "NONE"

        self.message = "NORMAL MODE: Show SET, PRINT, or EXECUTE"

        self.program_commands = []
        self.gestura_code_lines = []
        self.output_lines = []
        self.variables = {}

    def time_remaining(self):
        remaining = COOLDOWN_SECONDS - (time.time() - self.last_input_time)
        return max(0, remaining)

    def can_accept_input(self):
        return time.time() - self.last_input_time >= COOLDOWN_SECONDS

    def accept(self, gesture):
        if gesture == "UNKNOWN" or gesture == "NO HAND":
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

        elif self.mode == "SET_STRING_MODE":
            self.handle_set_string_mode(gesture)

    def handle_normal_mode(self, gesture):
        if gesture == "SET":
            self.mode = "SET_MODE"
            self.selected_variable = None
            self.number_total = 0
            self.message = "SET MODE: Pick variable (ONE=x, TWO=y, THREE=z, FOUR=w, FIVE=v)"

        elif gesture == "PRINT":
            self.mode = "PRINT_MODE"
            self.message = "PRINT MODE: ONE = variable, TWO = string"

        elif gesture == "EXECUTE":
            self.execute_program()

        elif gesture == "CLEAR":
            self.clear_program()

        else:
            self.message = "NORMAL MODE: Show SET, PRINT, or EXECUTE"

    def handle_set_mode(self, gesture):
        if gesture in VARIABLE_MAP:
            self.selected_variable = VARIABLE_MAP[gesture]
            self.mode = "WAIT_FOR_NUMBER"
            self.message = f"Selected variable {self.selected_variable}. Show NUMBER."

        elif gesture == "OK":
            self.reset_current_input()

        else:
            self.message = "SET MODE: Pick ONE-FIVE for variable"
    
    def handle_set_string_mode(self, gesture):
        if gesture in LETTER_MAP:
            letter = LETTER_MAP[gesture]
            self.string_buffer += letter
            self.message = f"SET STRING MODE: Current string = {self.string_buffer}. OK saves."

        elif gesture == "OK":
            if self.selected_variable is not None:
                command = {
                    "type": "SET",
                    "variable": self.selected_variable,
                    "value": self.string_buffer,
                }

                self.program_commands.append(command)

                gestura_line = f'SET({self.selected_variable.upper()}, STRING("{self.string_buffer}")) OK'
                python_line = f'{self.selected_variable} = "{self.string_buffer}"'

                self.gestura_code_lines.append(gestura_line)
                self.gestura_code_lines.append("Python: " + python_line)

                self.message = f"Added command: {gestura_line}"

            self.string_buffer = ""
            self.reset_current_input()

        else:
            self.message = "SET STRING MODE: Use letters or OK"

    def handle_wait_for_number(self, gesture):
        if gesture == "NUMBER":
            self.mode = "NUMBER_MODE"
            self.number_total = 0
            self.message = "NUMBER MODE: Show number gestures. OK saves command."

        elif gesture == "PRINT":
            self.mode = "SET_STRING_MODE"
            self.string_buffer = ""
            self.message = "SET STRING MODE: Use letter gestures. OK saves string variable."

        elif gesture == "OK":
            self.reset_current_input()

        else:
            self.message = "WAITING: Show NUMBER for number value or PRINT for string value"

    def handle_number_mode(self, gesture):
        if gesture in NUMBER_MAP:
            self.number_total += NUMBER_MAP[gesture]
            self.message = f"NUMBER MODE: Current total = {self.number_total}. OK saves."

        elif gesture == "OK":
            if self.selected_variable is not None:
                command = {
                    "type": "SET",
                    "variable": self.selected_variable,
                    "value": self.number_total,
                }

                self.program_commands.append(command)

                gestura_line = f"SET({self.selected_variable.upper()}, NUMBER({self.number_total})) OK"
                python_line = f"{self.selected_variable} = {self.number_total}"

                self.gestura_code_lines.append(gestura_line)
                self.gestura_code_lines.append("Python: " + python_line)

                self.message = f"Added command: {gestura_line}"

            self.reset_current_input()

        else:
            self.message = "NUMBER MODE: Show ONE-FIVE or OK"

    def handle_print_mode(self, gesture):
        if gesture == "ONE":
            self.mode = "PRINT_VARIABLE_MODE"
            self.message = "PRINT VARIABLE MODE: Pick variable ONE-FIVE"

        elif gesture == "TWO":
            self.mode = "STRING_MODE"
            self.string_buffer = ""
            self.message = "STRING MODE: Use letter gestures. OK saves print string."

        elif gesture == "OK":
            self.reset_current_input()

        else:
            self.message = "PRINT MODE: ONE = variable, TWO = string"

    def handle_print_variable_mode(self, gesture):
        if gesture in VARIABLE_MAP:
            variable_name = VARIABLE_MAP[gesture]

            command = {
                "type": "PRINT_VAR",
                "variable": variable_name,
            }

            self.program_commands.append(command)

            gestura_line = f"PRINT({variable_name.upper()}) OK"
            python_line = f"print({variable_name})"

            self.gestura_code_lines.append(gestura_line)
            self.gestura_code_lines.append("Python: " + python_line)

            self.message = f"Added command: {gestura_line}"
            self.reset_current_input()

        elif gesture == "OK":
            self.reset_current_input()

        else:
            self.message = "PRINT VARIABLE MODE: Pick ONE-FIVE"

    def handle_string_mode(self, gesture):
        if gesture in LETTER_MAP:
            letter = LETTER_MAP[gesture]
            self.string_buffer += letter
            self.message = f"STRING MODE: Current string = {self.string_buffer}. OK saves."

        elif gesture == "OK":
            command = {
                "type": "PRINT_STRING",
                "value": self.string_buffer,
            }

            self.program_commands.append(command)

            gestura_line = f'PRINT("{self.string_buffer}") OK'
            python_line = f'print("{self.string_buffer}")'

            self.gestura_code_lines.append(gestura_line)
            self.gestura_code_lines.append("Python: " + python_line)

            self.message = f"Added command: {gestura_line}"
            self.string_buffer = ""
            self.reset_current_input()

        else:
            self.message = "STRING MODE: Use letters or OK"

    def execute_program(self):
        self.output_lines = []
        self.variables = {}

        for command in self.program_commands:
            if command["type"] == "SET":
                self.variables[command["variable"]] = command["value"]

            elif command["type"] == "PRINT_VAR":
                variable_name = command["variable"]

                if variable_name in self.variables:
                    self.output_lines.append(str(self.variables[variable_name]))
                else:
                    self.output_lines.append(f"Error: {variable_name} is undefined")

            elif command["type"] == "PRINT_STRING":
                self.output_lines.append(command["value"])

        self.message = "Program executed. Check Interpreted Output window."

    def clear_program(self):
        self.program_commands = []
        self.gestura_code_lines = []
        self.output_lines = []
        self.variables = {}
        self.reset_current_input()
        self.message = "Program cleared."

    def reset_current_input(self):
        self.mode = "NORMAL"
        self.selected_variable = None
        self.number_total = 0
        self.string_buffer = ""

    def get_status_lines(self):
        lines = [
            f"Mode: {self.mode}",
            f"Last accepted: {self.last_accepted}",
        ]

        if self.selected_variable is not None:
            lines.append(f"Variable: {self.selected_variable}")

        if self.mode == "NUMBER_MODE":
            lines.append(f"Number total: {self.number_total}")

        if self.mode in ["STRING_MODE", "SET_STRING_MODE"]:
            lines.append(f"String: {self.string_buffer}")

        remaining = self.time_remaining()
        if remaining > 0:
            lines.append(f"Cooldown: {remaining:.1f}s")

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
    h, _, _ = frame.shape

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
        "Gestura Code",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
    )

    cv2.putText(
        canvas,
        "Code being built:",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    y = 135

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
                0.65,
                (0, 255, 0),
                2,
            )
            y += 32

    cv2.putText(
        canvas,
        "Use EXECUTE gesture to run program.",
        (20, 470),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )

    cv2.imshow("Gestura Code", canvas)


def draw_output_window(runtime):
    canvas = np.zeros((500, 800, 3), dtype=np.uint8)

    cv2.putText(
        canvas,
        "Interpreted Output",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
    )

    cv2.putText(
        canvas,
        "Output appears after EXECUTE:",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    y = 135

    if not runtime.output_lines:
        cv2.putText(
            canvas,
            "No output yet.",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (180, 180, 180),
            2,
        )
    else:
        for line in runtime.output_lines[-12:]:
            cv2.putText(
                canvas,
                str(line),
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            y += 35

    cv2.putText(
        canvas,
        f"Variables after run: {runtime.variables}",
        (20, 455),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )

    cv2.imshow("Interpreted Output", canvas)


def main():
    samples = load_gesture_data()
    runtime = GesturaRuntime()

    print(f"Loaded {len(samples)} gesture samples.")
    print("Press Q to quit.")
    print("Press R to reset current input.")
    print("Use CLEAR gesture to clear whole program if trained.")

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

    cv2.namedWindow("Interpreted Output", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Interpreted Output", 800, 500)

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
            draw_output_window(runtime)

            cv2.imshow("Gestura Camera", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("r"):
                runtime.reset_current_input()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
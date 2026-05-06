import csv
import math
import time
import cv2
import mediapipe as mp
import numpy as np


DATA_FILE = "gesture_data.csv"
COOLDOWN_SECONDS = 3.0
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

NUMBER_MAP = {
    "ONE": 1,
    "TWO": 2,
    "THREE": 3,
    "FOUR": 4,
    "FIVE": 5,
}

LOOP_MAP = {
    "ONE": "FOR",
    "TWO": "WHILE",
}

MATH_OPERATOR_MAP = {
    "ONE": "+",
    "TWO": "-",
    "THREE": "*",
    "FOUR": "/",
    "FIVE": "%",
}

EQUALITY_OPERATOR_MAP = {
    "ONE": "<=",
    "TWO": ">=",
    "THREE": "==",
}

COMPARISON_OPERATOR_MAP = {
    "ONE": "<",
    "TWO": ">",
}

INC_DEC_OPERATOR_MAP = {
    "ONE": "++",
    "TWO": "--",
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

        self.loop_type = None
        self.condition_parts = []
        self.condition_context = None

        self.last_input_time = 0
        self.last_accepted = "NONE"

        self.message = "NORMAL MODE: Show SET, PRINT, LOOP, CONDITIONAL_STATEMENT, END, or EXECUTE"

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

        elif self.mode == "WAIT_FOR_VALUE":
            self.handle_wait_for_value(gesture)

        elif self.mode == "NUMBER_MODE":
            self.handle_number_mode(gesture)

        elif self.mode == "SET_STRING_MODE":
            self.handle_set_string_mode(gesture)

        elif self.mode == "PRINT_MODE":
            self.handle_print_mode(gesture)

        elif self.mode == "PRINT_VARIABLE_MODE":
            self.handle_print_variable_mode(gesture)

        elif self.mode == "STRING_MODE":
            self.handle_string_mode(gesture)

        elif self.mode == "LOOP_MODE":
            self.handle_loop_mode(gesture)

        elif self.mode == "CONDITION_VALUE_TYPE_MODE":
            self.handle_condition_value_type_mode(gesture)

        elif self.mode == "CONDITION_VARIABLE_MODE":
            self.handle_condition_variable_mode(gesture)

        elif self.mode == "CONDITION_NUMBER_MODE":
            self.handle_condition_number_mode(gesture)

        elif self.mode == "CONDITION_OPERATOR_TYPE_MODE":
            self.handle_condition_operator_type_mode(gesture)

        elif self.mode == "MATH_OPERATOR_MODE":
            self.handle_math_operator_mode(gesture)

        elif self.mode == "EQUALITY_OPERATOR_MODE":
            self.handle_equality_operator_mode(gesture)

        elif self.mode == "COMPARISON_OPERATOR_MODE":
            self.handle_comparison_operator_mode(gesture)

        elif self.mode == "INC_DEC_VARIABLE_MODE":
            self.handle_inc_dec_variable_mode(gesture)

        elif self.mode == "INC_DEC_OPERATOR_MODE":
            self.handle_inc_dec_operator_mode(gesture)

    def handle_normal_mode(self, gesture):
        if gesture == "SET":
            self.mode = "SET_MODE"
            self.selected_variable = None
            self.number_total = 0
            self.message = "SET MODE: Pick variable ONE-FIVE"

        elif gesture == "PRINT":
            self.mode = "PRINT_MODE"
            self.message = "PRINT MODE: ONE = variable, TWO = string"

        elif gesture == "LOOP":
            self.mode = "LOOP_MODE"
            self.loop_type = None
            self.condition_parts = []
            self.condition_context = "LOOP"
            self.message = "LOOP MODE: ONE = for loop, TWO = while loop"

        elif gesture == "CONDITIONAL_STATEMENT":
            self.mode = "CONDITION_VALUE_TYPE_MODE"
            self.condition_parts = []
            self.condition_context = "IF"
            self.message = "IF MODE: Choose first value. ONE = variable, TWO = number"

        elif gesture == "INC_DEC_OPERATOR":
            self.mode = "INC_DEC_VARIABLE_MODE"
            self.message = "INC/DEC MODE: Pick variable ONE-FIVE"

        elif gesture == "END":
            self.add_end_command()

        elif gesture == "EXECUTE":
            self.execute_program()

        elif gesture == "CLEAR":
            self.clear_program()

        else:
            self.message = "NORMAL MODE: Show SET, PRINT, LOOP, IF, INC/DEC, END, or EXECUTE"

    def handle_set_mode(self, gesture):
        if gesture in VARIABLE_MAP:
            self.selected_variable = VARIABLE_MAP[gesture]
            self.mode = "WAIT_FOR_VALUE"
            self.message = f"Selected variable {self.selected_variable}. NUMBER = number, PRINT = string"

        elif gesture == "OK":
            self.reset_current_input()

        else:
            self.message = "SET MODE: Pick variable ONE-FIVE"

    def handle_wait_for_value(self, gesture):
        if gesture == "NUMBER":
            self.mode = "NUMBER_MODE"
            self.number_total = 0
            self.message = "NUMBER MODE: Show number gestures. OK saves number variable."

        elif gesture == "PRINT":
            self.mode = "SET_STRING_MODE"
            self.string_buffer = ""
            self.message = "SET STRING MODE: Use letter gestures. OK saves string variable."

        elif gesture == "OK":
            self.reset_current_input()

        else:
            self.message = "WAITING: Show NUMBER for number or PRINT for string"

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

                indent = self.current_indent()
                gestura_line = f"SET({self.selected_variable.upper()}, NUMBER({self.number_total})) OK"
                python_line = f"{self.selected_variable} = {self.number_total}"

                self.gestura_code_lines.append(("    " * indent) + gestura_line)
                self.gestura_code_lines.append(("    " * indent) + "Python: " + python_line)

                self.message = f"Added command: {gestura_line}"

            self.reset_current_input()

        else:
            self.message = "NUMBER MODE: Show ONE-FIVE or OK"

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

                indent = self.current_indent()
                gestura_line = f'SET({self.selected_variable.upper()}, STRING("{self.string_buffer}")) OK'
                python_line = f'{self.selected_variable} = "{self.string_buffer}"'

                self.gestura_code_lines.append(("    " * indent) + gestura_line)
                self.gestura_code_lines.append(("    " * indent) + "Python: " + python_line)

                self.message = f"Added command: {gestura_line}"

            self.string_buffer = ""
            self.reset_current_input()

        else:
            self.message = "SET STRING MODE: Use letters or OK"

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

            indent = self.current_indent()
            gestura_line = f"PRINT({variable_name.upper()}) OK"
            python_line = f"print({variable_name})"

            self.gestura_code_lines.append(("    " * indent) + gestura_line)
            self.gestura_code_lines.append(("    " * indent) + "Python: " + python_line)

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

            indent = self.current_indent()
            gestura_line = f'PRINT("{self.string_buffer}") OK'
            python_line = f'print("{self.string_buffer}")'

            self.gestura_code_lines.append(("    " * indent) + gestura_line)
            self.gestura_code_lines.append(("    " * indent) + "Python: " + python_line)

            self.message = f"Added command: {gestura_line}"
            self.string_buffer = ""
            self.reset_current_input()

        else:
            self.message = "STRING MODE: Use letters or OK"

    def handle_loop_mode(self, gesture):
        if gesture in LOOP_MAP:
            self.loop_type = LOOP_MAP[gesture]
            self.mode = "CONDITION_VALUE_TYPE_MODE"
            self.condition_context = "LOOP"
            self.condition_parts = []
            self.message = f"{self.loop_type} LOOP: Choose first value. ONE = variable, TWO = number"

        elif gesture == "OK":
            self.reset_current_input()

        else:
            self.message = "LOOP MODE: ONE = for loop, TWO = while loop"

    def handle_condition_value_type_mode(self, gesture):
        if gesture == "ONE":
            self.mode = "CONDITION_VARIABLE_MODE"
            self.message = "VARIABLE MODE: Pick variable ONE-FIVE"

        elif gesture == "TWO":
            self.mode = "CONDITION_NUMBER_MODE"
            self.number_total = 0
            self.message = "NUMBER MODE: Add number gestures. OK saves number."

        elif gesture == "OK":
            self.finish_condition_header()

        else:
            self.message = "Choose value first: ONE=variable or TWO=number"

    def handle_condition_variable_mode(self, gesture):
        if gesture in VARIABLE_MAP:
            variable_name = VARIABLE_MAP[gesture]
            self.condition_parts.append(variable_name)
            self.mode = "CONDITION_OPERATOR_TYPE_MODE"
            self.message = "Choose operator: MATH_OPERATOR, EQUALITY_OPERATOR, COMPARISON_OPERATOR, or OK"

        elif gesture == "OK":
            self.finish_condition_header()

        else:
            self.message = "VARIABLE MODE: Pick ONE-FIVE"

    def handle_condition_number_mode(self, gesture):
        if gesture in NUMBER_MAP:
            self.number_total += NUMBER_MAP[gesture]
            self.message = f"NUMBER MODE: Current number = {self.number_total}. OK saves."

        elif gesture == "OK":
            self.condition_parts.append(str(self.number_total))
            self.number_total = 0
            self.mode = "CONDITION_OPERATOR_TYPE_MODE"
            self.message = "Choose operator: MATH_OPERATOR, EQUALITY_OPERATOR, COMPARISON_OPERATOR, or OK"

        else:
            self.message = "NUMBER MODE: Show ONE-FIVE or OK"

    def handle_condition_operator_type_mode(self, gesture):
        if gesture == "MATH_OPERATOR":
            self.mode = "MATH_OPERATOR_MODE"
            self.message = "MATH OPERATOR: ONE=+, TWO=-, THREE=*, FOUR=/, FIVE=%"

        elif gesture == "EQUALITY_OPERATOR":
            self.mode = "EQUALITY_OPERATOR_MODE"
            self.message = "EQUALITY OPERATOR: ONE=<=, TWO=>=, THREE==="

        elif gesture == "COMPARISON_OPERATOR":
            self.mode = "COMPARISON_OPERATOR_MODE"
            self.message = "COMPARISON OPERATOR: ONE=<, TWO=>"

        elif gesture == "OK":
            self.finish_condition_header()

        else:
            self.message = "Choose MATH_OPERATOR, EQUALITY_OPERATOR, COMPARISON_OPERATOR, or OK"

    def handle_math_operator_mode(self, gesture):
        if gesture in MATH_OPERATOR_MAP:
            op = MATH_OPERATOR_MAP[gesture]
            self.condition_parts.append(op)
            self.mode = "CONDITION_VALUE_TYPE_MODE"
            self.message = "Math operator added. Choose next value: ONE=variable, TWO=number"

        elif gesture == "OK":
            self.finish_condition_header()

        else:
            self.message = "MATH OPERATOR: ONE=+, TWO=-, THREE=*, FOUR=/, FIVE=%"

    def handle_equality_operator_mode(self, gesture):
        if gesture in EQUALITY_OPERATOR_MAP:
            op = EQUALITY_OPERATOR_MAP[gesture]
            self.condition_parts.append(op)
            self.mode = "CONDITION_VALUE_TYPE_MODE"
            self.message = "Equality operator added. Choose next value: ONE=variable, TWO=number"

        elif gesture == "OK":
            self.finish_condition_header()

        else:
            self.message = "EQUALITY OPERATOR: ONE=<=, TWO=>=, THREE==="

    def handle_comparison_operator_mode(self, gesture):
        if gesture in COMPARISON_OPERATOR_MAP:
            op = COMPARISON_OPERATOR_MAP[gesture]
            self.condition_parts.append(op)
            self.mode = "CONDITION_VALUE_TYPE_MODE"
            self.message = "Comparison operator added. Choose next value: ONE=variable, TWO=number"

        elif gesture == "OK":
            self.finish_condition_header()

        else:
            self.message = "COMPARISON OPERATOR: ONE=<, TWO=>"

    def finish_condition_header(self):
        condition = " ".join(self.condition_parts)

        if not condition:
            self.message = "Condition incomplete."
            self.reset_current_input()
            return

        indent = self.current_indent()

        if self.condition_context == "LOOP":
            if not self.loop_type:
                self.message = "Loop header incomplete."
                self.reset_current_input()
                return

            command = {
                "type": "LOOP_START",
                "loop_type": self.loop_type,
                "condition": condition,
            }

            self.program_commands.append(command)

            gestura_line = f"{self.loop_type}({condition}) OK"

            if self.loop_type == "WHILE":
                python_line = f"while {condition}:"
            else:
                python_line = f"# for loop condition: {condition}"

            self.gestura_code_lines.append(("    " * indent) + gestura_line)
            self.gestura_code_lines.append(("    " * indent) + "Python: " + python_line)

            self.message = f"Added loop header: {gestura_line}"

        elif self.condition_context == "IF":
            command = {
                "type": "IF_START",
                "condition": condition,
            }

            self.program_commands.append(command)

            gestura_line = f"IF({condition}) OK"
            python_line = f"if {condition}:"

            self.gestura_code_lines.append(("    " * indent) + gestura_line)
            self.gestura_code_lines.append(("    " * indent) + "Python: " + python_line)

            self.message = f"Added IF statement: {gestura_line}"

        self.reset_current_input()

    def handle_inc_dec_variable_mode(self, gesture):
        if gesture in VARIABLE_MAP:
            self.selected_variable = VARIABLE_MAP[gesture]
            self.mode = "INC_DEC_OPERATOR_MODE"
            self.message = "INC/DEC OPERATOR: ONE = increment, TWO = decrement"

        elif gesture == "OK":
            self.reset_current_input()

        else:
            self.message = "INC/DEC MODE: Pick variable ONE-FIVE"

    def handle_inc_dec_operator_mode(self, gesture):
        if gesture in INC_DEC_OPERATOR_MAP:
            operator = INC_DEC_OPERATOR_MAP[gesture]

            command = {
                "type": "INC_DEC",
                "variable": self.selected_variable,
                "operator": operator,
            }

            self.program_commands.append(command)

            if operator == "++":
                gestura_line = f"INC({self.selected_variable.upper()}) OK"
                python_line = f"{self.selected_variable} += 1"
            else:
                gestura_line = f"DEC({self.selected_variable.upper()}) OK"
                python_line = f"{self.selected_variable} -= 1"

            indent = self.current_indent()
            self.gestura_code_lines.append(("    " * indent) + gestura_line)
            self.gestura_code_lines.append(("    " * indent) + "Python: " + python_line)

            self.message = f"Added command: {gestura_line}"
            self.reset_current_input()

        elif gesture == "OK":
            self.reset_current_input()

        else:
            self.message = "INC/DEC OPERATOR: ONE = increment, TWO = decrement"

    def add_end_command(self):
        self.program_commands.append({"type": "END"})

        indent = max(0, self.current_indent() - 1)
        self.gestura_code_lines.append(("    " * indent) + "END")

        self.message = "Added END block."

    def current_indent(self):
        indent = 0

        for command in self.program_commands:
            if command["type"] in ["LOOP_START", "IF_START"]:
                indent += 1
            elif command["type"] == "END":
                indent = max(0, indent - 1)

        return indent

    def execute_program(self):
        self.output_lines = []
        self.variables = {}

        try:
            self.run_block(0, len(self.program_commands), max_loop_count=100)
            self.message = "Program executed. Check Interpreted Output window."
        except Exception as error:
            self.output_lines.append(f"Runtime error: {error}")
            self.message = "Runtime error. Check Interpreted Output window."

    def run_block(self, start_index, end_index, max_loop_count=100):
        i = start_index

        while i < end_index:
            command = self.program_commands[i]

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

            elif command["type"] == "INC_DEC":
                variable_name = command["variable"]

                if variable_name not in self.variables:
                    self.output_lines.append(f"Error: {variable_name} is undefined")
                elif command["operator"] == "++":
                    self.variables[variable_name] += 1
                elif command["operator"] == "--":
                    self.variables[variable_name] -= 1

            elif command["type"] == "IF_START":
                block_end = self.find_matching_end(i)

                if block_end is None:
                    self.output_lines.append("Error: IF missing END")
                    return

                if self.evaluate_condition(command["condition"]):
                    self.run_block(i + 1, block_end, max_loop_count)

                i = block_end

            elif command["type"] == "LOOP_START":
                loop_end = self.find_matching_end(i)

                if loop_end is None:
                    self.output_lines.append("Error: LOOP missing END")
                    return

                if command["loop_type"] == "WHILE":
                    loop_count = 0

                    while self.evaluate_condition(command["condition"]):
                        self.run_block(i + 1, loop_end, max_loop_count)
                        loop_count += 1

                        if loop_count >= max_loop_count:
                            self.output_lines.append("Loop stopped: max loop count reached")
                            break

                else:
                    self.output_lines.append("FOR loop stored, but execution is not implemented yet.")

                i = loop_end

            elif command["type"] == "END":
                return

            i += 1

    def find_matching_end(self, start_index):
        depth = 0

        for i in range(start_index + 1, len(self.program_commands)):
            command_type = self.program_commands[i]["type"]

            if command_type in ["LOOP_START", "IF_START"]:
                depth += 1

            elif command_type == "END":
                if depth == 0:
                    return i
                depth -= 1

        return None

    def evaluate_condition(self, condition):
        allowed_names = {}

        for name, value in self.variables.items():
            allowed_names[name] = value

        return bool(eval(condition, {"__builtins__": {}}, allowed_names))

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
        self.loop_type = None
        self.condition_parts = []
        self.condition_context = None

    def get_status_lines(self):
        lines = [
            f"Mode: {self.mode}",
            f"Last accepted: {self.last_accepted}",
        ]

        if self.selected_variable is not None:
            lines.append(f"Variable: {self.selected_variable}")

        if self.mode in ["NUMBER_MODE", "CONDITION_NUMBER_MODE"]:
            lines.append(f"Number total: {self.number_total}")

        if self.mode in ["STRING_MODE", "SET_STRING_MODE"]:
            lines.append(f"String: {self.string_buffer}")

        if self.condition_context is not None:
            lines.append(f"Context: {self.condition_context}")
            lines.append(f"Condition: {' '.join(self.condition_parts)}")

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
    canvas = np.zeros((600, 900, 3), dtype=np.uint8)

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
        for line in runtime.gestura_code_lines[-14:]:
            cv2.putText(
                canvas,
                line,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 255, 0),
                2,
            )
            y += 32

    cv2.putText(
        canvas,
        "Use EXECUTE gesture to run program.",
        (20, 565),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )

    cv2.imshow("Gestura Code", canvas)


def draw_output_window(runtime):
    canvas = np.zeros((600, 900, 3), dtype=np.uint8)

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
        for line in runtime.output_lines[-14:]:
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
        (20, 565),
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
    print("Use CLEAR gesture to clear the whole program if trained.")

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open camera.")
        return

    cv2.namedWindow("Gestura Camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Gestura Camera", CAMERA_WIDTH, CAMERA_HEIGHT)

    cv2.namedWindow("Gestura Code", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Gestura Code", 900, 600)

    cv2.namedWindow("Interpreted Output", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Interpreted Output", 900, 600)

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
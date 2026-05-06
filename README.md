# Gestura

Gestura is a gesture-based interpreted programming language that lets you write and execute code using hand gestures. It uses computer vision (OpenCV + MediaPipe) to recognize gestures, converts them into tokens, builds structured commands, and executes them through a custom runtime.

---

## Features

- Gesture-based programming (no keyboard required)
- Real-time hand tracking using OpenCV and MediaPipe
- Custom interpreter for executing Gestura programs
- Supports:
  - Variables (`SET`)
  - Output (`PRINT`)
  - Loops (`WHILE`)
  - Conditionals (`IF / ELIF / ELSE`)
  - Increment / Decrement operators
- Multi-window UI:
  - Camera view
  - Gestura code builder
  - Interpreted output display
- Ability to save and run `.gestura` files

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/gestura.git
cd gestura
```

### 2. Create a virtual environment
```bash
python -m venv .venv
```

### 3. Activate the virtual environment

Windows:
```bash
.venv\Scripts\activate
```

Linux / Mac:
```bash
source .venv/bin/activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Running Gestura (Camera Mode)

```bash
python gestura_main.py
```

This opens:
- Camera window (gesture detection)
- Gestura code window (program being built)
- Output window (program results)

---

## Running a `.gestura` file

```bash
python gestura_main.py your_program.gestura
```

This will:
- Skip camera input
- Execute the Gestura program
- Print output directly to the terminal

---

## Example Program

### `fizz_buzz.gestura`

```text
SET(X, NUMBER(1)) OK
WHILE(x <= 15) OK
IF(x % 15 == 0) OK
PRINT("FIZZBUZZ") OK
ELIF(x % 3 == 0) OK
PRINT("FIZZ") OK
ELIF(x % 5 == 0) OK
PRINT("BUZZ") OK
ELSE OK
PRINT(X) OK
END
INC(X) OK
END
```

### Run:
```bash
python gestura_main.py fizz_buzz.gestura
```

### Output:
```text
1
2
FIZZ
4
BUZZ
FIZZ
7
8
FIZZ
BUZZ
11
FIZZ
13
14
FIZZBUZZ
```

---


---

## Notes

- Works best in good lighting conditions
- Make sure your webcam is accessible
- Gesture accuracy depends on training data (`gesture_data.csv`)
- Cooldown system prevents accidental repeated inputs

---

## Authors

Samuel Gotama  
Jayden Breesam

---

## Future Improvements

- Better gesture classification (ML-based instead of distance matching)
- Support for more complex syntax
- UI improvements and syntax highlighting
- More efficient gesture recognition pipeline
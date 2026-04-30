# Gestura

Gestura is a gesture-based interpreted programming language where hand gestures are converted into tokens, parsed into command structures, and executed in real time.

---

## Setup Instructions

### 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/gestura.git
cd gestura

### 2. Install Python

Make sure you have Python 3.11 installed:
python3.11 --version

### 3. Create a virtual environment
python3.11 -m venv .venv

### 4. Activate the virtual environment
source .venv/bin/activate

You should now see:
(.venv)

### 5. Install dependencies
pip install -r requirements.txt

### 6. Run the project
python hand_tracking.py

Press q to close the camera window.

---

## Project Structure

gestura/
├── .venv/
├── main.py
├── hand_tracking.py
├── finger_count_test.py
├── requirements.txt
└── README.md

---

## Features

- Real-time hand tracking using OpenCV and MediaPipe
- Basic gesture detection
- Foundation for gesture-based programming

---

## Notes

- Make sure your camera is connected
- Works best in good lighting
- If camera doesn’t open, try changing:
  cv2.VideoCapture(0)
  to:
  cv2.VideoCapture(1)

---

## Authors

Samuel Gotama and Jayden Breesam
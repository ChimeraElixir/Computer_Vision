from imutils.video import VideoStream
import numpy as np
import imutils
import dlib
import cv2
import time
from imutils import face_utils


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("assets/shape_predictor_68_face_landmarks.dat")

# Create a mapping dictionary for landmark states
LANDMARK_MAPPING = {
    "m": "mouth",
    "i": "inner_mouth",
    "r": "right_eyebrow",
    "l": "left_eyebrow",
    "e": "right_eye",
    "f": "left_eye",
    "n": "nose",
    "j": "jaw",
}

# Define instructions once
INSTRUCTIONS = [f"{key} - {value}" for key, value in LANDMARK_MAPPING.items()]

# print("Select the landmarks you want to see:")
# for instruction in INSTRUCTIONS:
#     print(instruction)

print("[INFO] camera sensor warming up...")
vs = VideoStream(0).start()

landmark_states = None  # Initialize without redundant checks

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=1080)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clone = frame.copy()

    # Draw instructions
    y_offset = 0
    for instruction in INSTRUCTIONS:
        cv2.putText(
            clone,
            instruction,
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )
        y_offset += 20

    # Process faces
    for rect in detector(gray, 0):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        if landmark_states:
            (i, j) = face_utils.FACIAL_LANDMARKS_IDXS[landmark_states]

            # Display current landmark name
            cv2.putText(
                clone,
                f"Current: {landmark_states}",
                (10, y_offset + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )

            # Draw landmark points more efficiently
            points = shape[i:j]
            for x, y in points:
                cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)

    cv2.imshow("Frame", clone)

    key = chr(cv2.waitKey(1) & 0xFF)

    # Handle landmark state changes
    if key in LANDMARK_MAPPING:
        landmark_states = LANDMARK_MAPPING[key]
    elif key == "q":
        break

cv2.destroyAllWindows()
vs.stop()

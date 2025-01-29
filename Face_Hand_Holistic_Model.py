import cv2
import mediapipe as mp


mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()

    frame = cv2.resize(frame,(800,600))

    image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    results = holistic_model.process(image)

    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1),
    )

    mp_drawing.draw_landmarks(
        image,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1),
    )
    mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1),
    )

    cv2.imshow("frame",image)

    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()

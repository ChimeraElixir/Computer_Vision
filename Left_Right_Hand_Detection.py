import cv2
import mediapipe as mp
from google.protobuf.json_format import MessageToDict

mphands = mp.solutions.hands
hands = mphands.Hands(
    max_num_hands=2, model_complexity=1, min_detection_confidence=0.75, min_tracking_confidence=0.75
)

cap = cv2.VideoCapture(0)

while True:

    _, frame = cap.read()

    frame = cv2.flip(frame, 1)

    frame = cv2.resize(frame, (800, 550))

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(image)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        if len(results.multi_hand_landmarks) == 2:
            cv2.putText(
                frame, "Both Hands ", (250, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2
            )
        else:
            for i in results.multi_handedness:
                print(i)
                label = MessageToDict(i)["classification"][0]["label"]
                if label == "Left":
                    cv2.putText(
                        frame,
                        "Left Hand ",
                        (250, 50),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )
                else:
                    cv2.putText(
                        frame,
                        "Right Hand ",
                        (250, 50),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )

    cv2.imshow("Image", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()

    cv2.imshow("frame", frame)

    b = frame[:, :, 0]
    g = frame[:, :, 1]
    r = frame[:, :, 2]

    b_mean = np.mean(b)
    g_mean = np.mean(g)
    r_mean = np.mean(r)

    if b_mean > g_mean and b_mean > r_mean:
        print("Blue")
    elif g_mean > b_mean and g_mean > r_mean:
        print("Green")
    elif r_mean > b_mean and r_mean > g_mean:
        print("Red")
    else:
        print("Not Red")

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

import cv2


cap = cv2.VideoCapture(0)

harr_file = "haarcascade_frontalface_default.xml"
eye_file = "haarcascade_eye.xml"

face_cascade = cv2.CascadeClassifier(harr_file)
eye_cascade = cv2.CascadeClassifier(eye_file)

while True:
    _, frame = cap.read()

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.1,6)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_face = gray[y:y+h,x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_face,1.1,10)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(frame,(x+ex,y+ey),(x+ex+ew,y+ey+eh),(0,255,0),2)


    cv2.imshow("frame",frame)

    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()

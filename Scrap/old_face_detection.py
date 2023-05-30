# References used:
# https://towardsdatascience.com/face-detection-in-2-minutes-using-opencv-python-90f89d7c0f81

import cv2

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('../Extra_Files/haarcascade_frontalface_default.xml')

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Turn the frame into gray-scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) > 1:
        print("There are more than one person in frame. Please ask the other person to move away or move yourself to a location were no one can interrupt.")
        cv2.destroyWindow("face")

        # max_area, ind = 0, 0
        # for i, (x, y, w, h) in enumerate(faces):
        #     area = w * h
        #     if max_area < area:
        #         max_area = area
        #         ind = i
        # (x, y, w, h) = faces[ind]
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        # height, width, _ = frame.shape
        # cv2.imshow('face', frame[max(y - 10, 0):min(y + h + 10, height), max(x - 10, 0):min(x + w + 10, width)])

    elif len(faces) == 1:
        (x, y, w, h) = faces[0]
        print(x,y,w,h)
        height, width, _ = frame.shape
        cv2.imshow('face', frame[max(y-10, 0):min(y+h+10, height), max(x-10, 0):min(x+w+10, width)])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    else:
        print("No")

    cv2.imshow('img', frame)

    if cv2.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

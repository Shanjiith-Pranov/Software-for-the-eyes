#
import cv2

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('xml_files/haarcascade_frontalface_default.xml')
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
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Display the resulting frame
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) > 1:
        max_area, ind = 0, 0
        for i, (x, y, w, h) in enumerate(faces):
            area = w * h
            if max_area < area:
                max_area = area
                ind = i
        (x, y, w, h) = faces[ind]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        height, width, _ = frame.shape
        cv2.imshow('face', frame[max(y - 10, 0):min(y + h + 10, height), max(x - 10, 0):min(x + w + 10, width)])

    if len(faces) == 1:
        (x, y, w, h) = faces[0]
        print(x,y,w,h)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        height, width, _ = frame.shape
        cv2.imshow('face', frame[max(y-10, 0):min(y+h+10, height), max(x-10, 0):min(x+w+10, width)])
        print("Yes")
    else:
        print("No")
    # print(1, faces)
    # Draw rectangle around the faces
    # for (x, y, w, h) in faces:
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    #     cv2.imshow(f'({x}, {y}, {w}, {h})', frame[x:w, y:h])
    cv2.imshow('img', frame)

    if cv2.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cv2.waitKey()
cap.release()
cv2.destroyAllWindows()

#
# # Load the cascade
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# # Read the input image
# img = cv2.imread('test.jpg')
# # Convert into grayscale
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # Detect faces
# faces = face_cascade.detectMultiScale(gray, 1.1, 4)
# # Draw rectangle around the faces
# for (x, y, w, h) in faces:
#     cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
# # Display the output
# cv2.imshow('img', img)

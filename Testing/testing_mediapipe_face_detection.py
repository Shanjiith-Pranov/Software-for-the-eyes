import cv2
import mediapipe as mp
from math import ceil, trunc
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5) as face_detection:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    results = face_detection.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    height, width, _ = image.shape
    try:
      x = trunc(results.detections[0].location_data.relative_bounding_box.xmin * width)
      y = trunc(results.detections[0].location_data.relative_bounding_box.ymin * height)
      w = ceil(results.detections[0].location_data.relative_bounding_box.width * width)
      h = ceil(results.detections[0].location_data.relative_bounding_box.height * height)
      print(x, y, w, h)
      # cv2.imshow('Test', image[y - 30:y + h, x - 30:x + w + 30])
      # cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

      # Draw the face detection annotations on the image.
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      if results.detections:
        for detection in results.detections:
          mp_drawing.draw_detection(image, detection)
      # Flip the image horizontally for a selfie-view display.


    except:
      print("No face")

    cv2.imshow('MediaPipe Face Detection', cv2.flip(image, 1))
    if cv2.waitKey(1) == 'q':
      break

cap.release()
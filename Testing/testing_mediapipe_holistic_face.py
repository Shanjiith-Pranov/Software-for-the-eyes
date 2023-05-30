import mediapipe as mp
import cv2
from math import ceil, trunc, dist
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

mp_holistic.POSE_CONNECTIONS

mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)

cap = cv2.VideoCapture(0)

# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()

        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Make Detections
        results = holistic.process(image)
        # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks

        # Recolor image back to BGR for rendering
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks == None: continue

        height, width, _ = image.shape
        # for i in range(len(results.pose_landmarks.landmark)):
        #
        #     x = trunc(results.pose_landmarks.landmark[i].x * width)
        #     y = trunc(results.pose_landmarks.landmark[i].y * height)
        #     image = cv2.circle(image, (x, y), 1, (0, 255, 0), 10)
        #     print(f'{i}: {x}, {y}')
        #     cv2.imshow('image', image)
        #     cv2.waitKey()
        x2 = trunc(results.pose_landmarks.landmark[2].x * width)
        y2 = trunc(results.pose_landmarks.landmark[2].y * height)
        eye2 = (x2, y2)

        x5 = trunc(results.pose_landmarks.landmark[5].x * width)
        y5 = trunc(results.pose_landmarks.landmark[5].y * height)
        eye5 = (x5, y5)

        x7 = trunc(results.pose_landmarks.landmark[7].x * width)
        y7 = trunc(results.pose_landmarks.landmark[7].y * height)

        x8 = trunc(results.pose_landmarks.landmark[8].x * width)
        y8 = trunc(results.pose_landmarks.landmark[8].y * height)

        x9 = trunc(results.pose_landmarks.landmark[9].x * width)
        y9 = trunc(results.pose_landmarks.landmark[9].y * height)

        x10 = trunc(results.pose_landmarks.landmark[10].x * width)
        y10 = trunc(results.pose_landmarks.landmark[10].y * height)

        x11 = trunc(results.pose_landmarks.landmark[11].x * width)
        y11 = trunc(results.pose_landmarks.landmark[11].y * height)

        x12 = trunc(results.pose_landmarks.landmark[12].x * width)
        y12 = trunc(results.pose_landmarks.landmark[12].y * height)

        top_left = (x8, min(y8, y7))
        bottom_right = (x7, max(y9, y10))

        dist_x = ceil(max(dist(eye2, (x7, y7)), dist(eye5, (x8, y8))))

        top_left = (x8 - dist_x, max(y8, y7) - (max(y11, y12) - max(y9, y10)))
        bottom_right = (x7 + dist_x, (max(y9, y10) + (max(y11, y12))) // 2)
        try: cv2.imshow('face-crop', image[max(0,top_left[1]):min(height,bottom_right[1]),max(0,top_left[0]):min(width,bottom_right[0])])
        except: "Error displaying face"

        cv2.circle(image, (x2, y2), 1, (55, 150, 200), 10)  # Brown
        cv2.circle(image, (x5, y5), 1, (200, 150, 55), 10)  # Blue
        image = cv2.circle(image, (x7, y7), 1, (255, 255, 0), 10)  # Cyan
        image = cv2.circle(image, (x8, y8), 1, (0, 255, 255), 10)  # Yellow
        image = cv2.circle(image, (x9, y9), 1, (255, 0, 0), 10)  # Violet
        image = cv2.circle(image, (x10, y10), 1, (0, 0, 255), 10)  # Red
        image = cv2.circle(image, (x11, y11), 1, (75, 75, 100), 10)  # Brown
        image = cv2.circle(image, (x12, y12), 1, (100, 75, 75), 10)  # Grey
        cv2.rectangle(image, top_left, bottom_right, (50, 100, 255), 2)

        cv2.imshow('cam', image)
        if cv2.waitKey(1) == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
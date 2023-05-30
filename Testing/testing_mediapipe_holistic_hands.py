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

        height, width, _ = image.shape

        if results.pose_landmarks == None:
            continue
        elif results.left_hand_landmarks == None:
            continue
        elif results.right_hand_landmarks == None:
            continue
        x15 = trunc(results.pose_landmarks.landmark[15].x * width)
        y15 = trunc(results.pose_landmarks.landmark[15].y * height)

        x17 = trunc(results.pose_landmarks.landmark[17].x * width)
        y17 = trunc(results.pose_landmarks.landmark[17].y * height)

        xtop_left = ceil(min([i.x for i in results.left_hand_landmarks.landmark]) * width)
        ytop_left = ceil(min([i.y for i in results.left_hand_landmarks.landmark]) * height)
        padding_left = dist((x15, y15), (x17, y17))
        topleft_left = (ceil(xtop_left - padding_left), ceil(ytop_left - padding_left))

        xbottom_left = ceil(max([i.x for i in results.left_hand_landmarks.landmark]) * width)
        ybottom_left = ceil(max([i.y for i in results.left_hand_landmarks.landmark]) * height)

        bottomright_left = (ceil(xbottom_left + padding_left), ceil(ybottom_left + padding_left))

        x16 = trunc(results.pose_landmarks.landmark[16].x * width)
        y16 = trunc(results.pose_landmarks.landmark[16].y * height)

        x18 = trunc(results.pose_landmarks.landmark[18].x * width)
        y18 = trunc(results.pose_landmarks.landmark[18].y * height)

        xtop_right = ceil(min([i.x for i in results.right_hand_landmarks.landmark]) * width)
        ytop_right = ceil(min([i.y for i in results.right_hand_landmarks.landmark]) * height)
        padding_right = dist((x16, y16), (x18, y18))
        topleft_right = (ceil(xtop_right - padding_right), ceil(ytop_right - padding_right))

        xbottom_right = ceil(max([i.x for i in results.right_hand_landmarks.landmark]) * width)
        ybottom_right = ceil(max([i.y for i in results.right_hand_landmarks.landmark]) * height)

        bottomright_right = (ceil(xbottom_right + padding_right), ceil(ybottom_right + padding_right))

        try:
            cv2.imshow('left-hand-crop', image[max(0, topleft_left[1]):min(height, bottomright_left[1]),
                                         max(0, topleft_left[0]):min(width, bottomright_left[0])])
        except:
            "Error displaying left hand"
        try:
            cv2.imshow('right-hand-crop', image[max(0, topleft_right[1]):min(height, bottomright_right[1]),
                                          max(0, topleft_right[0]):min(width, bottomright_right[0])])
        except:
            "Error displaying right hand"

        image = cv2.circle(image, (x15, y15), 1, (255, 0, 0), 10)  # Violet
        image = cv2.circle(image, (x17, y17), 1, (75, 75, 100), 10)  # Brown
        image = cv2.circle(image, (xtop_left, ytop_left), 1, (55, 150, 200), 10)  # Orangeish-Brown
        image = cv2.circle(image, (xbottom_left, ybottom_left), 1, (200, 150, 55), 10)  # Blue
        cv2.rectangle(image, topleft_left, bottomright_left, (50, 100, 255), 2)
        image = cv2.circle(image, (x16, y16), 1, (0, 0, 255), 10)  # Red
        image = cv2.circle(image, (x18, y18), 1, (100, 75, 75), 10)  # Grey
        image = cv2.circle(image, (xtop_right, ytop_right), 1, (55, 150, 200), 10)  # Orangeish-Brown

        image = cv2.circle(image, (xbottom_right, ybottom_right), 1, (200, 150, 55), 10)  # Blue

        cv2.rectangle(image, topleft_right, bottomright_right, (50, 100, 255), 2)

        cv2.imshow('cam', image)
        if cv2.waitKey(1) == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
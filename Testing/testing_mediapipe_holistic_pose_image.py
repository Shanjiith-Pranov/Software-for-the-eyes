import mediapipe as mp
import cv2
from math import ceil, trunc, dist
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

mp_holistic.POSE_CONNECTIONS

mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)

image_path = '../Extra_Files/image_before_drawing_2.jpg'

# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    # Make Detections
    results = holistic.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 1. Draw face landmarks
    # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
    #                           mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
    #                           mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
    #                           )

    # 2. Right hand
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                              )

    # 3. Left Hand
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                              )

    # 4. Pose Detections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                              )

    # cv2.imwrite('image_after_drawing.jpeg',image)
    cv2.imshow('image', image)
    cv2.waitKey()
    cv2.destroyAllWindows()
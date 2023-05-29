import mediapipe as mp
import cv2
from math import ceil, trunc, dist
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

mp_holistic.POSE_CONNECTIONS

mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)

image_path = 'image_before_drawing_1.jpg'

# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    # Make Detections
    results = holistic.process(image)
    # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks

    # Recolor image back to BGR for rendering
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    print(len(results.pose_landmarks.landmark))
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
    image = cv2.circle(image, (x2, y2), 1, (55, 150, 200), 10) # Brown
    eye2 = (x2,y2)

    x5 = trunc(results.pose_landmarks.landmark[5].x * width)
    y5 = trunc(results.pose_landmarks.landmark[5].y * height)
    image = cv2.circle(image, (x5, y5), 1, (200, 150, 55), 10) # Blue
    eye5 = (x5,y5)

    x7 = trunc(results.pose_landmarks.landmark[7].x * width)
    y7 = trunc(results.pose_landmarks.landmark[7].y * height)
    image = cv2.circle(image, (x7, y7), 1, (255, 255, 0), 10) # Cyan

    x8 = trunc(results.pose_landmarks.landmark[8].x * width)
    y8 = trunc(results.pose_landmarks.landmark[8].y * height)
    image = cv2.circle(image, (x8, y8), 1, (0, 255, 255), 10) # Yellow

    x9 = trunc(results.pose_landmarks.landmark[9].x * width)
    y9 = trunc(results.pose_landmarks.landmark[9].y * height)
    image = cv2.circle(image, (x9, y9), 1, (255, 0, 0), 10) # Violet

    x10 = trunc(results.pose_landmarks.landmark[10].x * width)
    y10 = trunc(results.pose_landmarks.landmark[10].y * height)
    image = cv2.circle(image, (x10, y10), 1, (0, 0, 255), 10) # Red

    top_left = (x8, min(y8,y7))
    bottom_right = (x7, max(y9, y10))

    dist_y = ceil(dist(top_left,bottom_right)/3*2)
    dist_x = ceil(max(dist(eye2,(x7,y7)),dist(eye5,(x8,y8))))

    top_left = (x8-dist_x, min(y8, y7) - dist_y)
    bottom_right = (x7+dist_x, max(y9, y10))

    cv2.rectangle(image, top_left, bottom_right, (50, 100, 255), 2)

    cv2.imshow('image', image)
    cv2.waitKey()
    image = cv2.imread(image_path)

    x15 = trunc(results.pose_landmarks.landmark[15].x * width)
    y15 = trunc(results.pose_landmarks.landmark[15].y * height)
    image = cv2.circle(image, (x15, y15), 1, (55, 150, 200), 10)  # Brown
    wrist15 = (x15,y15)

    x16 = trunc(results.pose_landmarks.landmark[16].x * width)
    y16 = trunc(results.pose_landmarks.landmark[16].y * height)
    image = cv2.circle(image, (x16, y16), 1, (200, 150, 55), 10)  # Blue
    wrist16 = (x16,y16)

    x17 = trunc(results.pose_landmarks.landmark[17].x * width)
    y17 = trunc(results.pose_landmarks.landmark[17].y * height)
    image = cv2.circle(image, (x17, y17), 1, (255, 255, 0), 10)  # Cyan
    hand17 = (x17,y17)

    x18 = trunc(results.pose_landmarks.landmark[18].x * width)
    y18 = trunc(results.pose_landmarks.landmark[18].y * height)
    image = cv2.circle(image, (x18, y18), 1, (0, 255, 255), 10)  # Yellow
    hand18 = (x18,y18)

    dist_left = ceil(dist(wrist15,hand17)*2)
    dist_right = ceil(dist(wrist16,hand18)*2)


    cv2.rectangle(image, (hand17[0]-dist_left,hand17[1]-dist_left), (hand17[0]+dist_left,hand17[1]+dist_left), (255, 100, 50), 2)
    cv2.rectangle(image, (hand18[0] - dist_left, hand18[1] - dist_left), (hand18[0] + dist_left, hand18[1] + dist_left), (50, 100, 255), 2)


    cv2.imshow('image', image)
    cv2.waitKey()
    image = cv2.imread(image_path)

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
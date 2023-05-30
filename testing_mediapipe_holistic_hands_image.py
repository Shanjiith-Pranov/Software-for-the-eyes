import mediapipe as mp
import cv2
from math import ceil, trunc, dist
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

mp_holistic.POSE_CONNECTIONS

mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)

image_path = 'image_before_drawing_2.jpg'

# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    # Make Detections
    results = holistic.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape

    # print(len(results.right_hand_landmarks.landmark))
    # for i in range(len(results.right_hand_landmarks.landmark)):
    #
    #     x = trunc(results.right_hand_landmarks.landmark[i].x * width)
    #     y = trunc(results.right_hand_landmarks.landmark[i].y * height)
    #     image = cv2.circle(image, (x, y), 1, (0, 255, 0), 10)
    #     print(f'{i}: {x}, {y}')
    #     cv2.imshow('image', image)
    #     cv2.waitKey()

    x15 = trunc(results.pose_landmarks.landmark[15].x * width)
    y15 = trunc(results.pose_landmarks.landmark[15].y * height)
    image = cv2.circle(image, (x15, y15), 1, (255, 0, 0), 10)  # Violet

    x17 = trunc(results.pose_landmarks.landmark[17].x * width)
    y17 = trunc(results.pose_landmarks.landmark[17].y * height)
    image = cv2.circle(image, (x17, y17), 1, (75, 75, 100), 10)  # Brown

    xtop_left = ceil(min([i.x for i in results.left_hand_landmarks.landmark]) * width)
    ytop_left = ceil(min([i.y for i in results.left_hand_landmarks.landmark]) * height)
    padding_left = dist((x15, y15), (x17, y17))
    topleft_left = (ceil(xtop_left - padding_left), ceil(ytop_left - padding_left))
    image = cv2.circle(image, (xtop_left, ytop_left), 1, (55, 150, 200), 10)  # Orangeish-Brown

    xbottom_left = ceil(max([i.x for i in results.left_hand_landmarks.landmark]) * width)
    ybottom_left = ceil(max([i.y for i in results.left_hand_landmarks.landmark]) * height)

    bottomright_left = (ceil(xbottom_left + padding_left), ceil(ybottom_left + padding_left))

    image = cv2.circle(image, (xbottom_left, ybottom_left), 1, (200, 150, 55), 10)  # Blue

    cv2.rectangle(image, topleft_left, bottomright_left, (50, 100, 255), 2)

    x16 = trunc(results.pose_landmarks.landmark[16].x * width)
    y16 = trunc(results.pose_landmarks.landmark[16].y * height)
    image = cv2.circle(image, (x16, y16), 1, (0, 0, 255), 10)  # Red

    x18 = trunc(results.pose_landmarks.landmark[18].x * width)
    y18 = trunc(results.pose_landmarks.landmark[18].y * height)
    image = cv2.circle(image, (x18, y18), 1, (100, 75, 75), 10)  # Grey

    xtop_right = ceil(min([i.x for i in results.right_hand_landmarks.landmark]) * width)
    ytop_right = ceil(min([i.y for i in results.right_hand_landmarks.landmark]) * height)
    padding_right = dist((x16, y16), (x18, y18))
    topleft_right = (ceil(xtop_right - padding_right), ceil(ytop_right - padding_right))
    image = cv2.circle(image, (xtop_right, ytop_right), 1, (55, 150, 200), 10)  # Orangeish-Brown

    xbottom_right = ceil(max([i.x for i in results.right_hand_landmarks.landmark]) * width)
    ybottom_right = ceil(max([i.y for i in results.right_hand_landmarks.landmark]) * height)

    bottomright_right = (ceil(xbottom_right + padding_right), ceil(ybottom_right + padding_right))

    image = cv2.circle(image, (xbottom_right, ybottom_right), 1, (200, 150, 55), 10)  # Blue

    cv2.rectangle(image, topleft_right, bottomright_right, (50, 100, 255), 2)

    # cv2.rectangle(image, (hand17[0]-dist_left,hand17[1]-dist_left), (hand17[0]+dist_left,hand17[1]+dist_left), (255, 100, 50), 2)
    # cv2.rectangle(image, (hand18[0] - dist_left, hand18[1] - dist_left), (hand18[0] + dist_left, hand18[1] + dist_left), (50, 100, 255), 2)

    cv2.imshow('image', image)
    cv2.waitKey()
    cv2.destroyAllWindows()
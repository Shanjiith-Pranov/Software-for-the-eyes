import mediapipe as mp
import cv2
from math import ceil, trunc, dist
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

mp_holistic.POSE_CONNECTIONS

mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)

toggle = 0

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

        if toggle == 0:
            if results.pose_landmarks == None: continue
            else: print("Face not detected")

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
            cv2.circle(image, (x2, y2), 1, (55, 150, 200), 10)  # Brown
            eye2 = (x2, y2)

            x5 = trunc(results.pose_landmarks.landmark[5].x * width)
            y5 = trunc(results.pose_landmarks.landmark[5].y * height)
            cv2.circle(image, (x5, y5), 1, (200, 150, 55), 10)  # Blue
            eye5 = (x5, y5)

            x7 = trunc(results.pose_landmarks.landmark[7].x * width)
            y7 = trunc(results.pose_landmarks.landmark[7].y * height)
            image = cv2.circle(image, (x7, y7), 1, (255, 255, 0), 10)  # Cyan

            x8 = trunc(results.pose_landmarks.landmark[8].x * width)
            y8 = trunc(results.pose_landmarks.landmark[8].y * height)
            image = cv2.circle(image, (x8, y8), 1, (0, 255, 255), 10)  # Yellow

            x9 = trunc(results.pose_landmarks.landmark[9].x * width)
            y9 = trunc(results.pose_landmarks.landmark[9].y * height)
            image = cv2.circle(image, (x9, y9), 1, (255, 0, 0), 10)  # Violet

            x10 = trunc(results.pose_landmarks.landmark[10].x * width)
            y10 = trunc(results.pose_landmarks.landmark[10].y * height)
            image = cv2.circle(image, (x10, y10), 1, (0, 0, 255), 10)  # Red

            x11 = trunc(results.pose_landmarks.landmark[11].x * width)
            y11 = trunc(results.pose_landmarks.landmark[11].y * height)
            image = cv2.circle(image, (x11, y11), 1, (75, 75, 100), 10)  # Brown

            x12 = trunc(results.pose_landmarks.landmark[12].x * width)
            y12 = trunc(results.pose_landmarks.landmark[12].y * height)
            image = cv2.circle(image, (x12, y12), 1, (100, 75, 75), 10)  # Grey

            top_left = (x8, min(y8, y7))
            bottom_right = (x7, max(y9, y10))

            dist_x = ceil(max(dist(eye2, (x7, y7)), dist(eye5, (x8, y8))))

            top_left = (x8 - dist_x, max(y8, y7) - (max(y11, y12) - max(y9, y10)))
            bottom_right = (x7 + dist_x, (max(y9, y10) + (max(y11, y12))) // 2)

            cv2.rectangle(image, top_left, bottom_right, (50, 100, 255), 2)
        elif toggle == 1:
            if results.pose_landmarks == None: continue
            elif results.left_hand_landmarks == None: continue
            elif results.right_hand_landmarks == None: continue
            else: print("Please put both hands in frame")

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
        else:
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

        cv2.imshow('image', image)

        if cv2.waitKey(1) == ord('a'):
            toggle += 1
            if toggle == 3:
                toggle = 0
            cv2.destroyAllWindows()

        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
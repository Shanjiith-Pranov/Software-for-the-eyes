import numpy as np
import dlib
import cv2
from PIL import Image

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')

# Path of image
path = "../Extra_Files/face1.jpg"

img = dlib.load_rgb_image(path)

if len(detector(img)) == 0:
    print('No face detected')
rect = detector(img)[0]
sp = predictor(img, rect)
landmarks = np.array([[p.x, p.y] for p in sp.parts()])

nose_bridge_x = []
nose_bridge_y = []

for i in [28, 29, 30, 31, 33, 34, 35]:
    nose_bridge_x.append(landmarks[i][0])
    nose_bridge_y.append(landmarks[i][1])

### x_min and x_max
x_min = min(nose_bridge_x)
x_max = max(nose_bridge_x)

### ymin (from top eyebrow coordinate),  ymax
y_min = landmarks[20][1]
y_max = landmarks[30][1]

img2 = Image.open(path)
img2 = img2.crop((x_min, y_min, x_max, y_max))

img_blur = cv2.GaussianBlur(np.array(img2), (3, 3), sigmaX=0, sigmaY=0)

edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)

edges_center = edges.T[(int(len(edges.T) / 2))]

# Displaying images
img3 = np.array(img2)

cv2.imshow("Original image", img)
cv2.imshow("Cropped image", img3)
cv2.imshow("Edges", edges)

# waits for user to press any key
# (this is necessary to avoid Python kernel form crashing)
cv2.waitKey(0)

# closing all open windows
cv2.destroyAllWindows()


if 255 in edges_center:
    print("Glasses are present")
else:
    print("Glasses are not present")
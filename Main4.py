import cv2
import numpy as np

img = cv2.imread('IMG/face5.jpg')
height, weight = img.shape[:2]

nose_cascade = cv2.CascadeClassifier('Noses.xml')
eye_cascade = cv2.CascadeClassifier('Eyes.xml')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
eyesRes = eye_cascade.detectMultiScale(gray, 1.3, 4)
noseRes = nose_cascade.detectMultiScale(gray, 1.3, 5)

index = 0
for (ex, ey,  ew,  eh) in eyesRes:
    if index == 0:
        eye_1 = (ex, ey, ew, eh)
    elif index == 1:
        eye_2 = (ex, ey, ew, eh)
    index = index + 1

if eye_1[0] < eye_2[0]:
   left_eye = eye_1
   right_eye = eye_2
else:
   left_eye = eye_2
   right_eye = eye_1

left_eye_center = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
right_eye_center = (int(right_eye[0] + (right_eye[2] / 2)), int(right_eye[1] + (right_eye[3] / 2)))

delta_x = right_eye_center[0] - left_eye_center[0]
delta_y = right_eye_center[1] - left_eye_center[1]
angle = ((np.arctan(delta_y/delta_x)) * 180) / np.pi

center = (weight / 2, height / 2)
M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated = cv2.warpAffine(img, M, (weight, height))

height = int(height / 5)
for (x, y, w, h) in noseRes:
    x = x + int(w/2)
    print('Координата носа', x)
    cv2.rectangle(rotated, (x, y-h), (x, y+height), (0, 255, 0), 2)

index = 0
for (ex, ey,  ew,  eh) in eyesRes:
    if index == 2:
        break
    if index == 0:
        eye_1 = (ex, ey, ew, eh)
    elif index == 1:
        eye_2 = (ex, ey, ew, eh)

    ex = ex + int(ew/2)
    print('Координата глаза', ex)
    cv2.rectangle(rotated, (ex, ey-eh), (ex, ey+height), (0, 0, 255), 2)
    index = index + 1

cv2.imshow('Rotated', rotated)
# cv2.resizeWindow('Rotated', 500, 500)
cv2.waitKey(0)
cv2.destroyAllWindows()
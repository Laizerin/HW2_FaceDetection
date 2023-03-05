import numpy as np
import cv2

img = cv2.imread('IMG/ORL/TemplateLackNose.jpg')
template = cv2.imread('IMG/ORL/TemplateBright.jpg')

h, w = template.shape[:-1]

# methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]
methods = [cv2.TM_CCOEFF]

for method in methods:
    img2 = img.copy()

    result = cv2.matchTemplate(img2, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    # if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
    #     location = min_loc
    # else:
    location = max_loc

    bottom_right = (location[0] + w, location[1] + h)
    cv2.rectangle(img2, location, bottom_right, 255, 2)
    cv2.imshow('Result', img2)
    # cv2.resizeWindow('Result', 200, 100)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

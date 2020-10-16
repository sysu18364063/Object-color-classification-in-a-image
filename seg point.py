"""
Filename: init.py
Usage: This script will measure different objects in the frame using a reference object of known dimension.
The object with known dimension must be the leftmost object.
Author: Shashank Sharma
"""
from scipy.spatial.distance import euclidean
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2

# Function to show array of images (intermediate results)
def show_images(images):
	for i, img in enumerate(images):
		cv2.imshow("image_" + str(i), img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()



img_path = "1250_1.jpg"

# Read image and preprocess
image = cv2.imread(img_path)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (9, 9), 0)

# show_images([gray, blur])

# edged = cv2.Canny(blur, 50, 100)
edged = cv2.Canny(blur, 30, 90)


edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

# show_images([image, blur, edged])

# Find contours
# cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = imutils.grab_contours(cnts)

contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours([contours, hierarchy])

# Sort contours from left to right as leftmost contour is reference object


# Remove contours which are not large enough

#cv2.drawContours(image, cnts, -1, (0,255,0), 3)

#show_images([image, edged])
#print(len(cnts))
h, w, ch = image.shape
result = np.zeros((h, w, ch), dtype=np.uint8)

for cnt in range(len(contours)):
    if cv2.contourArea(cnts[cnt]) < 10:
        continue
    # cnts = [x for x in cnts if cv2.contourArea(x) > 100]
    # 提取与绘制轮廓
    cv2.drawContours(result, contours, cnt, (0, 255, 0), 2)

    # 求解中心位置
    mm = cv2.moments(contours[cnt])
    temp_sum = 0
    b = 0
    g = 0
    r = 0
    for i in [-1,0,1]:
        cx = int(mm['m10']/mm['m00'])
        cy = int(mm['m01']/mm['m00'])
        cx = cx + i
        cy = cy + i

        color = image[cy][cx]

        b += color[0]
        g += color[1]
        r += color[2]
    temp_sum = b + g + r
    b_norm = (b/temp_sum)
    g_norm = (g/temp_sum)
    r_norm = (r/temp_sum)
    if r_norm < 0.5: # 归一化后如果红色的占比就认为是黑色
        print("黑点")
        cv2.circle(result, (cx, cy), 3, (255, 255, 255), -1)
    else:
        print("红点")
        cv2.circle(result, (cx, cy), 3, (0, 0, 255), -1)

    # color_str = "(" + str(color[0]) + ", " + str(color[1]) + ", " + str(color[2]) + ")"
    # print("颜色: %s  " % (color_str))
    # show_images([image, result])


show_images([image, result])
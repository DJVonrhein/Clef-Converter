import cv2 as cv
import numpy as np
from math import ceil, floor

input_file_name = 'img/takefive.png'
# input_file_name = 'img/reminiscences.png'

img_color =  cv.imread(input_file_name) 
img = cv.imread(input_file_name, cv.IMREAD_GRAYSCALE)    # input is converted from color to grayscale
assert img is not None, "file could not be read, check with os.path.exists()"

ret2,otsu = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU) # otsu is generally used as the main binary form of image
# cv.imshow('grayscale', img)
cv.imshow('otsu', otsu)

def is_contour_bad(c):
    return not cv.isContourConvex(c)

contour_img = np.ones(img.shape[:2], dtype="uint8") * 255
contours, hierarchy = cv.findContours(otsu, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

cv.drawContours(contour_img, contours, -1,  255, 1 )

mask = np.ones(img.shape[:2], dtype="uint8") * 255
# loop over the contours
for c in contours:
	# if the contour is bad, draw it on the mask
	if is_contour_bad(c):
		cv.drawContours(mask, [c], -1, 0, -1)

cv.imshow('mask', mask)
cv.waitKey(0)
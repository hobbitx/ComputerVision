import cv2 as cv
import sys
img = cv.imread("image.png")
if img is None:
    sys.exit("Could not read the image.")
cv.imshow("Display window", img)

cv.waitKey(0)
cv.destroyAllWindows()
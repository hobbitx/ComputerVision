import cv2 as cv
import numpy as np
import math
def calcule_mean(img):
    x,y,n = img.shape
    total = 0
    for a in range(0,y-1):
        for b in range(0,x-1):
            for k in range(0,n-1):
                total = total + img.item(b,a,k)
    calculate_mean = round(total/img.size,4)
    return calculate_mean

def calcule_variance(img,mean = -1):
    if mean == -1:
        mean = calcule_mean(img)
    x,y,n = img.shape
    total = 0
    for a in range(0,y-1):
        for b in range(0,x-1):
            for k in range(0,n-1):
                total = total + pow((img.item(b,a,k)-mean),2)
    variance = round(total/img.size,4)
    return variance

def calcule_deviation(img,variance = -1):
    if variance == -1:
        variance = calcule_variance(img)
    return round(math.sqrt(variance),4)

img = cv.imread("./imgs/image.png")
overlay = img.copy()
output = img.copy()
size = 0.6

val = np.reshape(img[:,:,0], -1)
mean = calcule_mean(img)
img_mean = "Mean: " + str(mean)
img_std = "Deviation: " + str(calcule_deviation(img))
strXY = "Position: " 
strBGR = "RGB: "
strI = "Intesify: "
#click event function
def click_event(event, x, y, flags, param):
    info = np.full((200, 200, 3), fill_val)
    overlay = img.copy()
    blue = img[y, x, 0]
    green = img[y, x, 1]
    red = img[y, x, 2]
    cv.circle(overlay,(x,y), 13, (255,255,255),2)
    font = cv.FONT_HERSHEY_SIMPLEX
    strXY = "Position: " + str(x)+", "+str(y)
    strBGR = "RGB: "+str(blue)+","+str(green)+","+str(red)
    intensify = round(((red + green + blue)/3),2)
    strI = "Intesify: " + str(intensify)
    cv.putText(info, strXY, (5,15), font, size, (255,255,255), 2)
    cv.putText(info, strBGR, (5,35), font, size, (255,255,255), 2)
    cv.putText(info, strI, (5,55), font, size, (255,255,255), 2)
    cv.putText(info, img_mean, (5,75), font, size, (255,255,255), 2)
    cv.putText(info, img_std, (5,95), font, size, (255,255,255), 2)
    cv.addWeighted(overlay, 1.0, output, 0,0,output)
    cv.imshow("image", output)
  
    cv.imshow('window',info)

cv.imshow("image", img)
fill_val = np.array([0, 0, 0], np.uint8)
info = np.full((200, 200, 3), fill_val)
cv.imshow('window', info)
cv.setMouseCallback("image", click_event)
cv.waitKey(0)
cv.destroyAllWindows()
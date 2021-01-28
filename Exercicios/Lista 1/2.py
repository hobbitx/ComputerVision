import os
import cv2 as cv
import numpy as np
import math  
from matplotlib import pyplot as plt

max_size = 50

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

calcs = list()
arr = os.listdir("./imgs")
count = 0
deviations = []
means = []
times = []
for file in arr:
    if count == max_size:
        break
    img = cv.imread(f"imgs/{str(file)}")
    if img is not None:
        val = np.reshape(img[:,:,1], -1)
        mean = calcule_mean(img)
        variance = calcule_variance(img,mean)
        deviation = calcule_deviation(img,variance)
        dic = {
            "name":file,
            "mean": mean,
            "variance": variance,
            "deviation": deviation
        }
        calcs.append(dic)
        deviations.append(deviation)
        means.append(mean)
        times.append(count)
        count = count + 1

plt.plot(times,deviations,label = "deviations") 
plt.xlabel('frames')
plt.plot(times,means,label = "mean")
plt.legend() 
plt.show()

"""
for i in range(0,len(calcs)-1):
    g = calcs[i]
    f = calcs[i+1]
    print(g,f)
    alpha = (g["deviation"]/f["deviation"])*f["mean"]-g["mean"]
    beta = f["deviation"]/g["deviation"]
    print(alpha,beta)
"""

import os
import cv2 as cv
import numpy as np
import math  
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
import statistics

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

def calculate_contrast(img):
    x,y,n = img.shape
    total = 0
    for a in range(0,y-1):
        for b in range(0,x-1):
            for k in range(0,n-1):
                adjacent = [
                    img.item(b-1,a,k),
                    img.item(b+1,a,k),
                    img.item(b,a-1,k),
                    img.item(b,a+1,k)
                ]
                total = total + abs( img.item(b,a,k) - statistics.mean(adjacent) )
    contrast = round(total/img.size,4)
    return contrast

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
contrasts = []
means = []
times = []

for file in arr:
    if count == max_size:
        break
    img = cv.imread(f"imgs/{str(file)}")
    if img is not None:
        print("Img",str(count))
        val = np.reshape(img[:,:,1], -1)
        mean = calcule_mean(img)
        variance = calcule_variance(img,mean)
        deviation = calcule_deviation(img,variance)
        contrast = calculate_contrast(img)
        dic = {
            "name":file,
            "mean": mean,
            "variance": variance,
            "deviation": deviation,
            "contrast": contrast
        }
        calcs.append(dic)
        deviations.append(deviation)
        contrasts.append(contrast)
        means.append(mean)
        times.append(count)
        count = count + 1



f2 = interp1d(times,deviations, kind='cubic')
g2 = interp1d(times,means, kind='cubic')
h2 = interp1d(times,contrasts, kind='cubic')

g_mean = statistics.mean(g2(times))
f_mean = statistics.mean(f2(times))
h_mean = statistics.mean(h2(times))
g_dev = statistics.stdev(g2(times))
f_dev = statistics.stdev(f2(times))
h_dev = statistics.stdev(h2(times))


alpha = (g_dev/f_dev)*f_mean-g_mean
beta = f_dev/g_dev
alpha2 = (h_dev/f_dev)*f_mean-h_mean
beta2 = f_dev/h_dev


f = f2(times)
g = g2(times)

g_new = []
h_new = []
y_new = g2(times)
a_new = h2(times)
for i in range(0,len(calcs)):
    g_new.append(beta*(y_new[i]+alpha))
    h_new.append(beta2*(a_new[i]+alpha2))


plt.figure()

plt.subplot(211)
plt.plot(times,deviations,label = "deviations") 
plt.xlabel('frames')
plt.plot(times,means,label = "mean")
plt.plot(times,contrasts,label="contrast")
plt.legend() 

plt.subplot(212)
plt.ylim(-1, max(max(deviations),max(g_new)))
plt.plot(times,deviations,label = "deviations") 
plt.xlabel('frames')
plt.plot(times,g_new,label = "mean")
plt.plot(times,h_new,label="contrast")
plt.legend()

d = 1/max(times) 
sum_d2 = 0
sum_new_d2= 0
sum_new_h2= 0
for i in times:
    sum_d2 = pow((f[i]-g[i]),2)
    sum_new_d2 = pow((f[i]-g_new[i]),2)
    sum_new_h2 = pow((f[i]-h_new[i]),2)
    
d1_g=d*math.sqrt(sum_d2)
d1_g_new = d*math.sqrt(sum_new_d2)
d1_h_new = d*math.sqrt(sum_new_h2)

print("d_2(f,g)=",d1_g)
print("d_2(f,g_new)=",d1_g_new)
print("d_2(f,h_new)=",d1_g_new)

plt.show()
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import math
from matplotlib import cm
import tkinter as tk
from tkinter import simpledialog    

N = 40
G_MAX = 255

blank_image = np.zeros((N,N,3), np.uint8)
blank_image[:,0:N//2] = (255,0,0)      # (B, G, R)
blank_image[:,N//2:N] = (0,255,0)
img = np.zeros((G_MAX+N,G_MAX+N,3), np.uint8)


def get_H(r,g,b):
    eps = get_eps(r,g,b)
    if b <= g:
        return eps
    else:
        return 2*math.pi-eps

def get_eps(r,g,b):
    if r == 0 and b == 0 and g==0:
        return 0
    if r == b == g:
        return 0
    up = ((r-g)+(r-b))
    dow = 2*math.sqrt(pow(r-g,2)+(r-b)*(g-b))

    return math.acos(up/dow)
def get_S(r,g,b):
    if r == 0 and b == 0 and g==0:
        return 0
    minimun = min([r,g,b])
    return 1 - ((3*minimun)/(r+b+g))

def get_values(r,g,b):
    eps = get_eps(r,g,b)
    h = get_H(eps,r,g,b)
    s = get_S(r,g,b)
    return h,eps,s

def distance(r,theta,r2,theta2):
    x1 = r * math.cos(theta)
    y1 = r * math.sin(theta)  
    x2 = r2 * math.cos(theta2)
    y2 = r2 * math.sin(theta2)
    d = math.sqrt(pow(x1-x2,2)+pow(y1-y2,2))
    return d

def generate_img(u,img):
    alpha = 54.74
    beta = 7.95
    d= 3*(u)
    s= get_S(u,u,u)
    h= get_H(u,u,u)
    print(d)
    for x in range(0,G_MAX):
        for y in range(0,G_MAX):
            for z in range(0,G_MAX):
                new_1 = [x* math.cos(alpha)-x*math.sin(alpha),y*math.sin(alpha)+y*math.cos(alpha),z]
                new_2 = [new_1[0],new_1[1]* math.cos(alpha)-new_1[1]*math.sin(alpha),new_1[2]*math.sin(alpha)+new_1[2]* math.cos(alpha)]
                new_3 = [new_2[0]* math.cos(beta)+new_2[0]*math.sin(beta),new_2[1],new_2[2]*math.cos(beta)-new_2[2]* math.sin(beta)]
                if -x-y-z == -d:
                    img[int(new_3[0]),int(new_3[1])] = [x,y,z]
                else:
                    img[int(new_3[0]),int(new_3[1])] = [0,0,0]        
   
    print(img.shape[:2])
    return img

ROOT = tk.Tk()

ROOT.withdraw()
# the input dialog
U = int(simpledialog.askstring(title="Test",
                                  prompt="Digite o valor de U:"))

img_2 = generate_img(U,img)
cv.imshow('image',img_2)
cv.waitKey()
cv.destroyAllWindows()
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2


rgbImg = mpimg.imread('rgb.jpg')
#if r first image (0,0)should have a value of approx 255, 0, 0
rgbImg[0,0,:]

cap = cv2.VideoCapture('rgb.mp4')
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

ret, frame = cap.read()
frame[0,0,:] #according to the internet is bgr so should get 0,0,255
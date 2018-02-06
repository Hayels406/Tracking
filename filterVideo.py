import numpy as np
import cv2
import os
from skimage import measure
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

if os.getcwd().rfind('b1033128') > 0:
    videoLocation = '/home/b1033128/Documents/throughFenceRL.mp4'
    save = '/home/b1033128/Documents/throughFenceRL/'
elif os.getcwd().rfind('hayley') > 0:
    videoLocation = '/users/hayleymoore/Documents/PhD/Tracking/throughFenceRL.mp4'
    save = '/users/hayleymoore/Documents/PhD/Tracking/throughFenceRL/'

plot = 's'
darkTolerance = 173.5
sizeOfObject = 60
restart = 6

maxfilter = []
frameID = 0


cap = cv2.VideoCapture(videoLocation)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print 'You have', length, 'frames', videoLocation

while(frameID <= length):
    plt.close('all')
    ret, frame = cap.read()
    if ret == True:
        print frameID
        grey = frame[:,:,0] #extract blue channel

        grey[grey < darkTolerance] = 0.0
        grey[grey > darkTolerance+10.] = 255.

        img = cv2.GaussianBlur(grey,(5,5),2)

        maxfilter += [ndimage.maximum_filter(img, size=2)]
    else:
        break
    if np.mod(frameID, 1000) == 999:
        np.save('frames'+str(frameID).zfill(4), maxfilter)
        maxfilter = []
    frameID += 1

np.save('frames'+str(frameID), maxfilter)

cap.release()

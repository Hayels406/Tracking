import numpy as np
import cv2
import os
from skimage import measure
import scipy.ndimage as ndimage
from imutils import contours
import imutils as im
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from collections import Counter
import copy

from trackingFunctions import movingCrop

if os.getcwd().rfind('Uni') > 0:
    videoLocation = '/home/b1033128/Documents/throughFenceRL.mp4'
    save = '/home/b1033128/Documents/throughFenceRL/'
    dell = True
    brk = False
elif os.getcwd().rfind('hayley') > 0:
    videoLocation = '/users/hayleymoore/Documents/PhD/Tracking/throughFenceRL.mp4'
    save = '/users/hayleymoore/Documents/PhD/Tracking/throughFenceRL/'
else:
    videoLocation = '/data/b1033128/Videos/throughFenceRL.mp4'
    save = '/data/b1033128/Tracking/throughFenceRL/'
    dell = False

plot = 's'
darkTolerance = 173.5
sizeOfObject = 60
restart = 50

sheepLocations = []
sheepVelocity = []
frameID = 0
cropVector = [1000,1000,2000,2028]




cap = cv2.VideoCapture(videoLocation)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print 'You have', length, 'frames', videoLocation

sheep = np.load('locfull.npy')
sheep =  map(np.array,  sheep)
print 'You have analysed', len(sheep), 'frames'

S = 0
F = len(sheep)

lenTraj = 10
movingAverage = 6

N = len(sheep[0])

while(frameID <= len(sheep)-6):
    ret, frame = cap.read()
    print frameID
    full = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    fullCropped, cropVector = movingCrop(frameID, full, sheep, cropVector)
    cropX, cropY, _,  _ = cropVector
    plt.clf()
    plt.imshow(fullCropped)
    for j in range(N):
    	smoothX = np.convolve(np.array(sheep)[S:F,j,0], np.ones((movingAverage,))/movingAverage, mode='valid')
    	smoothY = np.convolve(np.array(sheep)[S:F,j,1], np.ones((movingAverage,))/movingAverage, mode='valid')
    	for i in range(3):
    		smoothX =  np.append([smoothX[0]],  smoothX)
    		smoothY =  np.append([smoothY[0]],  smoothY)
    		smoothX =  np.append(smoothX, [smoothX[-1]])
    		smoothY =  np.append(smoothY, [smoothY[-1]])
    	plt.plot(smoothX[max(frameID-lenTraj,  0):frameID]-cropX, smoothY[max(frameID-lenTraj,  0):frameID]-cropY, lw = 1, color='blue')
    	plt.gca().set_aspect('equal')
    plt.gca().set_axis_off()
    plt.savefig(save+'traj'+str(frameID).zfill(4), bbox_inches='tight',  dpi = 300)




    frameID +=1
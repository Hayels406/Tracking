import numpy as np
import cv2
import os
import sys
from skimage import measure
import scipy.ndimage as ndimage
from imutils import contours
import imutils as im
from glob import glob
from matplotlib import cm
import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from collections import Counter
import copy

from trackingFunctions import movingCrop

video = sys.argv[1]
videoLocation = '/data/b1033128/Videos/'+video
save = '/data/b1033128/Tracking/'+video[:-4] +'/'
print save

plot = 's'
darkTolerance = 173.5
sizeOfObject = 60
restart = 50

sheepLocations = []
sheepVelocity = []
frameID = 0
cropInit = [1000,1000,2000,2028]




cap = cv2.VideoCapture(videoLocation)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print 'You have', length, 'frames', videoLocation

if (videoLocation.rfind('data') > 0) and (videoLocation.rfind('throughFenceRL') > 0):
    print 'Skipping first 15 frames'
    while(frameID <= 15):
        ret, frame = cap.read()
        frameID += 1

    if frameID > 0:
        frameID = 0

if (videoLocation.rfind('data') > 0) and (videoLocation.rfind('CaseJ2') > 0):
    print 'Skipping first 10 frames'
    while(frameID <= 10):
        ret, frame = cap.read()
        frameID += 1
if (videoLocation.rfind('data') > 0) and (videoLocation.rfind('CaseH2') > 0):
    print 'Skipping first 19 frames'
    while(frameID <= 19):
        ret, frame = cap.read()
        frameID += 1

if frameID > 0:
    frameID = 0
data = glob(save+'Final-loc*')[-1]
sheep = np.load(data)
quad = np.load(glob(save+'Final-quad*')[-1])
sheep =  map(np.array,  sheep)
print 'You have analysed', len(sheep), 'frames'

S = 0
F = len(sheep)

lenTraj = 75
movingAverage = 6

N = len(sheep[0])
alph = np.linspace(0,1,F)
colors = [ cm.prism(x) for x in np.linspace(0., 1., N)]


while(frameID <= len(sheep)-movingAverage):
    ret, frame = cap.read()
    print frameID
    full = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    if frameID == 0:
        cropVector = cropInit
    _, cropVector = movingCrop(frameID, full, sheep, cropVector)
    fullCropped = full[min(cropInit[1], cropVector[1]):cropInit[3], min(cropInit[0],cropVector[0]):cropInit[2],:]
    cropX, cropY, _,  _ = cropVector
    plt.clf()
    plt.imshow(full)
    for j in range(N):

    	smoothX = np.convolve(np.array(sheep)[S:F,j,0], np.ones((movingAverage,))/movingAverage, mode='valid')
    	smoothY = np.convolve(np.array(sheep)[S:F,j,1], np.ones((movingAverage,))/movingAverage, mode='valid')
    	for i in range(3):
    		smoothX =  np.append([smoothX[0]],  smoothX)
    		smoothY =  np.append([smoothY[0]],  smoothY)
    		smoothX =  np.append(smoothX, [smoothX[-1]])
    		smoothY =  np.append(smoothY, [smoothY[-1]])
        pltX = smoothX[max(frameID-lenTraj,  0):frameID]
        pltY = smoothY[max(frameID-lenTraj,  0):frameID]
        L = len(pltX)
        c = np.abs(colors[j])
        c = np.tile(c, L).reshape(L,4)
        c[:,3] = np.exp(5*np.linspace(0.01,1,L))/np.exp(5)
    	plt.scatter(pltX, pltY,s=0.2, color=c)
    	plt.gca().set_aspect('equal')

    smoothX = np.convolve(np.array(quad)[S:F,0], np.ones((movingAverage,))/movingAverage, mode='valid')
    smoothY = np.convolve(np.array(quad)[S:F,1], np.ones((movingAverage,))/movingAverage, mode='valid')
    for i in range(3):
        smoothX =  np.append([smoothX[0]],  smoothX)
        smoothY =  np.append([smoothY[0]],  smoothY)
        smoothX =  np.append(smoothX, [smoothX[-1]])
        smoothY =  np.append(smoothY, [smoothY[-1]])
    pltX = smoothX[max(frameID-lenTraj,  0):frameID]
    pltY = smoothY[max(frameID-lenTraj,  0):frameID]
    L = len(pltX)
    c = (0., 0., 0., 1.0)
    c = np.tile(c, L).reshape(L,4)
    c[:,3] = np.exp(5*np.linspace(0.01,1,L))/np.exp(5)
    plt.scatter(pltX, pltY,s=0.2, color=c)
    plt.gca().set_axis_off()
    plt.savefig(save+'trajMovie/traj'+str(frameID).zfill(4)+'.png', format='png', bbox_inches='tight',  dpi = 300)




    frameID +=1

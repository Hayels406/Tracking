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

frameID = 0




cap = cv2.VideoCapture(videoLocation)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print 'You have', length, 'frames', videoLocation

if (videoLocation.rfind('data') > 0) and (videoLocation.rfind('CaseJ2') > 0):
    blackSheepID = True
    print 'Skipping first 10 frames'
    while(frameID <= 10):
        ret, frame = cap.read()
        frameID += 1

if (videoLocation.rfind('data') > 0) and (videoLocation.rfind('CaseH2') > 0):
    blackSheepID = True
    print 'Skipping first 19 frames'
    while(frameID <= 19):
        ret, frame = cap.read()
        frameID += 1

if (videoLocation.rfind('data') > 0) and (videoLocation.rfind('CaseH3') > 0):
    blackSheepID = True

if (videoLocation.rfind('data') > 0) and (videoLocation.rfind('CaseJ') > 0):
    blackSheepID = False
    save = '/data/b1033128/Tracking/'+video[:-4] +'1/'
    print 'Skipping first 15 frames'
    while(frameID <= 15):
        ret, frame = cap.read()
        frameID += 1

if frameID > 0:
    frameID = 0
data = glob(save+'Final-loc*')[-1]
sheep = np.load(data)
if blackSheepID:
    blackSheepData = glob(save+'Final-blackSheep*')[-1]
    blackSheep = np.load(blackSheepData)
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

    delay = 0
    if (video[:-4] == 'CaseJ') and (frameID > 80):
        delay = 80
    
    smoothX = np.convolve(np.array(quad)[S:F,0], np.ones((movingAverage,))/movingAverage, mode='valid')
    smoothY = np.convolve(np.array(quad)[S:F,1], np.ones((movingAverage,))/movingAverage, mode='valid')
    for i in range(3):
        smoothX =  np.append([smoothX[0]],  smoothX)
        smoothY =  np.append([smoothY[0]],  smoothY)
        smoothX =  np.append(smoothX, [smoothX[-1]])
        smoothY =  np.append(smoothY, [smoothY[-1]])
    pltX = smoothX[max(frameID-lenTraj-delay,  0):frameID-delay]
    pltY = smoothY[max(frameID-lenTraj-delay,  0):frameID-delay]
    L = len(pltX)
    c = (0., 0., 0., 1.0)
    c = np.tile(c, L).reshape(L,4)
    c[:,3] = np.exp(5*np.linspace(0.01,1,L))/np.exp(5)
    if (video[:-4] == 'CaseJ') and (frameID > 80):
        plt.scatter(pltX, pltY, s=0.2, color=c)
    elif video[:-4] != 'CaseJ':
        plt.scatter(pltX, pltY, s=0.2, color=c)

    if blackSheepID:
        smoothX = np.convolve(np.array(blackSheep)[S:F,0], np.ones((movingAverage,))/movingAverage, mode='valid')
        smoothY = np.convolve(np.array(blackSheep)[S:F,1], np.ones((movingAverage,))/movingAverage, mode='valid')
        for i in range(3):
            smoothX =  np.append([smoothX[0]],  smoothX)
            smoothY =  np.append([smoothY[0]],  smoothY)
            smoothX =  np.append(smoothX, [smoothX[-1]])
            smoothY =  np.append(smoothY, [smoothY[-1]])
        pltX = smoothX[max(frameID-lenTraj,  0):frameID]
        pltY = smoothY[max(frameID-lenTraj,  0):frameID]
        L = len(pltX)
        c = np.abs(colors[0])
        c = np.tile(c, L).reshape(L,4)
        c[:,3] = np.exp(5*np.linspace(0.01,1,L))/np.exp(5)
        plt.scatter(pltX, pltY,s=0.2, color=c)

    plt.gca().set_axis_off()
    plt.savefig(save+'trajMovie/traj'+str(frameID).zfill(4)+'.png', format='png', bbox_inches='tight',  dpi = 300)

    frameID +=1


os.system('for a in '+save+'trajMovie/*.png; do convert -trim "$a" "$a"; done')
os.system('ffmpeg -framerate 25 -pattern_type glob -i "'+save+'trajMovie/traj0*.png" -c:v libx264 -r 30 -pix_fmt yuv420p -vf scale=1200:1180 "'+save+'trajMovie.mp4"')

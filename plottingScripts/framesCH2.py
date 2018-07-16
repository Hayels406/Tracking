import numpy as np
import cv2
import os
import sys
from skimage import measure
import scipy.ndimage as ndimage
from imutils import contours
import imutils as im

import matplotlib.pyplot as plt
from sklearn.mixture import BayesianGaussianMixture as bgm
from sklearn.mixture import GaussianMixture as gm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from collections import Counter
import copy
from glob import glob
from scipy.stats import norm

from trackingFunctions_ngs54_changes import *

if os.getcwd().rfind('hayley') > 0:
    videoLocation = '/users/hayleymoore/Documents/PhD/Tracking/CaseH2.mov'
    save = '/users/hayleymoore/Documents/PhD/Tracking/CaseH2/'
elif os.getcwd().rfind('Uni') > 0:
    videoLocation = '/home/b1033128/Documents/CaseH2.mov'
    save = '/home/b1033128/Documents/CaseH2/'
    dell = True
    brk = False
else:#Kiel
    videoLocation = '/data/b1033128/Videos/CaseH2.mov'
    save = '/data/b1033128/Tracking/CaseH2/'
    dell = False

dell = True
brk = False
init = False

plot = 's'
tlPercent = 0.995#float(sys.argv[4])
tuPercent = 0.1#float(sys.argv[5])

quadDark = 100.
bsDark = 0.5
weight = 0.3
gamma = 1.5#float(sys.argv[3])

sheepLocations = []
blackSheepLocations = []
quadLocation = []
sheepCov = []
blackSheepCov = []
frameID = 0
cropVector = [100,400,800,1300]
quadCrop   = [700,1200,900,1400]
bsCrop = [375,725,475,825]

cap = cv2.VideoCapture(videoLocation)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print 'You have', length, 'frames', videoLocation

print 'Skipping first 19 frames while the sheep are hiding under the tree'
while(frameID <= 19):
    ret, frame = cap.read()
    frameID += 1
skip = frameID
if frameID > 0:
    frameID = 0

while(frameID <= length):
    plt.close('all')
    ret, frame = cap.read()
    if ret == True:
        objectLocations = []
        frameCov = []
        assignmentVec = []
        full = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        if frameID == 0:
            print 0
            plt.imshow(full)
            plt.gca().set_axis_off()
            plt.savefig(save+'frames/0.png', bbox_inches='tight')
        elif frameID == 100:
            print 100
            plt.imshow(full)
            plt.gca().set_axis_off()
            plt.savefig(save+'frames/100.png', bbox_inches='tight')
        elif frameID == 300:
            print 300
            plt.imshow(full)
            plt.gca().set_axis_off()
            plt.savefig(save+'frames/300.png', bbox_inches='tight')
        elif frameID == 500:
            print 500
            plt.imshow(full)
            plt.gca().set_axis_off()
            plt.savefig(save+'frames/500.png', bbox_inches='tight')
    frameID += 1

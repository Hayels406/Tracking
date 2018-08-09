import numpy as np
import cv2
import os
import sys
from skimage import measure
import scipy.ndimage as ndimage
from imutils import contours
import imutils as im
#import matplotlib as mpl
#mpl.use('Agg')

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

from trackingFunctions import *


sys.argv = ['CJ1.py', '0']
execfile('CJ1.py')

ymax, xmax, _ = np.shape(full)

plt.ion()
plt.imshow(full)
fence1 = np.array([ [289,  0],
                    [406,  1125],
                    [265,  1365],
                    [114,  ymax],
                    [265,  1365],
                    [389,  ymax]  ])

fence2 = np.array([ [2349,  0],
                    [2225,  1030],
                    [1975,  ymax]  ])
plt.plot(fence1[:-2,0], fence1[:-2,1], color='k', marker='.')
plt.plot(fence1[-2:,0], fence1[-2:,1], color='k', marker='.')
plt.plot(fence2[:,0], fence2[:,1], color='k', marker='.')
plt.scatter(fence1[:,0], fence1[:,1], color='r')
plt.scatter(fence2[:,0], fence2[:,1], color='r')

ax = plt.gca()
ax.add_artist(plt.Circle((307,143), 64, color='g'))
ax.add_artist(plt.Circle((330,460), 64, color='g'))
ax.add_artist(plt.Circle((396,788), 39, color='g'))
ax.add_artist(plt.Circle((355,1178), 39, color='g'))
ax.add_artist(plt.Circle((232,1510), 124, color='g'))

plt.xlim(0,xmax)
plt.ylim(ymax,0)
plt.gca().set_aspect('equal')
plt.gca().set_axis_off()
plt.savefig(save+'features.png')
os.system('convert '+save+'features.png -trim '+save+'features.png')

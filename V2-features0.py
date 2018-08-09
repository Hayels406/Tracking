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


sys.argv = ['CH2.py', '0']
execfile('CH2.py')

plt.ion()
plt.imshow(full)
corners = np.array([[2704,  463],
                    [2174,  517],
                    [2063,  371],
                    [1952,  364],
                    [1717,  374],
                    [1389,  390],
                    [1400,  578],
                    [1389,  390],
                    [1098,  401],
                    [723,   357],
                    [600,   363],
                    [600,   288],
                    [89,    307],
                    [11,    1502],
                    [494,   1439],
                    [894,   1485],
                    [1182,  1442],
                    [1275,  1419],
                    [1462,  1387],
                    [1421,  893],
                    [1462,  1387],
                    [1968,  1293],
                    [2036,  1515],
                    [1408,  691],
                    [1418,  826]  ])
plt.plot(corners[:4,0], corners[:4,1], color='k', marker='.')
plt.plot(corners[4:9,0], corners[4:9,1], color='k', marker='.')
plt.plot(corners[9:-2,0], corners[9:-2,1], color='k', marker='.')
plt.plot(corners[-2:,0], corners[-2:,1], color='k', marker='.')
plt.scatter(corners[:,0], corners[:,1], color='r')

ax = plt.gca()
ax.add_artist(plt.Circle((1838,369), 110, color='g'))
ax.add_artist(plt.Circle((903,396), 190, color='g'))
ax.add_artist(plt.Circle((377,290), 45, color='g'))
ax.add_artist(plt.Circle((200,450), 120, color='g'))
ax.add_artist(plt.Circle((133,665), 50, color='g'))
ax.add_artist(plt.Circle((196,890), 85, color='g'))
ax.add_artist(plt.Circle((110,1100), 125, color='g'))
ax.add_artist(plt.Circle((173,1291), 80, color='g'))
ax.add_artist(plt.Circle((418,1485), 65, color='g'))
ax.add_artist(plt.Circle((795,1155), 125, color='g'))
ax.add_artist(plt.Circle((795,1440), 70, color='g'))
ax.add_artist(plt.Circle((1075,1450), 40, color='g'))
ax.add_artist(plt.Circle((1416,1378), 50, color='g'))
ax.add_artist(plt.Circle((2150,1290), 105, color='g'))
ax.add_artist(plt.Circle((2529,1335), 220, color='g'))
ax.add_artist(plt.Circle((2240,1455), 165, color='g'))

plt.xlim(0,2704)
plt.ylim(1520,0)
plt.gca().set_aspect('equal')
plt.gca().set_axis_off()
plt.savefig(save+'features.png')
os.system('convert '+save+'features.png -trim '+save+'features.png')

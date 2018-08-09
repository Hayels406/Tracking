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


sys.argv = ['CH1.py', '0']
execfile('CH1.py')
###End
ymax, xmax, _ = np.shape(full)

plt.ion()
plt.imshow(full)
#fence = np.array([ [265,  1365],
                    #[389,  ymax]  ])
#
#plt.plot(fence[:,0], fence[:,1], color='k')
#
#ax = plt.gca()
#ax.add_artist(plt.Circle((1013,365), 110, color='g'))
#ax.add_artist(plt.Circle((991,148), 200, color='g'))
#ax.add_artist(plt.Circle((1245,116), 110, color='g'))
#ax.add_artist(plt.Circle((777,125), 117, color='g'))
#ax.add_artist(plt.Circle((644,144), 88, color='g'))
#ax.add_artist(plt.Circle((530,131), 86, color='g'))
#ax.add_artist(plt.Circle((432,108), 78, color='g'))
#ax.add_artist(plt.Circle((362,69), 74, color='g'))
#ax.add_artist(plt.Circle((242,26), 43, color='g'))
ax = plt.gca()
ax.plot([0, 2517, 2704], [1383, 1147, 1082], color = 'k')
ax.add_artist(plt.Circle((730,152), 110, color='g'))

plt.xlim(0,xmax)
plt.ylim(ymax,0)
plt.gca().set_aspect('equal')
plt.gca().set_axis_off()
plt.savefig(save+'features.png')
os.system('convert '+save+'features.png -trim '+save+'features.png')

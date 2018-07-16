import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
from scipy.spatial.distance import mahalanobis
import cv2
from scipy.stats import gaussian_kde



sys.argv = ['CH2.py', '19', '0']
execfile('CH2.py')



X = np.linspace(0,1,1000)

#gamma correction
R = np.copy(fullCropped)[:,:,0]/255.
G = np.copy(fullCropped)[:,:,1]/255.
B = np.copy(fullCropped)[:,:,2]/255.


for gamma in [1, 1.5, 2, 2.5, 3]:
    print gamma
    gCor = ((R**gamma + G**gamma + B**gamma)/3.).flatten()
    kernel = gaussian_kde(methods[i])
    dens = kernel(X)
    plt.plot(X, dens, label = '$\gamma =$ '+str(gamma))
    np.savetxt(save+'pgfData/gamma'+str(int(gamma*10.))+'.txt', np.append(X.reshape(1000,1), dens.reshape(1000,1), axis = 1))




os.system('for a in '+save+'grey/*.pdf; do pdfcrop "$a" "$a"; done')

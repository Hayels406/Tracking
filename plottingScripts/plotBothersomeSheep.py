import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.spatial.distance import mahalanobis
import cv2
from trackingFunctions import bivariateNormal
from trackingFunctions import predictKalman

def plotSheepCov(sCov, objectLoc, cropx, cropy):
        for i in range(len(sCov)):
                xx, yy = np.meshgrid(np.arange(objectLoc[i][0]-30-cropx,objectLoc[i][0]+30-cropx,0.1), np.arange(objectLoc[i][1]-30-cropy,objectLoc[i][1]+30-cropy, 0.1))
                z = bivariateNormal(xx, yy, objectLoc[i][0]-cropx, objectLoc[i][1]-cropy, sCov[i][0], sCov[i][1], sCov[i][2])
                plt.contour(xx,yy,z,cmap='plasma')



t = 75

sys.argv = ['getCoordsVideo.py', t, 'N']
execfile('getCoordsVideo.py')


cropped = np.copy(fullCropped)

plt.ion()
plt.imshow(cropped)
plt.xlim(400, 460)
plt.ylim(840, 780)
plt.scatter(np.array(sheepLocations)[-1,[133,137,138,140],0]-cropX, np.array(sheepLocations)[-1,[133,137,138,140],1]-cropY)

l, d = predictKalman(np.array(sheepLocations)[:-1][:,[133,137,138,140]])

plt.scatter(l[:,0]-cropX, l[:,1]-cropY)

cov = d[:,:2,:2].flatten().reshape(4,4)[:,[0,1,-1]]
s_x = np.sqrt(cov[:,0])
s_y = np.sqrt(cov[:,-1])
rho = cov[:,1]/(s_x*s_y)

scov = np.append(np.append(s_x.reshape(4,1), s_y.reshape(4,1), axis = 1),rho.reshape(4,1), axis = 1)

plotSheepCov(scov, l.tolist(), cropX, cropY)
plt.savefig(save+'bothersome/'+str(t)+'.pdf', bbox_inches='tight', pad_inches=0, format = 'pdf', dpi=300)

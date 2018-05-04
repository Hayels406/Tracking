import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.spatial.distance import mahalanobis
import cv2


def plotSheepCov(sCov, objectLoc, cropx, cropy):
        for i in range(len(sCov)):
                xx, yy = np.meshgrid(np.arange(objectLoc[i][0]-30-cropx,objectLoc[i][0]+30-cropx,0.1), np.arange(objectLoc[i][1]-30-cropy,objectLoc[i][1]+30-cropy, 0.1))
                z = bivariateNormal(xx, yy, objectLoc[i][0]-cropx, objectLoc[i][1]-cropy, sCov[i][0], sCov[i][1], sCov[i][2])
                plt.contour(xx,yy,z,cmap='plasma')




sys.argv = ['getCoordsVideo.py', 10, 'N']
execfile('getCoordsVideo.py')


cropped = np.copy(fullCropped)
greyScale = np.copy(cropped[:,:,2])

plt.imshow(fullCropped)
plotSheepCov(frameCov, objectLocations, cropX, cropY)
plt.gca().set_axis_off()
plt.xlim(225,425)
plt.ylim(225,25)
plt.savefig(save+'sheepCov/colour.pdf', bbox_inches='tight', pad_inches=0, format = 'pdf', dpi=300)
plt.close('all')


plt.imshow(grey, cmap='gray')
plotSheepCov(frameCov, objectLocations, cropX, cropY)
plt.gca().set_axis_off()
plt.xlim(225,425)
plt.ylim(225,25)
plt.savefig(save+'sheepCov/grey.pdf', bbox_inches='tight', pad_inches=0, format = 'pdf', dpi=300)
plt.close('all')

os.system('for a in '+save+'sheepCov/*.pdf; do pdfcrop "$a" "$a"; done')

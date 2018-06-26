import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
from scipy.spatial.distance import mahalanobis
import cv2

def mahalDist(image):
    blackSheepImage = image
    shape = np.shape(blackSheepImage)
    noElements = np.product(np.array(np.shape(blackSheepImage)))
    fullArray = np.transpose(blackSheepImage.reshape(noElements/3,3))

    Cov = np.cov(fullArray)
    backgroundElem = np.product(np.array(np.shape(blackSheepImage[50:, 50:,:])))
    m = (blackSheepImage[50:, 50:,:]).reshape(backgroundElem/3, 3).mean(axis = 0)
    d = map(lambda i:mahalanobis(m, fullArray[:,i], np.linalg.inv(Cov)), range(np.shape(fullArray)[1]))

    mahalDist = np.array(d).reshape(*shape[:-1])

    plt.imshow(mahalDist, cmap = 'gray')
    return mahalDist




sys.argv = ['CH2.py', '0', '0']
execfile('CH2.py')
blackSheepImage = full[725:825,375:475]
blackSheepImage = blackSheepImage[:,:,::-1]

#red channel
plt.imshow(np.copy(blackSheepImage)[:,:,0], cmap = 'gray')
plt.gca().set_axis_off()
plt.savefig(save+'blackSheep/red.pdf', bbox_inches='tight', pad_inches=0, format = 'pdf', dpi=300)
plt.close('all')
np.save(save+'blackSheep/red.npy', np.copy(blackSheepImage)[:,:,0].flatten())

#green channel
plt.imshow(np.copy(blackSheepImage)[:,:,1], cmap = 'gray')
plt.gca().set_axis_off()
plt.savefig(save+'blackSheep/green.pdf', bbox_inches='tight', pad_inches=0, format = 'pdf', dpi=300)
plt.close('all')
np.save(save+'blackSheep/green.npy', np.copy(blackSheepImage)[:,:,1].flatten())

#blue channel
plt.imshow(np.copy(blackSheepImage)[:,:,2], cmap = 'gray')
plt.gca().set_axis_off()
plt.savefig(save+'blackSheep/blue.pdf', bbox_inches='tight', pad_inches=0, format = 'pdf', dpi=300)
plt.close('all')
np.save(save+'blackSheep/blue.npy', np.copy(blackSheepImage)[:,:,2].flatten())

#product
plt.imshow(np.product(np.copy(blackSheepImage), axis = 2), cmap = 'gray')
plt.gca().set_axis_off()
plt.savefig(save+'blackSheep/product.pdf', bbox_inches='tight', pad_inches=0, format = 'pdf', dpi=300)
plt.close('all')
np.save(save+'blackSheep/product.npy', np.product(np.copy(blackSheepImage), axis = 2).flatten())

#gamma correction
R = np.copy(blackSheepImage)[:,:,0]/255.
G = np.copy(blackSheepImage)[:,:,1]/255.
B = np.copy(blackSheepImage)[:,:,2]/255.

gamma = .5
gCor = (R**gamma + G**gamma + B**gamma)/3.
plt.imshow(gCor, cmap = 'gray')
plt.gca().set_axis_off()
plt.savefig(save+'blackSheep/gammaCorrection.pdf', bbox_inches='tight', pad_inches=0, format = 'pdf', dpi=300)
plt.close('all')
np.save(save+'blackSheep/gammaCorrection.npy', gCor.flatten())


#difference G-R
plt.imshow(np.copy(blackSheepImage)[:,:,1] - np.copy(blackSheepImage)[:,:,0], cmap='gray')
plt.gca().set_axis_off()
plt.savefig(save+'blackSheep/G-R.pdf', bbox_inches='tight', pad_inches=0, format = 'pdf', dpi=300)
plt.close('all')
np.save(save+'blackSheep/GR.npy', (- np.copy(blackSheepImage)[:,:,0] + np.copy(blackSheepImage)[:,:,1]).flatten())

#difference G-B
plt.imshow(np.copy(blackSheepImage)[:,:,1] - np.copy(blackSheepImage)[:,:,2], cmap='gray')
plt.gca().set_axis_off()
plt.savefig(save+'blackSheep/G-B.pdf', bbox_inches='tight', pad_inches=0, format = 'pdf', dpi=300)
plt.close('all')
np.save(save+'blackSheep/GB.npy', (- np.copy(blackSheepImage)[:,:,2] + np.copy(blackSheepImage)[:,:,1]).flatten())

#difference R-B
plt.imshow(np.copy(blackSheepImage)[:,:,0] - np.copy(blackSheepImage)[:,:,2], cmap='gray')
plt.gca().set_axis_off()
plt.savefig(save+'blackSheep/R-B.pdf', bbox_inches='tight', pad_inches=0, format = 'pdf', dpi=300)
plt.close('all')
np.save(save+'blackSheep/RB.npy', (- np.copy(blackSheepImage)[:,:,2] + np.copy(blackSheepImage)[:,:,0]).flatten())

#mahalanobis
mD = mahalDist(np.copy(blackSheepImage))
plt.gca().set_axis_off()
plt.savefig(save+'blackSheep/mahalanobis.pdf', bbox_inches='tight', pad_inches=0, format = 'pdf', dpi=300)
plt.close('all')
np.save(save+'blackSheep/mahalanobis.npy', mD.flatten())

#conventional
plt.imshow(cv2.cvtColor(np.copy(blackSheepImage),cv2.COLOR_RGB2GRAY), cmap = 'gray')
plt.gca().set_axis_off()
plt.savefig(save+'blackSheep/grey.pdf', bbox_inches='tight', pad_inches=0, format = 'pdf', dpi=300)
plt.close('all')
np.save(save+'blackSheep/grey.npy', cv2.cvtColor(np.copy(blackSheepImage),cv2.COLOR_RGB2GRAY).flatten())

os.system('for a in '+save+'blackSheep/*.pdf; do pdfcrop "$a" "$a"; done')

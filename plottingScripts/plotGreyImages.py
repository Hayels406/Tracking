import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
from scipy.spatial.distance import mahalanobis
import cv2

def mahalDist(image):
    fullCropped = image
    shape = np.shape(fullCropped)
    noElements = np.product(np.array(np.shape(fullCropped)))
    fullArray = np.transpose(fullCropped.reshape(noElements/3,3))

    Cov = np.cov(fullArray)
    backgroundElem = np.product(np.array(np.shape(fullCropped[:1300, :500,:])))
    m = (fullCropped[:1300, :500,:]).reshape(backgroundElem/3, 3).mean(axis = 0)
    d = map(lambda i:mahalanobis(m, fullArray[:,i], np.linalg.inv(Cov)), range(np.shape(fullArray)[1]))

    mahalDist = np.array(d).reshape(*shape[:-1])

    plt.imshow(mahalDist, cmap = 'gray')
    return mahalDist




sys.argv = ['getCoordsVideo.py', 10]
execfile('getCoordsVideo.py')


#red channel
plt.imshow(np.copy(fullCropped)[:,:,0], cmap = 'gray')
plt.gca().set_axis_off()
plt.savefig(save+'grey/red.pdf', bbox_inches='tight', pad_inches=0, format = 'pdf', dpi=300)
plt.close('all')
np.save(save+'grey/red.npy', np.copy(fullCropped)[:,:,0].flatten())

#green channel
plt.imshow(np.copy(fullCropped)[:,:,1], cmap = 'gray')
plt.gca().set_axis_off()
plt.savefig(save+'grey/green.pdf', bbox_inches='tight', pad_inches=0, format = 'pdf', dpi=300)
plt.close('all')
np.save(save+'grey/green.npy', np.copy(fullCropped)[:,:,1].flatten())

#blue channel
plt.imshow(np.copy(fullCropped)[:,:,2], cmap = 'gray')
plt.gca().set_axis_off()
plt.savefig(save+'grey/blue.pdf', bbox_inches='tight', pad_inches=0, format = 'pdf', dpi=300)
plt.close('all')
np.save(save+'grey/blue.npy', np.copy(fullCropped)[:,:,2].flatten())

#product
plt.imshow(np.product(np.copy(fullCropped), axis = 2), cmap = 'gray')
plt.gca().set_axis_off()
plt.savefig(save+'grey/product.pdf', bbox_inches='tight', pad_inches=0, format = 'pdf', dpi=300)
plt.close('all')
np.save(save+'grey/product.npy', np.product(np.copy(fullCropped), axis = 2).flatten())

#gamma correction
R = np.copy(fullCropped)[:,:,0]/255.
G = np.copy(fullCropped)[:,:,1]/255.
B = np.copy(fullCropped)[:,:,2]/255.

gamma = 5.
gCor = (R**gamma + G**gamma + B**gamma)/3.
plt.imshow(gCor, cmap = 'gray')
plt.gca().set_axis_off()
plt.savefig(save+'grey/gammaCorrection.pdf', bbox_inches='tight', pad_inches=0, format = 'pdf', dpi=300)
plt.close('all')
np.save(save+'grey/gammaCorrection.npy', gCor.flatten())


#difference G-R
plt.imshow(- np.copy(fullCropped)[:,:,0] + np.copy(fullCropped)[:,:,1], cmap='gray')
plt.gca().set_axis_off()
plt.savefig(save+'grey/G-R.pdf', bbox_inches='tight', pad_inches=0, format = 'pdf', dpi=300)
plt.close('all')
np.save(save+'grey/GR.npy', (- np.copy(fullCropped)[:,:,0] + np.copy(fullCropped)[:,:,1]).flatten())

#mahalanobis
mD = mahalDist(np.copy(fullCropped))
plt.gca().set_axis_off()
plt.savefig(save+'grey/mahalanobis.pdf', bbox_inches='tight', pad_inches=0, format = 'pdf', dpi=300)
plt.close('all')
np.save(save+'grey/mahalanobis.npy', mD.flatten())

#conventional
plt.imshow(cv2.cvtColor(np.copy(fullCropped),cv2.COLOR_RGB2GRAY), cmap = 'gray')
plt.gca().set_axis_off()
plt.savefig(save+'grey/grey.pdf', bbox_inches='tight', pad_inches=0, format = 'pdf', dpi=300)
plt.close('all')
np.save(save+'grey/grey.npy', cv2.cvtColor(np.copy(fullCropped),cv2.COLOR_RGB2GRAY).flatten())

os.system('for a in '+save+'grey/*.pdf; do pdfcrop "$a" "$a"; done')

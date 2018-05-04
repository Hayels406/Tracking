import numpy as np
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
    backgroundElem = np.product(np.array(np.shape(fullCropped[800:, :200,:])))
    m = (fullCropped[800:, :200,:]).reshape(backgroundElem/3, 3).mean(axis = 0)
    d = map(lambda i:mahalanobis(m, fullArray[:,i], np.linalg.inv(Cov)), range(np.shape(fullArray)[1]))

    mahalDist = np.array(d).reshape(*shape[:-1])

    plt.imshow(mahalDist, cmap = 'gray')




sys.argv = ['getCoordsVideo.py', 10]
execfile('getCoordsVideo.py')


#red channel
plt.imshow(np.copy(fullCropped)[:,:,0], cmap = 'gray')
plt.gca().set_axis_off()
plt.savefig(save+'grey/red.pdf', bbox_inches='tight', pad_inches=0, format = 'pdf', dpi=300)
plt.close('all')

#gree channel
plt.imshow(np.copy(fullCropped)[:,:,1], cmap = 'gray')
plt.gca().set_axis_off()
plt.savefig(save+'grey/green.pdf', bbox_inches='tight', pad_inches=0, format = 'pdf', dpi=300)
plt.close('all')

#blue channel
plt.imshow(np.copy(fullCropped)[:,:,2], cmap = 'gray')
plt.gca().set_axis_off()
plt.savefig(save+'grey/blue.pdf', bbox_inches='tight', pad_inches=0, format = 'pdf', dpi=300)
plt.close('all')

#product
plt.imshow(np.product(np.copy(fullCropped), axis = 2), cmap = 'gray')
plt.gca().set_axis_off()
plt.savefig(save+'grey/product.pdf', bbox_inches='tight', pad_inches=0, format = 'pdf', dpi=300)
plt.close('all')

#exponential
R = np.copy(fullCropped)[:,:,0]
G = np.copy(fullCropped)[:,:,1]
B = np.copy(fullCropped)[:,:,2]

m = 5.
expon = np.exp(m*np.copy(R)/255.) + np.exp(m*np.copy(G)/255.) + np.exp(m*np.copy(B)/255.)
plt.imshow(expon/np.max(expon), cmap = 'gray')
plt.gca().set_axis_off()
plt.savefig(save+'grey/exponential.pdf', bbox_inches='tight', pad_inches=0, format = 'pdf', dpi=300)
plt.close('all')

#difference R-G
plt.imshow(R-G, cmap='gray')
plt.gca().set_axis_off()
plt.savefig(save+'grey/R-G.pdf', bbox_inches='tight', pad_inches=0, format = 'pdf', dpi=300)
plt.close('all')

#mahalanobis
mahalDist(np.copy(fullCropped))
plt.gca().set_axis_off()
plt.savefig(save+'grey/mahalanobis.pdf', bbox_inches='tight', pad_inches=0, format = 'pdf', dpi=300)
plt.close('all')

#conventional
plt.imshow(cv2.cvtColor(np.copy(fullCropped),cv2.COLOR_RGB2GRAY), cmap = 'gray')
plt.gca().set_axis_off()
plt.savefig(save+'grey/grey.pdf', bbox_inches='tight', pad_inches=0, format = 'pdf', dpi=300)
plt.close('all')


os.system('for a in '+save+'grey/*.pdf; do pdfcrop "$a" "$a"; done')

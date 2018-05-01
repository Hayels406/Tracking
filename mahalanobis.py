import numpy as np
import cv2
import os
import sys
import imutils as im
if len(sys.argv) == 1:
    import matplotlib as mpl
    mpl.use('agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import mahalanobis

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
	
    return mahalDist #figsize 20*20 dpi =300 then in gimp resize to match others 

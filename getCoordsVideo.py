import numpy as np
import cv2
import os
import sys
from skimage import measure
import scipy.ndimage as ndimage
from imutils import contours
import imutils as im
import matplotlib as mpl

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

from trackingFunctions_ngs54_changes import *

if os.getcwd().rfind('hayley') > 0:
    videoLocation = '/users/hayleymoore/Documents/PhD/Tracking/throughFenceRL.mp4'
    save = '/users/hayleymoore/Documents/PhD/Tracking/throughFenceRL/'
elif os.getcwd().rfind('Uni') > 0:
    videoLocation = '/home/b1033128/Documents/throughFenceRL.mp4'
    save = '/home/b1033128/Documents/throughFenceRL/'
    dell = True
    brk = False
else:#Kiel
    videoLocation = '/data/b1033128/Videos/throughFenceRL.mp4'
    save = '/data/b1033128/Tracking/throughFenceRL/'
    dell = False

dell = True
brk = False
init = False

runUntil = int(sys.argv[1])
toSkip = sys.argv[2]
plot = 's'
tlPercent = 0.995#float(sys.argv[4])
tuPercent = 0.1#float(sys.argv[5])

quadDark = 100.
weight = 0.3
gamma = 3#float(sys.argv[3])

sheepLocations = []
sheepVelocity = []
quadLocation = []
sheepCov = []
frameID = 0
cropVector = [1000,1000,2000,2028]
quadCrop   = [2000,1500,2200,1700]

cap = cv2.VideoCapture(videoLocation)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print 'You have', length, 'frames', videoLocation


if (videoLocation.rfind('data') > 0) and (videoLocation.rfind('throughFenceRL') > 0):
    print 'Skipping first 15 frames'
    while(frameID <= 15):
        ret, frame = cap.read()
        frameID += 1
    if frameID > 0:
        frameID = 0

print 'Skipping first '+toSkip+' frames'
while(frameID < int(toSkip)):
    ret, frame = cap.read()
    frameID += 1
skip = frameID
if frameID > 0:
    frameID = 0

while(frameID <= runUntil):
    plt.close('all')
    ret, frame = cap.read()
    if ret == True:
        assignmentVec = []
        full = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        quadLocation, quadCrop = getQuad(full, quadLocation, quadCrop, quadDark, frameID)
        fullCropped, cropVector = movingCropTFRL(frameID, full, sheepLocations, cropVector)
        cropX, cropY, cropXMax, cropYMax = cropVector
        R = fullCropped[:,:,0]/255.
        G = fullCropped[:,:,1]/255.
        B = fullCropped[:,:,2]/255.

        grey = (R**gamma + G**gamma + B**gamma)/3.

        gmm = gm(n_components=3, covariance_type='full', weights_init=[0.8, 0.1,0.1], means_init=[[0.2],[0.6],[0.9]]).fit((grey.flatten()).reshape(-1,1))
        lower = norm(loc = gmm.means_[0], scale = np.sqrt(gmm.covariances_[0]))
        upper = norm(loc = gmm.means_[-1], scale = np.sqrt(gmm.covariances_[-1]))
        darkTolerance = lower.ppf(tlPercent)[0][0]
        darkTolerance2 = upper.ppf(tuPercent)[0][0]


        grey2 = np.copy(grey)
        grey2[grey2 < darkTolerance] = 0.0
        grey2[grey2 > darkTolerance2] = 1.

        img = cv2.GaussianBlur(grey2,(5,5),2)

        maxfilter = ndimage.maximum_filter(img, size=3)
        vel = []

        if frameID > 6:
            pass
            prediction_Objects, prediction_Distributions = predictKalman(np.array(sheepLocations))
            filtered, distImg = createBinaryImage(frameID, prediction_Objects, np.array(sheepCov), cropVector, maxfilter,darkTolerance, weight)
            #finalGrey = (np.array(distImg).sum(axis = 0))*np.copy(filtered)
            #finalGrey = finalGrey/np.max(finalGrey)
        else:
            prediction_Objects = []
            prediction_Distributions = []
            filtered, distImg = createBinaryImage(frameID, prediction_Objects, np.array(sheepCov), cropVector, maxfilter, darkTolerance)

        frameCov = []

        if plot != 'N':
            plt.imshow(filtered, cmap = 'gray')
            if frameID > 10:
                plt.scatter(np.array(prediction_Objects)[:,0]-cropX, np.array(prediction_Objects)[:,1]-cropY, s =1.)
            plt.gca().set_aspect('equal')
            plt.gca().set_axis_off()
            if plot == 's':
                plt.savefig(save+'filtered/'+str(frameID+skip).zfill(4), bbox_inches='tight')

        labels = measure.label(filtered, neighbors=8, background=0)
        labelPixels = map(lambda label: (labels == label).sum(), np.unique(labels))
        sizeSheep = np.percentile(labelPixels[1:],50)
        print("Estimating large sheep size at: "+str(sizeSheep))
        smallSheep = np.percentile(labelPixels[1:],5)
        print("Estimating small sheep size at: "+str(smallSheep))
        minPixels = smallSheep
        oneSheepPixels = sizeSheep

        objectLocations = []

        print 'Gamma', gamma, 't_l', str(round(tlPercent*100, 2))+'%', 't_u', str(round(tuPercent*100, 2))+'%'

        # loop over the unique components
        for label in np.unique(labels):
            check = 'On'
            # if this is the background label, ignore it
            if (labels == label).sum() == np.max(labelPixels):
                continue

            # otherwise, construct the label mask and count the
            # number of pixels
            labelMask = np.zeros(filtered.shape, dtype="uint8")
            labelMask[labels == label] = 1
            numPixels = labelPixels[label]

            # if the number of pixels in the component is sufficiently
            # large, then add it to our mask of "large blobs"
            if numPixels > minPixels:
                cnts = cv2.findContours(np.copy(labelMask), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1][0]
                ((cX, cY), radius) = cv2.minEnclosingCircle(cnts)
                rectangle = cv2.boundingRect(cnts)
                x,y,w,h = rectangle
                miniGrey = np.copy(filtered*grey2*labelMask)[y: y + h, x: x + w]
                miniImage = np.copy(fullCropped)[y:y+h,x:x+w]
                if frameID == 0:
                    if cX > 215: #fence limit
                        if numPixels < oneSheepPixels:
                            c = extractDensityCoordinates(miniGrey)
                            mm = bgm(n_components = 1, covariance_type='tied', random_state=1,max_iter=1000,tol=1e-6).fit(c.tolist())
                            objectLocations += (mm.means_ + [x,y]).tolist()
                            cov = mm.covariances_.flatten()[[0,1,-1]]
                            s_x = np.sqrt(cov[0])
                            s_y = np.sqrt(cov[2])
                            rho = cov[1]/(s_x*s_y)
                            frameCov += [[s_x, s_y, rho]]
                        else:
                            if init == False:
                                plt.clf()
                                ax1 = plt.subplot(1,2,1)
                                ax1.imshow(miniImage)
                                ax2 = plt.subplot(1,2,2)
                                ax2.imshow(fullCropped)
                                plt.scatter(x+w/2,y+h/2,color='r')
                                plt.pause(0.0001)
                                guessed = int(np.round(numPixels/np.percentile(labelPixels[1:],50)))
                                #text = raw_input("How many sheep in this mini image ["+str(guessed)+"]:")
                                text = ''
                                if text=='':
                                    number = guessed
                                else:
                                    number = int(text)
                            else:
                                number = int(initFile[initLoc])
                                initLoc += 1
                            if number==0:
                                continue

                            c = extractDensityCoordinates(miniGrey)
                            mm = bgm(n_components = number, covariance_type='tied', random_state=1,max_iter=1000,tol=1e-6).fit(c.tolist())
                            objectLocations += (mm.means_ + [x,y]).tolist()
                            cov = mm.covariances_.flatten()[[0,1,-1]]
                            s_x = np.sqrt(cov[0])
                            s_y = np.sqrt(cov[2])
                            rho = cov[1]/(s_x*s_y)
                            frameCov += [[s_x, s_y, rho]]*len(mm.means_)

                            if init == False:
                                ax1.scatter(mm.means_[:,0], mm.means_[:,1])
                                plt.pause(.5)
                elif frameID <= 6:
                    #if numPixels < oneSheepPixels:
                    #    c = extractDensityCoordinates(miniGrey)
                    #    mm = bgm(n_components = 1, covariance_type='tied', random_state=1,max_iter=1000,tol=1e-6).fit(c.tolist())
                    #    objectLocations += (mm.means_ + [x,y]).tolist()
                    #    cov = mm.covariances_.flatten()[[0,1,-1]]
                    #    s_x = np.sqrt(cov[0])
                    #    s_y = np.sqrt(cov[2])
                    #    rho = cov[1]/(s_x*s_y)
                    #    frameCov += [[s_x, s_y, rho]]
                    #else:
                    lastT = np.array(sheepLocations[-1])
                    dilation = cv2.dilate(labelMask,np.ones((5,5),np.uint8),iterations = 2)
                    _,Ids = getPredictedID(lastT, dilation, cropVector, rectangle)#getPreviousID(lastT, x, y, w, h, cropX, cropY)
                    k = len(Ids)
                    if k == 0:
                        continue
                    else:
                        coords =  extractDensityCoordinates(miniGrey)

                        mm = bgm(n_components = k, covariance_type='tied', random_state=1,max_iter=1000,tol=1e-6).fit(coords)
                        new_objects_bgm = (mm.means_ + [x,y])
                        cov = mm.covariances_.flatten()[[0,1,-1]]
                        s_x = np.sqrt(cov[0])
                        s_y = np.sqrt(cov[2])
                        rho = cov[1]/(s_x*s_y)
                        sCov_bgm = [[s_x, s_y, rho]]*len(mm.means_)

                        objectLocations += new_objects_bgm.tolist()
                        frameCov += sCov_bgm
                else:
                    dilation = cv2.dilate(labelMask,np.ones((5,5),np.uint8),iterations = 2)
                    pred_objects, Ids = getPredictedID(prediction_Objects, dilation,  cropVector, rectangle)
                    k = len(pred_objects)

                    if k == 1:
                        c = extractDensityCoordinates(miniGrey)
                        mm = bgm(n_components = 1, covariance_type='tied', random_state=1,max_iter=1000,tol=1e-6).fit(c.tolist())
                        objectLocations += (mm.means_ + [x,y]).tolist()
                        cov = mm.covariances_.flatten()[[0,1,-1]]
                        s_x = np.sqrt(cov[0])
                        s_y = np.sqrt(cov[2])
                        rho = cov[1]/(s_x*s_y)
                        frameCov += [[s_x, s_y, rho]]
                        assignmentVec += assignSheep([cX, cY], distImg, Ids, centre=[cX, cY])
                    elif k > 1:
                        coords =  extractDensityCoordinates(miniGrey)
                        mm = bgm(n_components = k, covariance_type='tied', random_state=1,max_iter=1000,tol=1e-6).fit(coords)
                        new_objects_bgm = (mm.means_ + [x,y])
                        cov = mm.covariances_.flatten()[[0,1,-1]]
                        s_x = np.sqrt(cov[0])
                        s_y = np.sqrt(cov[2])
                        rho = cov[1]/(s_x*s_y)
                        sCov_bgm = [[s_x, s_y, rho]]*len(mm.means_)
                        objectLocations += new_objects_bgm.tolist()
                        frameCov += sCov_bgm
                        assignmentVec += assignSheep(new_objects_bgm.tolist(), distImg, Ids, centre=[cX,  cY])

        objectLocations = np.array(objectLocations)

        objectLocations[:, 0] += cropX
        objectLocations[:, 1] += cropY
        objectLocations = objectLocations.tolist()

        if frameID == 0:
            N = len(objectLocations)

        if (frameID > 0) & (frameID <= 6):
            finalDist = cdist(sheepLocations[-1], objectLocations)
            _, assignmentVec = linear_sum_assignment(finalDist)

        finalLocations = organiseLocations(copy.deepcopy(objectLocations), copy.deepcopy(assignmentVec), frameID)
        finalCov = organiseLocations(copy.deepcopy(frameCov), copy.deepcopy(assignmentVec), frameID)

        l = len(finalLocations)
        if plot != 'N':
            plt.close('all')
            plt.imshow(fullCropped)
            plt.scatter(np.array(finalLocations)[:, 0]-cropX, np.array(finalLocations)[:, 1]-cropY, s = 1.)
            plt.gca().set_aspect('equal')
            plt.gca().set_axis_off()
            if plot == 's':
                plt.savefig(save+str(frameID+skip).zfill(4), bbox_inches='tight')
            else:
                plt.pause(15)

        sheepLocations = sheepLocations + [finalLocations]
        sheepVelocity = sheepVelocity + [vel]
        sheepCov = sheepCov + [np.array(finalCov).tolist()]


        print 'frameID: ' + str(skip+frameID)+ ', No. objects located:', l

        if l < N:
            brk = True
            print 'you lost sheep'
            break
        if l > N:
            brk = True
            print 'you gained sheep'
            break


        if np.mod(frameID,10) == 0:
            np.save(save+'loc-start'+str(skip)+'end'+str(frameID+skip), (np.array(sheepLocations), cropVector))
            np.save(save+'vel-start'+str(skip)+'end'+str(frameID+skip), np.array(sheepVelocity))
            np.save(save+'quad-start'+str(skip)+'end'+str(frameID+skip), np.array(quadLocation))
            np.save(save+'cov-start'+str(skip)+'end'+str(frameID+skip), np.array(sheepCov))

        plt.clf()
        frameID = frameID + 1

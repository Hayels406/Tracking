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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from collections import Counter
import copy
from glob import glob

from trackingFunctions import kmeansClustering
from trackingFunctions import iris
from trackingFunctions import predictEuler
from trackingFunctions import movingCrop
from trackingFunctions import createBinaryImage
from trackingFunctions import findVel
from trackingFunctions import assignSheep
from trackingFunctions import organiseLocations
from trackingFunctions import doCheck
from trackingFunctions import getPreviousID
from trackingFunctions import getPredictedID
from trackingFunctions import getQuad
from trackingFunctions import predictKalman
from trackingFunctions import bivariateNormal

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

plot = 's'
darkTolerance = 0.2
darkTolerance2 = 0.6
quadDark = 100.
sizeOfObject = 60
restart = 0

sheepLocations = []
sheepVelocity = []
quadLocation = []
sheepCov = []
frameID = 0
cropVector = [1000,1000,2000,2028]
quadCrop   = [2000,1500,2200,1700]

if len(glob(save+'init')) == 1:
    initFile = np.loadtxt(save+'init')
    initLoc = 0
    init = True
else:
    init = False


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

if restart > 0:
    sheepLocations, cropVector = np.load(save+'loc'+str(restart)+'.npy')
    sheepLocations =  map(np.array,  sheepLocations)
    sheepVelocity = np.load(save+'vel'+str(restart)+'.npy')
    sheepVelocity =  map(np.array, sheepVelocity)
    quadLocation = np.load(save+'quad'+str(restart)+'.npy')
    sheepCov = np.load(save+'cov'+str(restart)+'.npy')
    while(frameID <= restart):
        ret, frame = cap.read()
        print frameID
        frameID +=1
    N = len(sheepLocations[0])

if len(sys.argv) >= 2:
    runUntil = int(sys.argv[1])
else:
    runUntil = length

if len(sys.argv) == 3:
    plot = sys.argv[2]


while(frameID <= runUntil):
    plt.close('all')
    ret, frame = cap.read()
    if ret == True:
        assignmentVec = []
        full = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        quadLocation, quadCrop = getQuad(full, quadLocation, quadCrop, quadDark, frameID)

        fullCropped, cropVector = movingCrop(frameID, full, sheepLocations, cropVector)
        cropX, cropY, cropXMax, cropYMax = cropVector
        R = np.copy(fullCropped)[:,:,0]/255.
        G = np.copy(fullCropped)[:,:,1]/255.
        B = np.copy(fullCropped)[:,:,2]/255.

        gamma = 5.
        grey = (R**gamma + G**gamma + B**gamma)/3.

        grey2 = np.copy(grey)
        grey2[grey2 < darkTolerance] = 0.0
        grey2[grey2 > darkTolerance2] = 1.

        img = cv2.GaussianBlur(grey2,(5,5),2)

        maxfilter = ndimage.maximum_filter(img, size=3)
        vel = []

        if frameID > 6:
            prediction_Objects, prediction_Distributions = predictKalman(np.array(sheepLocations))
        else:
            prediction_Objects = []
            prediction_Distributions = []

        filtered, minPixels, oneSheepPixels, distImg = createBinaryImage(frameID, sizeOfObject, prediction_Objects, np.array(sheepCov), cropVector, maxfilter)

        frameCov = []

        if plot != 'N':
            plt.imshow(filtered, cmap = 'gray')
            if frameID > 6:
                plt.scatter(np.array(prediction_Objects)[:,0]-cropX, np.array(prediction_Objects)[:,1]-cropY, s =1.)
            plt.gca().set_aspect('equal')
            plt.gca().set_axis_off()
            if plot == 's':
                plt.savefig(save+'/filtered/'+str(frameID).zfill(4), bbox_inches='tight')

        labels = measure.label(filtered, neighbors=8, background=0)
        labelPixels = map(lambda label: (labels == label).sum(), np.unique(labels))

        objectLocations = []
        # loop over the unique components
        for label in np.unique(labels):
            check = 'On'
            # if this is the background label, ignore it
            if label == 0:
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
                if frameID == 0:
                    if numPixels < oneSheepPixels:
                        objectLocations += [[cX, cY]]
                        cov = np.cov(np.transpose(np.array(np.where(labelMask > 0)))[:,::-1], rowvar = False).flatten()[[0,1,-1]].tolist()
                        s_x = np.sqrt(cov[0])
                        s_y = np.sqrt(cov[2])
                        rho = cov[1]/(s_x*s_y)
                        frameCov += [[s_x, s_y, rho]]

                    else:
                        miniImage = np.copy(fullCropped)[y: y + h, x: x + w]
                        if init == False:
                            plt.clf()
                            plt.imshow(miniImage)
                            plt.pause(0.001)
                            text = raw_input("How many sheep in this mini image: ")
                            number = int(text)
                            plt.clf()
                        else:
                            number = int(initFile[initLoc])
                            initLoc += 1                        
                        coords = np.transpose(np.where(np.copy(maxfilter)[y: y+ h, x:x+w] > darkTolerance*2))[:,::-1] + np.array([x,y])
                        mm = bgm(n_components = number, covariance_type='tied').fit(coords)
                        objectLocations += mm.means_.tolist()
                        if init == False:
                            plt.imshow(miniImage)
                            plt.scatter(mm.means_[:,0]-x, mm.means_[:,1]-y)
                            plt.pause(0.5)
                        #miniImage = np.copy(img)[y: y + h, x: x + w]
                        #miniGrey = np.copy(grey)[y: y + h, x: x + w]

                        #k = -1

                        #new_objects_k, sCov_k = kmeansClustering(miniImage, miniGrey, numPixels, x, y, previous = k)
                        #new_objects_i, sCov_i = iris(miniImage, x, y)

                        #num_new_objects_k = np.shape(new_objects_k)[0]
                        #num_new_objects_i = np.shape(new_objects_i)[0]

                        #if num_new_objects_i == 1:
                        #    check = 'Off'
                        #    objectLocations += new_objects_k
                        #    frameCov += sCov_k

                        #elif num_new_objects_k == 1:
                        #    check = 'Off'
                        #    objectLocations += new_objects_i
                        #    frameCov += sCov_i

                        #elif num_new_objects_k == num_new_objects_i:
                        #    C = cdist(new_objects_i, new_objects_k)
                        #    row_ind, assignment = linear_sum_assignment(C)
                        #    av_dist = C[row_ind, assignment].sum()/num_new_objects_i

                            #if av_dist < 3.5:
                            #    check = 'Off'
                            #    objectLocations += new_objects_k
                            #    frameCov += sCov_k

                        #if check == 'On':
                        #    if init == True:
                        #        if initFile[initLoc] == 2:
                        #            objectLocations += new_objects_k
                        #            frameCov += sCov_k
                        #        elif initFile[initLoc] == 3:
                        #            objectLocations += new_objects_i
                        #            frameCov += sCov_i
                        #        initLoc += 1
                        #    else:
                        #        objectLocations = doCheck(fullCropped, objectLocations, cX, cY, img, new_objects_i, new_objects_k, rectangle, k)
                        #        if objectLocations[-1] == new_objects_k[-1]:
                        #            frameCov += sCov_k
                        #        elif objectLocations[-1] == new_objects_i[-1]:
                        #            frameCov += sCov_i
    
    
                        
    
                elif frameID <= 6:
                    if numPixels < oneSheepPixels:
                        objectLocations += [[cX, cY]]
                        cov = np.cov(np.transpose(np.array(np.where(labelMask > 0)))[:,::-1], rowvar = False).flatten()[[0,1,-1]].tolist()
                        s_x = np.sqrt(cov[0])
                        s_y = np.sqrt(cov[2])
                        rho = cov[1]/(s_x*s_y)
                        frameCov += [[s_x, s_y, rho]]
                    else:
                        miniImage = img[y: y + h, x: x + w]
                        miniGrey = np.copy(grey)[y: y + h, x: x + w]

                        lastT = np.array(sheepLocations[-1])
                        padding = 3
                        Ids = getPreviousID(lastT, x, y, w, h, cropX, cropY, padding)
                        k = len(Ids)

                        new_objects_k, sCov_k = kmeansClustering(miniImage, miniGrey, numPixels, x, y, previous = k)
                        new_objects_i, sCov_i = iris(miniImage, x, y)

                        num_new_objects_k = np.shape(new_objects_k)[0]
                        num_new_objects_i = np.shape(new_objects_i)[0]

                        if num_new_objects_k == num_new_objects_i:
                            C = cdist(new_objects_i, new_objects_k)
                            row_ind, assignment = linear_sum_assignment(C)
                            av_dist = C[row_ind, assignment].sum()/num_new_objects_i

                            if av_dist < 3.5:
                                check = 'Off'
                                objectLocations += new_objects_k
                                frameCov += sCov_k

                        else:
                            check = 'Off'
                            objectLocations += new_objects_k
                            frameCov += sCov_k

                        if check == 'On':
                            if init == True:
                                if initFile[initLoc] == 2:
                                    objectLocations += new_objects_k
                                    frameCov += sCov_k
                                elif initFile[initLoc] == 3:
                                    objectLocations += new_objects_i
                                    frameCov += sCov_i
                                initLoc += 1
                            else:
                                objectLocations = doCheck(fullCropped, objectLocations, cX, cY, img, new_objects_i, new_objects_k, rectangle, k)
                                if objectLocations[-1] == new_objects_k[-1]:
                                    frameCov += sCov_k
                                elif objectLocations[-1] == new_objects_i[-1]:
                                    frameCov += sCov_i

                else:
                    pred_objects, Ids = getPredictedID(prediction_Objects, np.copy(labelMask),  cropVector, rectangle)
                    k = len(pred_objects)
                    check = 'Off'

                    if (frameID == 16) & (label == np.unique(labels)[-1]):
                        objectLocations += [[cX, cY]]
                        assignmentVec += [141]
                        cov = np.cov(np.transpose(np.array(np.where(labelMask > 0)))[:,::-1], rowvar = False).flatten()[[0,1,-1]].tolist()
                        s_x = np.sqrt(cov[0])
                        s_y = np.sqrt(cov[2])
                        rho = cov[1]/(s_x*s_y)
                        frameCov += [[s_x, s_y, rho]]
                        for i in range(frameID):
                            sheepLocations[i] = np.append(sheepLocations[i], [[cX+cropX, cY+cropY]], axis = 0)
                            sheepCov[i] += [[s_x, s_y, rho]]
                        k = -1

                    if (frameID == 17) & (label == np.unique(labels)[-1]):
                        new_objects_manual = [[158.,  1030.], [147.,  np.shape(fullCropped)[0] - 1],  [170., np.shape(fullCropped)[0] - 1]]
                        objectLocations += new_objects_manual
                        assignmentVec += [141, 142,  143]
                        cov = np.cov(np.transpose(np.array(np.where(labelMask > 0)))[:,::-1], rowvar = False).flatten()[[0,1,-1]].tolist()
                        s_x = np.sqrt(cov[0])
                        s_y = np.sqrt(cov[2])
                        rho = cov[1]/(s_x*s_y)
                        frameCov += [[s_x, s_y, rho], [s_x, s_y, rho], [s_x, s_y, rho]]
                        for point in new_objects_manual[1:]:
                            for i in range(frameID):
                                sheepLocations[i] = np.append(sheepLocations[i], [[point[0]+cropX, point[1]+cropY]], axis = 0)
                                sheepCov[i] += [[1.5*s_x, 1.5*s_y, rho]]
                        k = -1



                    if k == 1:
                        objectLocations += [[cX, cY]]
                        assignmentVec += assignSheep([cX, cY], distImg, Ids, centre=[cX, cY])
                        cov = np.cov(np.transpose(np.array(np.where(labelMask > 0)))[:,::-1], rowvar = False).flatten()[[0,1,-1]].tolist()
                        s_x = np.sqrt(cov[0])
                        s_y = np.sqrt(cov[2])
                        rho = cov[1]/(s_x*s_y)
                        frameCov += [[s_x, s_y, rho]]
                    elif k > 1:
                        #pred_objects[:,0] -= cropX
                        #pred_objects[:,1] -= cropY

                        labelImg = np.copy(img)
                        labelImg[labels != label] = 0.

                        miniImage = labelImg[y: y + h, x: x + w]
                        miniGrey = np.copy(grey)[y: y + h, x: x + w]

                        new_objects_k, sCov_k = kmeansClustering(miniImage, miniGrey, numPixels, x, y, previous = k)
                        new_objects_i, sCov_i = iris(miniImage, x, y)

                        num_new_objects_k = np.shape(new_objects_k)[0]
                        num_new_objects_i = np.shape(new_objects_i)[0]

                        if num_new_objects_k == num_new_objects_i:
                            C = cdist(new_objects_i, pred_objects)
                            row_ind, assignment = linear_sum_assignment(C)
                            mean_C_i = (C[row_ind,  assignment].sum())/num_new_objects_i

                            C = cdist(new_objects_k, pred_objects)
                            row_ind, assignment = linear_sum_assignment(C)
                            mean_C_k = (C[row_ind,  assignment].sum())/num_new_objects_k

                            if mean_C_i < mean_C_k:
                                objectLocations += new_objects_i
                                assignmentVec += assignSheep(new_objects_i, distImg, Ids, centre=[cX,  cY])
                                frameCov += sCov_i
                            else:
                                objectLocations += new_objects_k
                                assignmentVec += assignSheep(new_objects_k, distImg, Ids, centre=[cX,  cY])
                                frameCov += sCov_k

                        else:
                            objectLocations += new_objects_k
                            assignmentVec += assignSheep(new_objects_k, distImg, Ids, centre=[cX,  cY])
                            frameCov += sCov_k





        objectLocations = np.array(objectLocations)

        objectLocations[:, 0] += cropX
        objectLocations[:, 1] += cropY
        objectLocations = objectLocations.tolist()

        if frameID < 16:
            N = 141
        elif frameID == 16:
            N += 1
        elif frameID == 17:
            N += 2

        if (frameID > 0) & (frameID <= 6):
            finalDist = cdist(sheepLocations[-1], objectLocations)
            _, assignmentVec = linear_sum_assignment(finalDist)

        finalLocations = organiseLocations(copy.deepcopy(objectLocations), copy.deepcopy(assignmentVec), frameID)
        finalCov = organiseLocations(copy.deepcopy(frameCov), copy.deepcopy(assignmentVec), frameID)

        l = len(finalLocations)
        if plot != 'N':
            plt.close('all')
            plt.imshow(fullCropped)
            plt.scatter(np.array(objectLocations)[:, 0]-cropX, np.array(objectLocations)[:, 1]-cropY, s = 1.)
            plt.gca().set_aspect('equal')
            plt.gca().set_axis_off()
            if plot == 's':
                plt.savefig(save+str(frameID).zfill(4), bbox_inches='tight')
            else:
                plt.pause(15)

        sheepLocations = sheepLocations + [finalLocations]
        sheepVelocity = sheepVelocity + [vel]
        sheepCov = sheepCov + [np.array(finalCov).tolist()]


        print 'frameID: ' + str(frameID)+ ', No. objects located:', l

        if l < N:
            brk = True
            print 'you lost sheep'
            break
        if l > N:
            brk = True
            print 'you gained sheep'
            break


        if np.mod(frameID,50) == 0:
            np.save(save+'loc'+str(frameID), (np.array(sheepLocations), cropVector))
            np.save(save+'vel'+str(frameID), np.array(sheepVelocity))
            np.save(save+'quad'+str(frameID), np.array(quadLocation))
            np.save(save+'cov'+str(frameID), np.array(sheepCov))

        frameID += 1

#cap.release()


np.save(save+'locfull.npy', np.array(sheepLocations))
np.save(save+'velfull.npy', np.array(sheepVelocity))
np.save(save+'quadfull.npy', np.array(quadLocation))
np.save(save+'covfull.npy', np.array(sheepCov))
plt.close('all')

if dell == True:
    if brk == False:
        os.system('notify-send Tracking Complete')
    else:
        os.system('notify-send Tracking Failed')

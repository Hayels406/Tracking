import numpy as np
import cv2
import os
from skimage import measure
import scipy.ndimage as ndimage
from imutils import contours
import imutils as im
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from collections import Counter
import copy

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

if os.getcwd().rfind('Uni') > 0:
    videoLocation = '/home/b1033128/Documents/throughFenceRL.mp4'
    save = '/home/b1033128/Documents/throughFenceRL/'
    dell = True
    brk = False
elif os.getcwd().rfind('hayley') > 0:
    videoLocation = '/users/hayleymoore/Documents/PhD/Tracking/throughFenceRL.mp4'
    save = '/users/hayleymoore/Documents/PhD/Tracking/throughFenceRL/'
else:
    videoLocation = '/data/b1033128/Videos/throughFenceRL.mp4'
    save = '/data/b1033128/Tracking/throughFenceRL/'
    dell = False

plot = 's'
darkTolerance = 173.5
sizeOfObject = 60
restart = 0

sheepLocations = []
sheepVelocity = []
frameID = 0
cropVector = [1000,1000,2000,2028]


cap = cv2.VideoCapture(videoLocation)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print 'You have', length, 'frames', videoLocation

if restart > 0:
    sheepLocations, cropVector = np.load('loc'+str(restart)+'.npy')
    sheepLocations =  map(np.array,  sheepLocations)
    sheepVelocity = np.load('vel'+str(restart)+'.npy')
    sheepVelocity =  map(np.array, sheepVelocity)
    while(frameID <= restart):
        ret, frame = cap.read()
        print frameID
        frameID +=1
    N = len(sheepLocations[0])


while(frameID <= 100):
    plt.close('all')
    ret, frame = cap.read()
    if ret == True:
        assignmentVec = []
        full = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        fullCropped, cropVector = movingCrop(frameID, full, sheepLocations, cropVector)
        cropX, cropY, cropXMax, cropYMax = cropVector
        grey = fullCropped[:,:,2] #extract blue channel

        grey[grey < darkTolerance] = 0.0
        grey[grey > darkTolerance+10.] = 255.

        img = cv2.GaussianBlur(grey,(5,5),2)

        maxfilter = ndimage.maximum_filter(img, size=2)

        if frameID > 6:
            vel = findVel(sheepLocations)
            prediction_Objects = predictEuler(np.array(sheepLocations), vel)
        else:
            vel = []
            prediction_Objects = []

        filtered, minPixels, oneSheepPixels, distImg = createBinaryImage(frameID, sizeOfObject, prediction_Objects, cropVector, maxfilter)

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
                    else:
                        miniImage = img[y: y + h, x: x + w]
                        k = -1

                        new_objects_K = kmeansClustering(miniImage, numPixels, x, y, previous = k)
                        new_objects_i   = iris(miniImage, x, y)

                        num_new_objects_K = np.shape(new_objects_K)[0]
                        num_new_objects_i = np.shape(new_objects_i)[0]

                        if num_new_objects_i == 1:
                            check = 'Off'
                            objectLocations += new_objects_K
                        
                        elif num_new_objects_K == 1:
                            check = 'Off'
                            objectLocations += new_objects_i

                        elif num_new_objects_K == num_new_objects_i:
                            C = cdist(new_objects_i, new_objects_K)
                            row_ind, assignment = linear_sum_assignment(C)
                            av_dist = C[row_ind, assignment].sum()/num_new_objects_i        

                            if av_dist < 3.5:
                                check = 'Off'
                                objectLocations += new_objects_K

                        if check == 'On':
                            objectLocations = doCheck(fullCropped, objectLocations, cX, cY, img, new_objects_i, new_objects_K, rectangle, k)

                elif frameID <= 6:
                    if numPixels < oneSheepPixels:
                        objectLocations += [[cX, cY]]
                    else:
                        miniImage = img[y: y + h, x: x + w]

                        lastT = np.array(sheepLocations[-1])
                        padding = 3
                        Ids = getPreviousID(lastT, x, y, w, h, cropX, cropY, padding)
                        k = len(Ids)

                        new_objects_K = kmeansClustering(miniImage, numPixels, x, y, previous = k)
                        new_objects_i   = iris(miniImage, x, y)

                        num_new_objects_K = np.shape(new_objects_K)[0]
                        num_new_objects_i = np.shape(new_objects_i)[0]

                        if num_new_objects_K == num_new_objects_i:
                            C = cdist(new_objects_i, new_objects_K)
                            row_ind, assignment = linear_sum_assignment(C)
                            av_dist = C[row_ind, assignment].sum()/num_new_objects_i        

                            if av_dist < 3.5:
                                check = 'Off'
                                objectLocations += new_objects_K
                                



                        else:
                            check = 'Off' 
                            objectLocations += new_objects_K

                        if check == 'On':
                            objectLocations = doCheck(fullCropped, objectLocations, cX, cY, img, new_objects_i, new_objects_K, rectangle, k)


                else:
                    pred_objects, Ids = getPredictedID(prediction_Objects, np.copy(labelMask),  cropVector)
                    k = len(pred_objects)
                    check = 'Off'

                    if (frameID == 16) & (label == np.unique(labels)[-2]):
                        objectLocations += [[cX, cY]]
                        assignmentVec += [141]
                        for i in range(frameID):
                            sheepLocations[i] = np.append(sheepLocations[i], [[cX+cropX, cY+cropY]], axis = 0)
                        k = -1

                    if (frameID == 17) & (label == np.unique(labels)[-1]):
                        new_objects_manual = [[158.,  1030.], [147.,  np.shape(fullCropped)[0] - 1],  [170., np.shape(fullCropped)[0] - 1]]
                        objectLocations += new_objects_manual
                        assignmentVec += [141, 142,  143]
                        for point in new_objects_manual[1:]:
                            for i in range(frameID):
                                sheepLocations[i] = np.append(sheepLocations[i], [[point[0]+cropX, point[1]+cropY]], axis = 0)
                        k = -1



                    if k == 1:
                        objectLocations += [[cX, cY]]
                        assignmentVec += assignSheep([cX, cY], distImg, Ids, centre=[cX, cY])
                    elif k > 1:
                        pred_objects[:,0] -= cropX
                        pred_objects[:,1] -= cropY

                        labelImg = np.copy(img)
                        labelImg[labels != label] = 0.

                        miniImage = labelImg[y: y + h, x: x + w]

                        new_objects_K = kmeansClustering(miniImage, numPixels, x, y, previous = k)
                        new_objects_i   = iris(miniImage, x, y)

                        num_new_objects_K = np.shape(new_objects_K)[0]
                        num_new_objects_i = np.shape(new_objects_i)[0]

                        if num_new_objects_K == num_new_objects_i:
                            C = cdist(new_objects_i, pred_objects)
                            row_ind, assignment = linear_sum_assignment(C)
                            mean_C_i = (C[row_ind,  assignment].sum())/num_new_objects_i

                            C = cdist(new_objects_K, pred_objects)
                            row_ind, assignment = linear_sum_assignment(C)
                            mean_C_k = (C[row_ind,  assignment].sum())/num_new_objects_K

                            if mean_C_i < mean_C_k:
                                objectLocations += new_objects_i
                                assignmentVec += assignSheep(new_objects_i, distImg, Ids, centre=[cX,  cY])
                            else:
                                objectLocations += new_objects_K
                                assignmentVec += assignSheep(new_objects_K, distImg, Ids, centre=[cX,  cY])

                        else:
                            objectLocations += new_objects_K
                            assignmentVec += assignSheep(new_objects_K, distImg, Ids, centre=[cX,  cY])




            
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

        l = len(finalLocations)
        if plot != 'N':
            plt.close()
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
            np.save('loc'+str(frameID), (np.array(sheepLocations), cropVector))
            np.save('vel'+str(frameID), np.array(sheepVelocity))

        frameID += 1

cap.release()


np.save('locfull.npy', np.array(sheepLocations))
np.save('velfull.npy', np.array(sheepVelocity))
plt.close()

if dell == True:
    if brk == False:
        os.system('notify-send Tracking Complete')
    else:
        os.system('notify-send Tracking Failed')
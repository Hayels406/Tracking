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

from trackingFunctions import kmeansClustering
from trackingFunctions import iris
from trackingFunctions import predictEuler
from trackingFunctions import movingCrop
from trackingFunctions import createBinaryImage
from trackingFunctions import findVel

if os.getcwd().rfind('b1033128') > 0:
    videoLocation = '/home/b1033128/Documents/throughFenceRL.mp4'
    save = '/home/b1033128/Documents/throughFenceRL'
elif os.getcwd().rfind('hayley') > 0:
    videoLocation = '/users/hayleymoore/Documents/PhD/Tracking/throughFenceRL.mp4'
    save = '/users/hayleymoore/Documents/PhD/Tracking/throughFenceRL'
plot = 's'
darkTolerance = 173.5
sizeOfObject = 60
radiIN = 5.
restart = 0

sheepLocations = []
frameID = 0
cropVector = [0,0,0,0]


cap = cv2.VideoCapture(videoLocation)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print 'You have', length, 'frames'

if restart > 0:
    sheepLocations, cropVector = np.load('loc'+str(restart)+'.npy')

    while(frameID <= restart):
        ret, frame = cap.read()
        print frameID
        frameID +=1


while(frameID <= 60):
    plt.close('all')
    ret, frame = cap.read()
    if ret == True:

        full = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        fullCropped, cropVector = movingCrop(frameID, full, sheepLocations, cropVector)
        cropX, cropY, cropXMax, cropYMax = cropVector
        grey = fullCropped[:,:,2] #extract blue channel

        grey[grey < darkTolerance] = 0.0
        grey[grey > darkTolerance+10.] = 255.

        img = cv2.GaussianBlur(grey,(5,5),2)

        maxfilter = ndimage.maximum_filter(img, size=2)

        filtered, minPixels, oneSheepPixels, radi, prediction_Objects = createBinaryImage(frameID, sizeOfObject, radiIN, sheepLocations, cropVector, maxfilter)

        plt.imshow(filtered, cmap = 'gray')
        if frameID > 6:
            plt.scatter(np.array(prediction_Objects)[:,0]-cropX, np.array(prediction_Objects)[:,1]-cropY, s =1.)
        plt.gca().set_aspect('equal')
        plt.gca().set_axis_off()
        if plot == 's':
            plt.savefig(save+'/filtered/'+str(frameID).zfill(4), bbox_inches='tight')

        labels = measure.label(filtered, neighbors=8, background=0)

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

            numPixels = (labelMask > 0).sum()
            # if the number of pixels in the component is sufficiently
            # large, then add it to our mask of "large blobs"

            if numPixels > minPixels:
                cnts = cv2.findContours(labelMask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1][0]
                ((cX, cY), radius) = cv2.minEnclosingCircle(cnts)
                if (radius > radi) & (radius < 200):
                    if numPixels < oneSheepPixels:
                        objectLocations = objectLocations + [[cX, cY]]
                    else:
                        x,y,w,h = cv2.boundingRect(cnts)

                        miniImage = img[y: y + h, x: x + w]
                        if frameID > 6:
                            lastT = np.array(sheepLocations[-1])
                            prevID = np.where((lastT[:,0] > x+cropX) & (lastT[:,0] < x+w+cropX) & (lastT[:,1] < y+h+cropY) & (lastT[:,1] > y+cropY))[0]
                            vel = findVel(np.array(sheepLocations[-1]), np.array(sheepLocations[-6]))[prevID]
                            velMean = np.mean(np.sqrt((vel**2).sum(axis = 1)))
                            leeway = 3*velMean
                            prevID = np.where((lastT[:,0] > x+cropX-leeway) & (lastT[:,0] < x+w+cropX+leeway) & (lastT[:,1] < y+h+cropY+leeway) & (lastT[:,1] > y+cropY-leeway))[0]
                            prev =  len(prevID)

                            pred_objects =  prediction_Objects[prevID]
                            pred_objects[:,0] -= cropX
                            pred_objects[:,1] -= cropY
                        elif frameID > 0:
                            lastT = np.array(sheepLocations[-1])
                            leeway = 3
                            prevID = np.where((lastT[:,0] > x+cropX-leeway) & (lastT[:,0] < x+w+cropX+leeway) & (lastT[:,1] < y+h+cropY+leeway) & (lastT[:,1] > y+cropY-leeway))[0]
                            prev =  len(prevID)
                        else:
                            prev = -1
                            leeway = 0

                        new_objects_K = kmeansClustering(miniImage, numPixels, x, y, previous = prev)
                        new_objects = iris(miniImage, x, y)

                        num_new_objects_i = np.shape(new_objects)[0]
                        num_new_objects_k = np.shape(new_objects_K)[0]

                        if num_new_objects_i == 1:
                            check = 'Off'
                            objectLocations += new_objects_K
                        elif num_new_objects_k == 1:
                            check = 'Off'
                            objectLocations += new_objects
                        elif (frameID > 0) & (num_new_objects_i != prev):
                            check = 'Off'
                            objectLocations += new_objects_K
                        elif num_new_objects_i == num_new_objects_k:
                            C = cdist(new_objects, new_objects_K)
                            row_ind, assignment = linear_sum_assignment(C)
                            av_dist = C[row_ind, assignment].sum()/num_new_objects_i

                            if av_dist < 3.5:
                                check = 'Off'
                                objectLocations = objectLocations + new_objects_K
                            elif frameID > 6:
                                C = cdist(new_objects, pred_objects)
                                row_ind, assignment = linear_sum_assignment(C)
                                mean_C_i = (C[row_ind,  assignment].sum())/num_new_objects_i
                                C = cdist(new_objects_K, pred_objects)
                                row_ind, assigment = linear_sum_assignment(C)
                                mean_C_k = (C[row_ind,  assignment].sum())/num_new_objects_k

                                check = 'Off'
                                if mean_C_i < mean_C_k:
                                    objectLocations = objectLocations + new_objects
                                else:
                                    objectLocations = objectLocations + new_objects_K




                        if check == 'On':
                            plt.close()
                            #plt.figure(dpi = 300)
                            plt.subplot(2, 2, 2)
                            plt.imshow(fullCropped)
                            plt.scatter(cX, cY, color = 'k')

                            plt.subplot(2, 2, 1)
                            plt.imshow(fullCropped)
                            plt.ylim(ymin=y+h-1, ymax=y)
                            plt.xlim(xmin=x, xmax=x+w-1)

                            plt.subplot(2, 2, 3)
                            plt.imshow(img)
                            plt.ylim(ymin=y+h-1+leeway, ymax=y-leeway)
                            plt.xlim(xmin=x-leeway, xmax=x+w-1+leeway)
                            if frameID > 6:
                                plt.scatter(lastT[prevID][:,0]-cropX, lastT[prevID][:,1]-cropY, color = 'k', alpha = 0.5, label = 'previous')
                                plt.scatter(prediction_Objects[prevID][:,0]-cropX, prediction_Objects[prevID][:,1]-cropY, color = 'b', marker = 's', label = 'prediction')
                            plt.scatter(np.array(new_objects_K)[:,0], np.array(new_objects_K)[:,1], color = 'green', marker = '^', label = 'kmeans')
                            plt.scatter(np.array(new_objects)[:,0], np.array(new_objects)[:,1], color = 'red', label = 'iris', alpha = 0.5)
                            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

                            plt.pause(0.00001)


                            if prev < 0:
                                text = raw_input("Choose method for this mini image: ")
                            else:
                                text = raw_input("Choose method for this mini image ("+str(prev)+"): ")

                            if text == '1' or text == 'centre':
                                objectLocations = objectLocations + [[cX+cropX, cY+cropY]]
                            elif text == '2' or text == 'kmeans':
                                objectLocations = objectLocations + new_objects_K
                            elif text == '3' or text == 'iris':
                                objectLocations = objectLocations + new_objects
                            elif text == '0':
                                objectLocations = objectLocations
                            elif text == 'd':
                                print 'kmeans: ',  new_objects_K
                                print 'iris: ',  new_objects
                                print 'previous: ',  prev
                                print 'dist: ',  sum_C/num_new_objects_i
                                objectLocations = objectLocations + new_objects_K
                            else:
                                print 'you gave an awful answer: used kmeans'
                                objectLocations = objectLocations + new_objects_K

                            plt.clf()


        objectLocations = np.array(objectLocations)
        objectLocations[:, 0] += cropX
        objectLocations[:, 1] += cropY

        if plot != 'N':
            plt.close()
            plt.imshow(fullCropped)
            plt.scatter(objectLocations[:, 0] - cropX, objectLocations[:, 1] - cropY, s = 1.)
            plt.gca().set_aspect('equal')
            plt.gca().set_axis_off()
            if plot == 's':
                plt.savefig(save+str(frameID).zfill(4), bbox_inches='tight')
            else:
                plt.pause(15)

        objectLocations = objectLocations.tolist()

        if frameID < 32:
            N = 141
        elif frameID == 32:
            prevID =  np.where(np.array(objectLocations)[:,1] == np.max(np.array(objectLocations)[:,1]))[0][0]
            saveLocation = objectLocations[prevID]
            for i in range(frameID):
                sheepLocations[i] += [saveLocation]
            N += 1
        elif frameID == 34:
            prevID =  np.where(np.array(objectLocations)[:,1] == np.max(np.array(objectLocations)[:,1]))[0][0]
            saveLocation = objectLocations[prevID]
            for i in range(frameID):
                sheepLocations[i] += [saveLocation]
            N += 1
        elif frameID == 39:
            prevID =  np.where(np.array(objectLocations)[:,1] == np.max(np.array(objectLocations)[:,1]))[0][0]
            saveLocation = objectLocations[prevID]
            for i in range(frameID):
                sheepLocations[i] += [saveLocation]
            N += 1

        l = len(objectLocations)
        while (l > N) & (frameID > 0):
            C = cdist(sheepLocations[-1],  objectLocations)
            r = set(range(N))
            _, assignment = linear_sum_assignment(C)
            extras = list(r-set(assignment))
            if len(extras) > 0:
                extras.sort()
                extras.reverse()
                print extras, N, len(objectLocations)
                for ex in extras:
                    objectLocations.pop(ex)
            l = len(objectLocations)

        sheepLocations = sheepLocations + [objectLocations]


        print 'frameID: ' + str(frameID)+ ', No. objects located:', l

        if l < N:
            print 'you lost sheep'
            break
        if l > N:
            print 'you gained sheep'
            break

        frameID += 1

        if np.mod(frameID,50) == 0:
            np.save('loc'+str(frameID), (np.array(sheepLocations), cropVector))

cap.release()
plt.clf()


np.save('locfull.npy', np.array(sheepLocations))

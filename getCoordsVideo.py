import numpy as np
import cv2
from skimage import measure
import scipy.ndimage as ndimage
from imutils import contours
import imutils as im
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score


def track(videoLocation, plot, darkTolerance, sizeOfObject, radi, test = False, lowerBoundY = 0, upperBoundY = 2500, lowerBoundX = 0, upperBoundX = 3000):
    test_xmin = 600
    test_xmax = 800
    test_ymin = 1400
    test_ymax = 1200

    test2 = 'O'


    sheepLocations = []
    frameID = 0

    cap = cv2.VideoCapture(videoLocation)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print length
    while(frameID < 700):
        ret, frame = cap.read()
        print frameID
        frameID +=1
    while(frameID <= 700):
        ret, frame = cap.read()
        if ret == True:



            full = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            if test == True:
                plt.clf()
                plt.imshow(full)
                plt.xlim(xmin = test_xmin, xmax = test_xmax)
                plt.ylim(ymax = test_ymax, ymin = test_ymin)
                id = 1
                plt.savefig('./test'+str(id)+'.png')

            grey = full[:,:,2]#cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

            grey[grey < darkTolerance] = 0.0
            grey[grey > 170.] = 255.
            if test == True:
                plt.imshow(grey)
                plt.xlim(xmin = test_xmin, xmax = test_xmax)
                plt.ylim(ymax = test_ymax, ymin = test_ymin)
                id += 1
                plt.savefig('./test'+str(id)+'.png')

            img = ndimage.gaussian_filter(grey, sigma = (2), order = 0)
            if test == True:
                plt.imshow(img)
                plt.xlim(xmin = test_xmin, xmax = test_xmax)
                plt.ylim(ymax = test_ymax, ymin = test_ymin)
                id += 1
                plt.savefig('./test'+str(id)+'.png')

            maxfilter = ndimage.maximum_filter(img, size=2)
            if test == True:
                plt.imshow(maxfilter)
                plt.xlim(xmin = test_xmin, xmax = test_xmax)
                plt.ylim(ymax = test_ymax, ymin = test_ymin)
                id += 1
                plt.savefig('./test'+str(id)+'.png')

            filtered = maxfilter
            filtered[filtered < 65.] = 0.0 #for removing extra shiney grass
            filtered[filtered > 0.] = 255.
            if test == True:
                plt.imshow(filtered)
                plt.xlim(xmin = test_xmin, xmax = test_xmax)
                plt.ylim(ymax = test_ymax, ymin = test_ymin)
                id += 1
                plt.savefig('./test'+str(id)+'.png')

            labels = measure.label(filtered, neighbors=8, background=0)

            objectLocations = []
            r = []
            pix = []
            # loop over the unique components

            for label in np.unique(labels):
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

                if numPixels > sizeOfObject:
                    cnts = cv2.findContours(labelMask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1][0]
                    ((cX, cY), radius) = cv2.minEnclosingCircle(cnts)
                    if cY < lowerBoundY:
                        radius = 0
                    if cX < lowerBoundX:
                        radius = 0
                    if cY > upperBoundY:
                        radius = 0
                    if cX > upperBoundX:
                        radius = 0

                    if radius > radi:
                        if radius < 200:
                            pix = pix + [numPixels]
                            r = r + [radius]
                            #objectLocations = objectLocations + [[cX, cY]]
                            if numPixels < 250:
                                objectLocations = objectLocations + [[cX, cY]]
                            else:
                                approxNumber = np.ceil((numPixels - 50.)/200.)
                                clusters = np.array([approxNumber - 2, approxNumber - 1, approxNumber, approxNumber + 1, approxNumber + 2])
                                av_score = []
                                thresh = 205
                                x,y,w,h = cv2.boundingRect(cnts)

                                maxfilter = ndimage.maximum_filter(img, size=2)
                                maxfilter[maxfilter < thresh] = 0

                                miniImg = maxfilter[y: y + h, x: x + w]
                                for n_clusters in np.array(clusters[clusters > 1]):
                                    n_clusters = int(n_clusters)
                                    if np.shape(np.transpose(np.where(miniImg > thresh)))[0] - 1 < n_clusters:
                                        av_score += [0]
                                        continue
                                    clusterer = KMeans(n_clusters=n_clusters, random_state=10)

                                    cluster_labels = clusterer.fit_predict(np.transpose(np.where(miniImg > thresh)))

                                    av_score += [silhouette_score(np.transpose(np.where(miniImg > thresh)), cluster_labels)]

                                approxNumber =  np.where(av_score == np.max(av_score))[0][0] + 2
                                if approxNumber < np.shape(np.transpose(np.where(miniImg > thresh)))[0] - 1:
                                    clusterer = KMeans(n_clusters=approxNumber, random_state=10).fit(np.transpose(np.where(miniImg > thresh)))
                                    cluster_centers = clusterer.cluster_centers_

                                    cluster_list = np.copy(cluster_centers)
                                    cluster_list[:,0] = cluster_centers[:,1] + x
                                    cluster_list[:,1] = cluster_centers[:,0] + y

                                    new_objects =  cluster_list.tolist()
                                else:
                                    new_objects = [[cX, cY]]

                                objectLocations = objectLocations + new_objects
                                if test2 == 'O':
                                    plt.subplot(1, 2, 1)
                                    plt.imshow(full)
                                    plt.ylim(ymin=y+h-1, ymax=y)
                                    plt.xlim(xmin=x, xmax=x+w-1)

                                    plt.subplot(1,2,2)
                                    plt.imshow(img)
                                    plt.ylim(ymin=y+h-1, ymax=y)
                                    plt.xlim(xmin=x, xmax=x+w-1)
                                    plt.colorbar()
                                    plt.scatter(np.array(new_objects)[:,0], np.array(new_objects)[:,1], color = 'red')
                                    plt.scatter(cX, cY, color = 'k')
                                    plt.pause(3)
                                    plt.clf()



            sheepLocations = sheepLocations + [objectLocations]
            print frameID, len(objectLocations)
            if plot != 'N':
                plt.cla()
                plt.imshow(full)
                plt.scatter(np.array(objectLocations)[:, 0], np.array(objectLocations)[:, 1], s = 6.)
                if test == True:
                    plt.xlim(xmin = test_xmin, xmax = test_xmax)
                    plt.ylim(ymax = test_ymax, ymin = test_ymin)
                    id += 1
                else:
                    plt.xlim(xmin = 0, xmax = 2500)
                    plt.ylim(ymax = 0, ymin = 2000)
                #plt.axis('equal')
                if plot == 's':
                    plt.savefig('/Users/hayleymoore/Documents/PhD/Tracking/throughFenceRL/'+str(frameID).zfill(4))
                else:
                    plt.pause(15)

            frameID += 1

    cap.release()
    plt.clf()
    return [np.array(sheepLocations), r, pix]

locations, radii, pixels = track('/Users/hayleymoore/Documents/PhD/Tracking/throughFenceRL.mp4', plot = 'Y', test = False, darkTolerance = 173.5, sizeOfObject = 60, radi = 5., upperBoundX = 2000, lowerBoundY = 500, lowerBoundX = 100)
np.save('locfull.npy',locations)

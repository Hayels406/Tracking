import numpy as np
import cv2
from skimage import measure
import scipy.ndimage as ndimage
from imutils import contours
import imutils as im
import matplotlib.pyplot as plt


def track(videoLocation, plot, darkTolerance, sizeOfObject, radi, lowerBoundY = 0, upperBoundY = 2500, lowerBoundX = 0, upperBoundX = 3000):
    sheepLocations = []
    frameID = 0

    cap = cv2.VideoCapture(videoLocation)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print length
    while(frameID <= 0):
        ret, frame = cap.read()
        if ret == True:

            print frameID

            full = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            #plt.imshow(full)
            #plt.pause(5)

            grey = full[:,:,2]#cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

            grey[grey < darkTolerance] = 0.0
            grey[grey > 170.] = 255.
            #plt.imshow(grey)
            #plt.pause(5)

            img = ndimage.gaussian_filter(grey, sigma = (2), order = 0)
            #plt.imshow(img)
            #plt.pause(5)

            filtered = ndimage.maximum_filter(img, size=2)

            filtered[filtered < 65.] = 0.0 #for removing extra shiney grass
            filtered[filtered > 0.] = 255.

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
                            elif numPixels < 450:
                                ymin = np.min(np.where(labelMask[:,1] == 1))
                                ymax = np.max(np.where(labelMask[:,1] == 1))
                                print ymin, ymax
                                objectLocations = objectLocations + [[cX-radius/2., cY-radius/2.], [cX+radius/2., cY+radius/2.]]
                            else:
                                objectLocations = objectLocations + [[cX-radius/2., cY-radius/2.], [cX+radius/2., cY+radius/2.], [cX-radius/2, cY+radius/2]]


            print np.shape(objectLocations)[0]
            sheepLocations = sheepLocations + [objectLocations]

            if plot != 0:
                plt.cla()
                plt.imshow(filtered)
                plt.scatter(np.array(objectLocations)[:, 0], np.array(objectLocations)[:, 1], s = 6.)
                plt.xlim(xmin = 1000, xmax = 2000)
                plt.ylim(ymax = 1000)
                plt.pause(15)

            frameID += 1

    cap.release()
    plt.clf()
    return [np.array(sheepLocations), r, pix]

locations, radii, pixels = track('/Users/hayleymoore/Documents/PhD/Tracking/throughFenceRL.mp4', plot = 1, darkTolerance = 172.5, sizeOfObject = 60, radi = 5., upperBoundX = 2000, lowerBoundY = 900, lowerBoundX = 100)
print locations

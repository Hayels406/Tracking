import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from skimage import measure
from imutils import contours
import scipy.ndimage as ndimage
import imutils

sheepImg = mpimg.imread('~/Desktop/Screenshot.png')
rgb = cv2.cvtColor(sheepImg,cv2.COLOR_RGBA2RGB)
plt.imshow(rgb)
plt.title('Original')
plt.show()


rgb[rgb.sum(axis = 2) < 3*190./255.] = 0.0 #edit 190 for removing dark areas
grey = cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
plt.imshow(grey)
plt.show()

img = ndimage.gaussian_filter(grey, sigma=(5), order=0)
plt.imshow(img)
plt.show()
largeObject = grey
largeObject[img < 40./255.] = 0.0 #edit 40 for removing extra shiney grass
largeObject[largeObject < 40./255.] = 0.0 #edit 40 for removing extra shiney grass
largeObject[largeObject > 0./255.] = 1.0


labels = measure.label(largeObject, neighbors=8, background=0)
mask = np.zeros(largeObject.shape, dtype="uint8")

# loop over the unique components
for label in np.unique(labels):
    # if this is the background label, ignore it
    if label == 0:
        continue

    # otherwise, construct the label mask and count the
    # number of pixels
    labelMask = np.zeros(largeObject.shape, dtype="uint8")
    labelMask[labels == label] = 255
    numPixels = cv2.countNonZero(labelMask)

    # if the number of pixels in the component is sufficiently
    # large, then add it to our mask of "large blobs"
    if numPixels > 1500: #edit 1500 for removing extra shiney grass
        mask = cv2.add(mask, labelMask)

cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
cnts = contours.sort_contours(cnts)[0]

# loop over the contours
objectLocations = []
for (i, c) in enumerate(cnts):
    # draw the bright spot on the image
    ((cX, cY), radius) = cv2.minEnclosingCircle(c)
    objectLocations = objectLocations + [[cX, cY]]

objectLocations = np.array(objectLocations)
sheepLocations = objectLocations[np.where(np.diff(objectLocations[:,0]) > 500)[0][-1] + 1:, :] #edit 500 for removing extra white areas

print np.shape(sheepLocations)[0]
plt.imshow(sheepImg)
plt.scatter(sheepLocations[:,0], sheepLocations[:,1], s = 6.)
plt.show()

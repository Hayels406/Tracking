import numpy as np
import cv2
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

def organiseLocations(objects, assign, frameId):
    if (frameId <= 6) & (frameId > 0):
        objects = np.array(objects)[assign]
    elif frameId > 6:
        count = Counter(assign)
        multiples = np.where(np.array(count.values()) > 1)[0]
        multiples = multiples.tolist()
        multiples.sort()
        multiples.reverse()
        for v in multiples:
            ind = np.where(np.array(assign) == v)[0]
            points = np.array(objects)[ind]
            newPoint = points.mean(axis = 0)
            for i in ind[::-1]:
                objects.pop(i)
                assign.pop(i)
            objects += [newPoint.tolist()]
            assign += [v]
        objects = np.array(objects)[np.argsort(assign)]
    return objects

def assignSheep(coords, dImg, prevId):
    if type(coords[0]) != float:
        den2 = []
        for predictionArea in np.array(dImg)[prevId]:
            den = []
            for point in np.floor(coords):
                den += [np.transpose(predictionArea)[int(point[0]), int(point[1])]]
            den2 += [(np.max(den) - den)/np.max(den)]
        _, assignment = linear_sum_assignment(den2)
        return prevId[assignment].tolist()
    else:
        return prevId.tolist()



def extractFromSlice(sheepBlob):
    sheepBlob = sheepBlob/np.max(sheepBlob) 
    blob = np.where(sheepBlob < 0.2, 0, 1)
    cnts = cv2.findContours(np.uint8(blob), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1][0]
    return cv2.minEnclosingCircle(cnts)[0]

def movingaverage(interval, window_size):
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

def combinePrediction(maxF, predIm, weight, plot = False):
    ind1 = 1 - weight
    ind2 = weight
    combined = maxF**ind1*predIm**ind2
    if plot == True:
        plt.subplot(1,3,1)
        plt.imshow(maxF,  cmap = 'magma')
        plt.title('Max Filter')

        plt.subplot(1,3,2)
        plt.imshow(predIm,  cmap = 'magma')
        plt.title('Prediction')

        plt.subplot(1,3,3)
        plt.imshow(combined, cmap =  'magma')
        plt.title('Combination')

        plt.show()

    return combined

def alpha(theta, m, atan):
    xr = int(round(m*np.sin(theta)))
    yr = int(round(m*np.cos(theta)))
    return np.roll(atan, (xr,yr), axis=(0,1))

def CI(i, N, m, atan):
	theta = 2*np.pi*(i-1)/N
	return np.cos(theta - alpha(theta, m, atan))

def convsum(atan, r, i, N):
	sumimg = atan*0.0
	for m in range(1,r+1):
		sumimg = sumimg+CI(i, N, m, atan)
	return sumimg/r

def iris(miniImg, X, Y):
    threshold = 0.4*np.max(miniImg)
    N = 8
    Lx  = cv2.Sobel(miniImg,cv2.CV_64F,1,0,ksize=5)
    Ly  = cv2.Sobel(miniImg,cv2.CV_64F,0,1,ksize=5)

    atanLxLy = np.arctan2(Ly,Lx)
    rsum = atanLxLy*0.0

    for i in range(N):
        rMax = atanLxLy*0.0
        for c in map(lambda r: convsum(atanLxLy, r, i, N), range(1,25)):
    	       rMax = np.maximum(rMax, c)
        rsum = rsum + rMax

    iris = rsum/N

    iris = iris - np.median(iris)
    iris[iris < 0] = 0
    iris = np.uint8(iris*255./np.max(iris))

    threshold= 0.3*np.max(iris)
    ret, thresh = cv2.threshold(iris,threshold,255,cv2.THRESH_BINARY)

    output = cv2.connectedComponentsWithStats(thresh, 8, cv2.CV_32S)
    centroids = output[3]
    centroids = np.transpose(centroids)

    new_objects = np.copy(centroids[:,1:])
    new_objects[0,:] += X
    new_objects[1,:] += Y
    new_objects = np.transpose(new_objects).tolist()#remove element 0 cost its the background!
    final_objects = []
    for i in range(np.shape(new_objects)[0]):
        if new_objects[i][0] < 3 + X:
            continue
        elif new_objects[i][0] > np.shape(miniImg)[1] - 3 + X:
            continue
        elif new_objects[i][1] < 3 + Y:
            continue
        elif new_objects[i][1] > np.shape(miniImg)[0] - 3 + Y:
            continue
        else:
            final_objects += [np.array(new_objects)[i].tolist()]

    return final_objects

def kmeansClustering(miniImg, numberPixels, X, Y, previous):
    threshold= 0.85*np.max(miniImg)
    if previous == 0:
        plt.imshow(miniImg)
        plt.axes().set_aspect('auto')
        plt.title('KMeans error')
        plt.show()
    if previous <  0:
        approxNumber = np.ceil((numberPixels - 50.)/200.)
        clusters = np.array([approxNumber - 2, approxNumber - 1, approxNumber, approxNumber + 1, approxNumber + 2])
        av_score = []

        for n_clusters in np.array(clusters[clusters > 1]):
            n_clusters = int(n_clusters)
            if np.shape(np.transpose(np.where(miniImg > threshold)))[0] - 1 < n_clusters:
                av_score += [0]
                continue
            clusterer = KMeans(n_clusters=n_clusters,random_state = 10)

            cluster_labels = clusterer.fit_predict(np.transpose(np.where(miniImg > threshold)))

            av_score += [silhouette_score(np.transpose(np.where(miniImg > threshold)), cluster_labels)]

        approxNumber =  np.where(av_score == np.max(av_score))[0][0] + 2
        if approxNumber < np.shape(np.transpose(np.where(miniImg > threshold)))[0] - 1:
            clusterer = KMeans(n_clusters=approxNumber,random_state = 10).fit(np.transpose(np.where(miniImg > threshold)))
            cluster_centers = clusterer.cluster_centers_

            cluster_list = np.copy(cluster_centers)
            cluster_list[:,0] = cluster_centers[:,1] + X
            cluster_list[:,1] = cluster_centers[:,0] + Y

            new_objects =  cluster_list.tolist()
        else:
            new_objects = [[cX, cY]]

    else:
        clusterer = KMeans(n_clusters=previous,random_state = 10)
        clusterer = clusterer.fit(np.transpose(np.where(miniImg > threshold)))
        cluster_centers = clusterer.cluster_centers_
        cluster_list = np.copy(cluster_centers)
        cluster_list[:,0] = cluster_centers[:,1] + X
        cluster_list[:,1] = cluster_centers[:,0] + Y


        new_objects =  cluster_list.tolist()

    return new_objects

def findVel(locs):
    vel = (locs[-1] - locs[-6])/5
    return vel

def predictEuler(locs):
    vel = findVel(locs)
    prediction_Objects = locs[-1] + vel
    return prediction_Objects

def doCheck(fullC, objL, cx, cy, Img, new_i, new_k, X, Y, H, W, K, margin=0):
    plt.close()
    plt.subplot(2, 2, 2)
    plt.imshow(fullC)
    plt.scatter(cx, cy, color = 'k')

    plt.subplot(2, 2, 1)
    plt.imshow(fullC)
    plt.ylim(ymin=Y+H-1, ymax=Y)
    plt.xlim(xmin=X, xmax=X+W-1)

    plt.subplot(2, 2, 3)
    plt.imshow(Img)
    plt.ylim(ymin=Y+H-1+margin, ymax=Y-margin)
    plt.xlim(xmin=X-margin, xmax=X+W-1+margin)
    plt.scatter(np.array(new_k)[:,0], np.array(new_k)[:,1], color = 'green', marker = '^', label = 'kmeans')
    plt.scatter(np.array(new_i)[:,0], np.array(new_i)[:,1], color = 'red', label = 'iris', alpha = 0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.pause(0.00001)


    if K < 0:
        text = raw_input("Choose method for this mini image: ")
    else:
        text = raw_input("Choose method for this mini image ("+str(K)+"): ")

    if text == '1' or text == 'centre':
        objL = objL + [[cx+cropX, cy+cropY]]
    elif text == '2' or text == 'kmeans':
        objL = objL + new_k
    elif text == '3' or text == 'iris':
        objL = objL + new_i
    elif text == '0':
        objL = objL
    elif text == 'd':
        print 'kmeans: ',  new_k
        print 'iris: ',  new_i
        print 'previous: ',  prev
        print 'dist: ',  sum_C/num_new_i_i
        objL = objL + new_k
    else:
        print 'you gave an awful answer: used kmeans'
        objL = objL + new_k

    plt.close()
    return objL

def getPreviousID(prev, X, Y, W, H, cropx, cropy, margin):
    return np.where((prev[:,0] > X+cropx-margin) & (prev[:,0] < X+W+cropx+margin) & (prev[:,1] < Y+H+cropy+margin) & (prev[:,1] > Y+cropy-margin))[0]

def getPredictedID(pred, X, Y, W, H, cropx, cropy):
    return np.where((pred[:,0] > X+cropx-3) & (pred[:,0] < X+W+cropx+3) & (pred[:,1] < Y+H+cropy+3) & (pred[:,1] > Y+cropy-3))[0]

def movingCrop(frameID, full, sheepLoc, cropVector):
    cropX, cropY, cropXMax, cropYMax = cropVector

    if frameID < 2:
        cropVector = cropVector
    elif frameID < 50:
        moveX, moveY = np.min(sheepLoc[-2], axis = 0) - np.min(sheepLoc[-1], axis = 0)
        cropX = int(np.floor(cropX + moveX))
        cropY = int(np.floor(cropY + moveY))
        moveX, moveY = np.max(sheepLoc[-2], axis = 0) - np.max(sheepLoc[-1], axis = 0)
        cropXMax = int(np.floor(cropXMax - moveX))
        cropYMax = 2028
    else:
        moveX, moveY = np.min(sheepLoc[-2], axis = 0) - np.min(sheepLoc[-1], axis = 0)
        cropX = int(np.floor(cropX - moveX))
        cropY = int(np.floor(cropY - moveY))
        moveX, moveY = np.max(sheepLoc[-2], axis = 0) - np.max(sheepLoc[-1], axis = 0)
        cropXMax = int(np.floor(cropXMax - moveX))
        cropYMax = int(np.floor(cropYMax - moveY))
    
    fullCropped = np.copy(full)[cropY:cropYMax, cropX:cropXMax, :]
    cropVector = [cropX, cropY, cropXMax, cropYMax]
    return (fullCropped, cropVector)

def createBinaryImage(frameID, sizeOfObject, pred_Objects, cropVector, maxF):
    cropX, cropY, cropXMax, cropYMax = cropVector
    if frameID <= 6:
        minPixels = sizeOfObject
        oneSheepPixels = 250
        filtered = np.copy(maxF)
        filtered[filtered < 65.] = 0.0 #for removing extra shiney grass
        filtered[filtered > 0.] = 255.
        z = []

    elif frameID > 6:
        minPixels = 0*sizeOfObject
        oneSheepPixels = 0.7*250
        
        x_r =  np.arange(cropX,  cropXMax)
        y_r =  np.arange(cropY,  cropYMax)
        xx, yy =  np.meshgrid(x_r, y_r)
        z =  []
        s_x, s_y = [3, 3]
        for point in pred_Objects:
            m_x = point[0]
            m_y = point[1]
            z += [(1/(2*np.pi*s_x*s_y))*np.exp(-((xx-m_x)**2/(2*s_x**2))-((yy-m_y)**2/(2*s_y**2)))]
        if (frameID > 15)*(frameID < 40):
            m_x = 150 + cropX
            m_y = cropYMax
            s_x,  s_y = [6, 6]
            extra = [(1/(2*np.pi*s_x*s_y))*np.exp(-((xx-m_x)**2/(2*s_x**2))-((yy-m_y)**2/(2*s_y**2)))]
            filtered = (np.array(z+extra).sum(axis = 0))*np.copy(maxF)
        else:
            filtered = (np.array(z).sum(axis = 0))*np.copy(maxF)

        filtered = 255.*filtered/np.max(filtered)
        filtered[filtered < 5.] = 0
        filtered[filtered > 0.] = 255.


    return (filtered, minPixels, oneSheepPixels, z)
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

import myKalman as mKF

def extractDensityCoordinates(miniG):
     c=[]
     np.random.seed(1)
     for xi in range(np.shape(miniG)[1]):
         for yi in range(np.shape(miniG)[0]):
             coords = np.array([yi,xi]) + np.random.rand(int((np.round(miniG, 3)*100)[yi,xi]), 2)  - 0.5
             if len(coords) > 0:
                 c += coords.tolist()
     return np.array(c)[::10,::-1]

def predictKalmanIndv(loc):
    x,nextX,nextP = mKF.kalman(loc)
    return (np.transpose(nextX[:2]).tolist()[0], nextP)

def predictKalman(locs):
    locations =  np.array(map(lambda i:predictKalmanIndv(locs[:,i,:])[0], range(np.shape(locs)[1])))
    distributions = np.array(map(lambda i:predictKalmanIndv(locs[:,i,:])[1], range(np.shape(locs)[1])))
    return locations, distributions

def movingCropQuad(frameID, fullIm, quadLoc, cropV):
    cropX, cropY, cropXMax, cropYMax = cropV

    if frameID > 2:
        moveX, moveY = np.array(quadLoc)[-2] - np.array(quadLoc)[-1]
        cropX = int(cropX - moveX)
        cropY = int(cropY - moveY)
        cropXMax = int(cropXMax - moveX)
        cropYMax = int(cropYMax - moveY)

        cropV = [cropX, cropY, cropXMax, cropYMax]

    fullCropped = np.copy(fullIm)[cropY:cropYMax, cropX:cropXMax, :]
    return (fullCropped, cropV)

def getQuad(fullImg, quadLoc, cropV, darkTolerance, frameId):
    fullCropped, cropV = movingCropQuad(frameId, np.copy(fullImg), quadLoc, cropV)
    grey = np.copy(fullCropped)[:,:,0] - np.copy(fullCropped)[:,:,1]
    binary = np.copy(grey)
    binary[binary < darkTolerance] = 0
    binary[binary > darkTolerance] = 255

    quad = np.array(np.where(binary == 0)).mean(axis = 1)[::-1]


    quadLoc += [(quad+[cropV[0],cropV[1]]).tolist()]
    return quadLoc, cropV

def getPredictedID(pred, mask, cropV, rect):
    cropX,  cropY, cropXMax, cropYMax = cropV
    objects = pred.tolist()
    points =  np.array(map(int,  np.floor(np.array(objects) - np.array([cropX, cropY])).flatten()))
    points =  (points.reshape(len(points)/2,  2)).tolist()
    containedPoints = []
    ids = []
    for i in range(len(points)):
        point = points[i]
        if point[1] >= cropYMax - cropY:
            point[1] = cropYMax - cropY - 1

        if point[0] >= cropXMax - cropX:
            point[0] = cropXMax - cropX - 1

        if mask[point[1],point[0]] == 1:
            containedPoints += [point]
            ids += [i]
    if len(np.array(containedPoints)) == 0:
        points =  np.array(map(int,  np.ceil(np.array(objects) - np.array([cropX, cropY])).flatten()))
        points =  (points.reshape(len(points)/2,  2)).tolist()
        containedPoints = []
        ids = []
        for i in range(len(points)):
            point = points[i]
            if point[1] >= cropYMax - cropY:
                point[1] = cropYMax - cropY - 1

            if point[0] >= cropXMax - cropX:
                point[0] = cropXMax - cropX - 1

            if mask[point[1],point[0]] == 1:
                containedPoints += [point]
                ids += [i]

    if len(np.array(containedPoints)) == 0:
        pointsX =  np.array(map(int,  np.floor(np.array(objects)[:,0] - np.array([cropX])).flatten()))
        pointsY =  np.array(map(int,  np.ceil(np.array(objects)[:,1] - np.array([cropY])).flatten()))
        points =  np.append(pointsX.reshape(len(pointsX), 1), pointsY.reshape(len(pointsY),1), axis= 1).tolist()

        containedPoints = []
        ids = []
        for i in range(len(points)):
            point = points[i]
            if point[1] >= cropYMax - cropY:
                point[1] = cropYMax - cropY - 1

            if point[0] >= cropXMax - cropX:
                point[0] = cropXMax - cropX - 1
            if mask[point[1],point[0]] == 1:
                containedPoints += [point]
                ids += [i]

    if len(np.array(containedPoints)) == 0:
        pointsX =  np.array(map(int,  np.ceil(np.array(objects)[:,0] - np.array([cropX])).flatten()))
        pointsY =  np.array(map(int,  np.floor(np.array(objects)[:,1] - np.array([cropY])).flatten()))
        points =  np.append(pointsX.reshape(len(pointsX), 1), pointsY.reshape(len(pointsY),1), axis= 1).tolist()

        containedPoints = []
        ids = []
        for i in range(len(points)):
            point = points[i]
            if point[1] >= cropYMax - cropY:
                point[1] = cropYMax - cropY - 1

            if point[0] >= cropXMax - cropX:
                point[0] = cropXMax - cropX - 1
            if mask[point[1],point[0]] == 1:
                containedPoints += [point]
                ids += [i]
    if len(np.array(containedPoints)) == 0:
        x, y, w, h = rect
        ids = getPreviousID(pred, x, y, w, h, cropX, cropY, 0)
        containedPoints = pred[ids] - np.array([cropX, cropY])

        ids = ids.tolist()
        containedPoints = containedPoints.tolist()

    return np.array(containedPoints), ids


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
            print 'multiples'
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


def assignSheep(coords, dImg, prevId, centre=[0,0]):
    if type(coords[0]) != float:
        den2 = []
        relativeCoords = np.copy(coords) - np.array(centre)
        for predictionArea in np.array(dImg)[prevId]:
            den = []
            for point in np.fix(relativeCoords):
                den += [np.transpose(predictionArea)[int(point[0]+centre[0]), int(point[1]+centre[1])]]
            den2 += [1.0/(np.array(den)+1e-8)]
        _, assignment = linear_sum_assignment(den2)
        return np.array(prevId)[np.argsort(assignment)].tolist()
    else:
        return prevId



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

    #plt.imshow(iris)
    #plt.colorbar()
    #plt.show()
    threshold= 0.3*np.max(iris)
    _, thresh = cv2.threshold(iris,threshold,255,cv2.THRESH_BINARY)

    output = cv2.connectedComponentsWithStats(thresh, 8, cv2.CV_32S)
    centroids = output[3][output[2][:,-1] > 4]
    centroids = np.transpose(centroids)

    new_objects = np.copy(centroids[:,1:])#remove element 0 cost its the background!
    new_objects[0,:] += X
    new_objects[1,:] += Y
    new_objects = np.transpose(new_objects).tolist()
    final_objects = []
    sCov = []
    for i in np.array(range(output[0]))[output[2][:,-1] <= 4]:
        output[1][output[1] == i] = 0
    for i in range(len(np.unique(output[1])[1:])):
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
            cov = np.cov(np.transpose(np.where(output[1] == np.unique(output[1])[1:][i])), rowvar=False).flatten()[[0,1,-1]].tolist()
            s_x = np.sqrt(cov[2])
            s_y = np.sqrt(cov[0])
            rho = cov[1]/(s_x*s_y)
            sCov += [[s_x, s_y, rho]]

    return final_objects, sCov

def kmeansClustering(miniImg, miniImg2, numberPixels, X, Y, previous):
    threshold= 0.5*np.max(miniImg)
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
            clusterer = KMeans(n_clusters=approxNumber,random_state = 10)
            clusterer1 = clusterer.fit(np.transpose(np.where(miniImg > threshold)))
            cluster_centers = clusterer.cluster_centers_

            cluster_list = np.copy(cluster_centers)
            cluster_list[:,0] = cluster_centers[:,1] + X
            cluster_list[:,1] = cluster_centers[:,0] + Y

            new_objects =  cluster_list.tolist()
        else:
            new_objects = [[cX, cY]]

    else:
        clusterer = KMeans(n_clusters=previous, random_state = 10)
        clusterer1 = clusterer.fit(np.transpose(np.where(miniImg > threshold)))
        cluster_centers = clusterer1.cluster_centers_
        cluster_list = np.copy(cluster_centers)
        cluster_list[:,0] = cluster_centers[:,1] + X
        cluster_list[:,1] = cluster_centers[:,0] + Y


        new_objects =  cluster_list.tolist()

    sCov = []
    miniImg2[miniImg2 < 0.5*np.max(miniImg2)] = 0
    prediction = clusterer.predict(np.transpose(np.where(miniImg2 > 0)))
    for value in np.unique(prediction):
        s = np.transpose(np.where(miniImg2 > 0))[np.where(prediction == value)]
        labelMask = np.zeros(miniImg2.shape, dtype='uint8')
        for k in s:
            labelMask[k[0], k[1]] = 1
        output = cv2.connectedComponentsWithStats(labelMask, 4, cv2.CV_32S)
        for i in range(output[0]):
            if output[2][i,-1] <= 4:
                labelMask[output[1] == i] = 0
        cov = np.cov(np.transpose(np.where(labelMask > 0)), rowvar=False).flatten()[[0,1,-1]].tolist()
        s_x = np.sqrt(cov[2])
        s_y = np.sqrt(cov[0])
        rho = cov[1]/(s_x*s_y)
        sCov += [[s_x, s_y, rho]]




    return new_objects, sCov

def findVel(locs):
    vel = (locs[-1] - locs[-6])/5
    return vel

def predictEuler(locs, vel):
    prediction_Objects = locs[-1] + vel
    return prediction_Objects

def doCheck(fullC, objL, cx, cy, Img, new_i, new_k, rect, K, margin=0):
    X, Y, W, H = rect
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
        objL = objL + [[cx, cy]]
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

def movingCrop(frameID, full, sheepLoc, cropVector):
    cropX, cropY, cropXMax, cropYMax = cropVector

    if frameID < 2:
        cropVector = cropVector
    elif frameID < 50:
        moveX, moveY = np.min(sheepLoc[frameID-2], axis = 0) - np.min(sheepLoc[frameID-1], axis = 0)
        cropX = int(np.floor(cropX + moveX))
        cropY = int(np.floor(cropY + moveY))
        moveX, moveY = np.max(sheepLoc[frameID-2], axis = 0) - np.max(sheepLoc[frameID-1], axis = 0)
        cropXMax = int(np.floor(cropXMax - moveX))
        cropYMax = 2028
    else:
        moveX, moveY = np.min(sheepLoc[frameID-2], axis = 0) - np.min(sheepLoc[frameID-1], axis = 0)
        cropX = min(int(np.floor(cropX - moveX)), int(np.min(sheepLoc[frameID-1], axis = 0)[0] - 50))
        cropY = min(int(np.floor(cropY - moveY)), int(np.min(sheepLoc[frameID-1], axis = 0)[1] - 50))

        moveX, moveY = np.max(sheepLoc[frameID-2], axis = 0) - np.max(sheepLoc[frameID-1], axis = 0)
        cropXMax = max(int(np.floor(cropXMax - moveX)), int(np.max(sheepLoc[frameID - 1], axis = 0)[0] + 50))
        cropYMax = min(max(int(np.floor(cropYMax - moveY)), int(np.max(sheepLoc[frameID - 1], axis = 0)[1] + 50)), 2028)

    fullCropped = np.copy(full)[cropY:cropYMax, cropX:cropXMax, :]
    cropVector = [cropX, cropY, cropXMax, cropYMax]
    return (fullCropped, cropVector)

def createBinaryImage(frameID, sizeOfObject, pred_Objects, pred_Dist, cropVector, maxF, weight=None):
    cropX, cropY, cropXMax, cropYMax = cropVector
    if frameID <= 6:
        minPixels = sizeOfObject
        oneSheepPixels = 250
        filtered = np.copy(maxF)
        filtered[filtered < 0.2] = 0.0 #for removing extra shiney grass
        filtered[filtered > 0.] = 1.
        z = []

    elif frameID > 6:
        minPixels = 10
        oneSheepPixels = 0.7*250

        x_r =  np.arange(cropX,  cropXMax)
        y_r =  np.arange(cropY,  cropYMax)
        xx, yy =  np.meshgrid(x_r, y_r)
        z =  []
        rho = 0


        if frameID == 18:
            pred_Objects = pred_Objects[:-1]

        for i in range(len(pred_Objects)):
            if frameID <= 5:
                s_x, s_y = [2.8, 2.8]
            else:
                sCov = pred_Dist[-5:,i,:].mean(axis = 0)
                s_x, s_y, rho = sCov
                s_x = s_x*2
                s_y = s_y*2
            point = pred_Objects[i]
            m_x = point[0]
            m_y = point[1]
            z += [(1/(2*np.pi*s_x*s_y*np.sqrt(1-rho**2)))*np.exp(-((xx-m_x)**2/(s_x**2) + (yy-m_y)**2/(s_y**2) - 2*rho*(xx-m_x)*(yy-m_y)/(s_x*s_y))/(2*(1-rho**2)))]
            z[-1] = z[-1]/np.max(z[-1])

        if (frameID == 16):
            m_x = 150 + cropX
            m_y = cropYMax
            s_x, s_y = [6, 6]
            extra = [(1/(2*np.pi*s_x*s_y*np.sqrt(1-rho**2)))*np.exp(-((xx-m_x)**2/(s_x**2) + (yy-m_y)**2/(s_y**2) - 2*rho*(xx-m_x)*(yy-m_y)/(s_x*s_y))/(2*(1-rho**2)))]
            filtered = (np.array(z+extra).sum(axis = 0))*np.copy(maxF)
        elif (frameID == 18):
            m_x = 140 + cropX
            m_y = cropYMax
            s_x, s_y = [2, 2]
            rho = 0
            z += [(1/(2*np.pi*s_x*s_y*np.sqrt(1-rho**2)))*np.exp(-((xx-m_x)**2/(s_x**2) + (yy-m_y)**2/(s_y**2) - 2*rho*(xx-m_x)*(yy-m_y)/(s_x*s_y))/(2*(1-rho**2)))]

            m_x = 170 + cropX
            m_y = cropYMax
            s_x, s_y = [2, 2]
            z += [(1/(2*np.pi*s_x*s_y*np.sqrt(1-rho**2)))*np.exp(-((xx-m_x)**2/(s_x**2) + (yy-m_y)**2/(s_y**2) - 2*rho*(xx-m_x)*(yy-m_y)/(s_x*s_y))/(2*(1-rho**2)))]
            filtered = (np.array(z).sum(axis = 0))*np.copy(maxF)
        else:
            filtered = (np.array(z).sum(axis = 0))*np.copy(maxF)#(np.array(z).sum(axis = 0))*(weight)+np.copy(maxF)*(1-weight)

        filtered = filtered/np.max(filtered)
        filtered[filtered <= 1e-3] = 0
        filtered[filtered > 1e-3] = 1.


    return (filtered, minPixels, oneSheepPixels, z)

def bivariateNormal(xx,yy,m_x, m_y, s_x, s_y, rho):
    return (1/(2*np.pi*s_x*s_y*np.sqrt(1-rho**2)))*np.exp(-((xx-m_x)**2/(s_x**2) + (yy-m_y)**2/(s_y**2) - 2*rho*(xx-m_x)*(yy-m_y)/(s_x*s_y))/(2*(1-rho**2)))


def plotSheepCov(sCov, objectLoc, cropx, cropy):
    for i in range(len(sCov)):
        xx, yy = np.meshgrid(np.arange(objectLoc[i][0]-30-cropx,objectLoc[i][0]+30-cropx,0.1), np.arange(objectLoc[i][1]-30-cropy,objectLoc[i][1]+30-cropy, 0.1))
        z = bivariateNormal(xx, yy, objectLoc[i][0]-cropx, objectLoc[i][1]-cropy, sCov[i][0], sCov[i][1], sCov[i][2])
        plt.contour(xx,yy,z,cmap='plasma')

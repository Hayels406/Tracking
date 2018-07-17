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
from sklearn.mixture import BayesianGaussianMixture as bgm
from sklearn.mixture import GaussianMixture as gm
from scipy.stats import norm
from scipy.spatial.distance import mahalanobis

import myKalman as mKF

def getBlackSheep(fullImg, sLoc, cropV, darkTolerance, frameId):
    blackSheepImage, cropV = movingCropBS(frameId, np.copy(fullImg), sLoc, cropV)
    cropX, cropY, cropXMax, cropYMax = cropV

    shape = np.shape(blackSheepImage)
    noElements = np.product(np.array(np.shape(blackSheepImage)))
    fullArray = np.transpose(blackSheepImage.reshape(noElements/3,3))

    Cov = np.cov(fullArray)
    backgroundElem = np.product(np.array(np.shape(blackSheepImage[50:, 50:,:])))
    m = (blackSheepImage[50:, 50:,:]).reshape(backgroundElem/3, 3).mean(axis = 0)
    d = map(lambda i:mahalanobis(m, fullArray[:,i], np.linalg.inv(Cov)), range(np.shape(fullArray)[1]))

    mahalDist = np.array(d).reshape(*shape[:-1])
    mD = mahalDist/np.max(mahalDist)

    prod = np.product(np.copy(blackSheepImage), axis = 2)
    prod = prod*1./np.max(prod)

    grey = mD/np.max(mD) - prod*1./np.max(prod)
    grey[grey < 0] = 0

    binary = np.copy(grey)
    binary[binary < darkTolerance] = 0
    binary[binary > darkTolerance] = 1

    labels = measure.label(binary, neighbors=8, background=0)
    labelMask = np.zeros(labels.shape, dtype="uint8")
    labelPixels = map(lambda label: (labels == label).sum(), np.unique(labels))
    labelMask[labels == np.where(labelPixels == np.max(labelPixels[1:]))[0][0]] = 1
    cnts = cv2.findContours(np.copy(labelMask), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1][0]
    rectangle = cv2.boundingRect(cnts)
    x,y,w,h = rectangle
    miniGrey = np.copy(grey*labelMask)[y: y + h, x: x + w]

    c = extractDensityCoordinates(miniGrey)
    mm = bgm(n_components = 1, covariance_type='tied', random_state=1,max_iter=1000,tol=1e-6).fit(c.tolist())
    blackSheep = (mm.means_ + [x+cropX,y+cropY]).tolist()
    cov = mm.covariances_.flatten()[[0,1,-1]]
    s_x = np.sqrt(cov[0])
    s_y = np.sqrt(cov[2])
    rho = cov[1]/(s_x*s_y)
    bsCov = [[s_x, s_y, rho]]

    sLoc += blackSheep
    return sLoc, bsCov, cropV

def movingCropBS(frameID, fullIm, loc, cropV):
    cropX, cropY, cropXMax, cropYMax = cropV

    if frameID == 1:
        center = map(int, loc[-1])
        cropX = center[0] - 50
        cropXMax = center[0] + 50
        cropY = center[1] - 50
        cropYMax = center[1] + 50

        cropV = [cropX, cropY, cropXMax, cropYMax]
    if frameID > 2:
        moveX, moveY = np.array(loc)[-2] - np.array(loc)[-1]
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
    gmm = gm(n_components=2, covariance_type='full').fit((binary.flatten()).reshape(-1,1))
    upper = norm(loc = gmm.means_[np.where(gmm.means_ == np.max(gmm.means_))[0][0]], scale = np.sqrt(gmm.covariances_[np.where(gmm.means_ == np.max(gmm.means_))[0][0]]))
    darkTolerance = upper.ppf(0.01)
    binary[binary < darkTolerance] = 0

    quad = np.array(np.where(binary == 0)).mean(axis = 1)[::-1]


    quadLoc += [(quad+[cropV[0],cropV[1]]).tolist()]
    return quadLoc, cropV

def getQuadCJ2(fullImg, quadLoc, cropV, darkTolerance, frameId):
    fullCropped, cropV = movingCropQuad(frameId, np.copy(fullImg), quadLoc, cropV)

    grey = np.copy(fullCropped)[:,:,0]
    grey = grey - np.min(grey)
    grey = grey*1./np.max(grey)

    binary = np.copy(grey)
    gmm = gm(n_components=2, covariance_type='full').fit((binary.flatten()).reshape(-1,1))
    upper = norm(loc = gmm.means_[np.where(gmm.means_ == np.max(gmm.means_))[0][0]], scale = np.sqrt(gmm.covariances_[np.where(gmm.means_ == np.max(gmm.means_))[0][0]]))
    darkTolerance = upper.ppf(0.25)
    binary[binary > darkTolerance] = 1

    quad = np.array(np.where(binary == 1)).mean(axis = 1)[::-1]


    quadLoc += [(quad+[cropV[0],cropV[1]]).tolist()]
    return quadLoc, cropV

def movingCropQuad(frameID, fullIm, quadLoc, cropV):
    cropX, cropY, cropXMax, cropYMax = cropV
    if frameID >= 1:
        center = map(int, quadLoc[-1])
        cropX = center[0] - 50
        cropXMax = center[0] + 50
        cropY = center[1] - 50
        cropYMax = center[1] + 50

        cropV = [cropX, cropY, cropXMax, cropYMax]

    fullCropped = np.copy(fullIm)[cropY:cropYMax, cropX:cropXMax, :]
    return (fullCropped, cropV)

def movingCropTFRL(frameID, full, sheepLoc, cropVector):
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

def movingCrop(frameID, full, sheepLoc, cropVector):
    cropX, cropY, cropXMax, cropYMax = cropVector

    if frameID < 2:
        cropVector = cropVector
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

def createBinaryImageTFRL(frameID, pred_Objects, pred_Dist, cropVector, maxF, darkTolerance, weight=None):
    cropX, cropY, cropXMax, cropYMax = cropVector
    if frameID <= 20:
        filtered = np.copy(maxF)
        filtered[filtered < darkTolerance] = 0.0 #for removing extra shiney grass
        filtered[filtered > 0.] = 1.
        z = []

    elif frameID > 6:

        x_r =  np.arange(cropX,  cropXMax)
        y_r =  np.arange(cropY,  cropYMax)
        xx, yy =  np.meshgrid(x_r, y_r)
        z =  []
        rho = 0

        #if frameID == 18:
        #    pred_Objects = pred_Objects[:-1]

        for i in range(len(pred_Objects)):
            if frameID <= 5:
                s_x, s_y = [2.8, 2.8]
            else:
                sCov = pred_Dist[-5:,i,:].mean(axis = 0)
                s_x, s_y, rho = sCov
                s_x = s_x
                s_y = s_y
            point = pred_Objects[i]
            m_x = point[0]
            m_y = point[1]
            z += [(1/(2*np.pi*s_x*s_y*np.sqrt(1-rho**2)))*np.exp(-((xx-m_x)**2/(s_x**2) + (yy-m_y)**2/(s_y**2) - 2*rho*(xx-m_x)*(yy-m_y)/(s_x*s_y))/(2*(1-rho**2)))]
            z[-1] = z[-1]/np.max(z[-1])

        filtered = (np.array(z).sum(axis = 0))
        filtered = filtered/np.max(filtered)
        filtered[filtered <= 0.5] = 0
        filtered[filtered > 0.5] = 1.

        filtered = np.maximum(np.copy(maxF),filtered)
        filtered[filtered < darkTolerance] = 0.0 #for removing extra shiney grass
        filtered[filtered > 0.] = 1.


    return (filtered, z)

def createBinaryImage(frameID, pred_Objects, pred_Dist, cropVector, maxF, darkTolerance, weight=None):
    cropX, cropY, cropXMax, cropYMax = cropVector

    filtered = np.copy(maxF)
    filtered[filtered < darkTolerance] = 0.0 #for removing extra shiney grass
    filtered[filtered > 0.] = 1.
    z = []

    if frameID > 1:
        x_r =  np.arange(cropX,  cropXMax)
        y_r =  np.arange(cropY,  cropYMax)
        xx, yy =  np.meshgrid(x_r, y_r)

        for i in range(len(pred_Objects)):
            if frameID < 5:
                sCov = pred_Dist[-5:,i,:].mean(axis = 0)
            else:
                sCov = pred_Dist[:,i,:].mean(axis = 0)
            s_x, s_y, rho = sCov
            s_x = 0.8*s_x
            s_y = 0.8*s_y
            point = pred_Objects[i]
            m_x = point[0]
            m_y = point[1]
            z += [(1/(2*np.pi*s_x*s_y*np.sqrt(1-rho**2)))*np.exp(-((xx-m_x)**2/(s_x**2) + (yy-m_y)**2/(s_y**2) - 2*rho*(xx-m_x)*(yy-m_y)/(s_x*s_y))/(2*(1-rho**2)))]
            z[-1] = z[-1]/np.max(z[-1])


    return (filtered, z)

def extractDensityCoordinates(miniG):
     c=[]
     np.random.seed(1)
     for xi in range(np.shape(miniG)[1]):
         for yi in range(np.shape(miniG)[0]):
             coords = np.array([yi,xi]) + np.random.rand(3*int((np.round(miniG[yi,xi], 2)*100)), 2) - 0.5 #if you have pure white you end up with 300 coords for that pixel
             if len(coords) > 0:
                 c += coords.tolist()
     return np.array(c)[:,::-1]

def organiseLocations(objects, assign, frameId):
    if frameId > 0:
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
        objects = np.array(objects)[assign]
    return objects

def organiseLocationsTFRL(objects, assign, frameId):
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

def getPreviousID(prev, X, Y, W, H, cropx, cropy):
    return np.where((prev[:,0] > X+cropx) & (prev[:,0] < X+W+cropx) & (prev[:,1] < Y+H+cropy) & (prev[:,1] > Y+cropy))[0]

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
        ids = getPreviousID(pred, x, y, w, h, cropX, cropY)
        containedPoints = pred[ids] - np.array([cropX, cropY])

        ids = ids.tolist()
        containedPoints = containedPoints.tolist()

    return np.array(containedPoints), ids


def predictEuler(locs):
    if len(locs) > 5:
        vel = (locs[-1] - locs[-6])/5
    else:
        vel = (locs[-1] - locs[0])/(len(locs)-1)
    prediction_Objects = locs[-1] + vel
    return prediction_Objects

def predictKalmanIndv(loc):
    x,nextX,nextP = mKF.kalman(loc)
    return (np.transpose(nextX[:2]).tolist()[0], nextP)

def predictKalman(locs):
    locations =  np.array(map(lambda i:predictKalmanIndv(locs[:,i,:])[0], range(np.shape(locs)[1])))
    distributions = np.array(map(lambda i:predictKalmanIndv(locs[:,i,:])[1], range(np.shape(locs)[1])))
    return locations, distributions

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

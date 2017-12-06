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
            clusterer = KMeans(n_clusters=n_clusters, random_state=10)

            cluster_labels = clusterer.fit_predict(np.transpose(np.where(miniImg > threshold)))

            av_score += [silhouette_score(np.transpose(np.where(miniImg > threshold)), cluster_labels)]

        approxNumber =  np.where(av_score == np.max(av_score))[0][0] + 2
        if approxNumber < np.shape(np.transpose(np.where(miniImg > threshold)))[0] - 1:
            clusterer = KMeans(n_clusters=approxNumber, random_state=10).fit(np.transpose(np.where(miniImg > threshold)))
            cluster_centers = clusterer.cluster_centers_

            cluster_list = np.copy(cluster_centers)
            cluster_list[:,0] = cluster_centers[:,1] + X
            cluster_list[:,1] = cluster_centers[:,0] + Y

            new_objects =  cluster_list.tolist()
        else:
            new_objects = [[cX, cY]]

    else:
        #plt.imshow(np.transpose(np.where(miniImg > threshold)))
        #plt.axes().set_aspect('auto')
        #plt.show()
        clusterer = KMeans(n_clusters=previous, random_state=10)
        clusterer = clusterer.fit(np.transpose(np.where(miniImg > threshold)))
        cluster_centers = clusterer.cluster_centers_
        cluster_list = np.copy(cluster_centers)
        cluster_list[:,0] = cluster_centers[:,1] + X
        cluster_list[:,1] = cluster_centers[:,0] + Y


        new_objects =  cluster_list.tolist()

    return new_objects

def track(videoLocation, plot, darkTolerance, sizeOfObject, radi, method = 'clustering', test = False, lowerBoundY = 0, upperBoundY = 2500, lowerBoundX = 0, upperBoundX = 3000):

    sheepLocations = []
    frameID = 0

    cap = cv2.VideoCapture(videoLocation)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print length
    while(frameID < 0):
        ret, frame = cap.read()
        print frameID
        frameID +=1
    while(frameID <= 10):
        plt.close('all')
        ret, frame = cap.read()
        if ret == True:

            full = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            if frameID < 2:
                cropX = 1000
                cropY = 1000
                cropXMax = 2000
                cropYMax = 2000
            else:
                print 'move window'

            fullCropped = np.copy(full)[cropY:cropYMax, cropX:cropXMax, :]

            if test == True:
                plt.clf()
                plt.imshow(fullCropped)
                plt.axis('off')
                id = 1
                plt.savefig('./test'+str(id)+'.png', bbox_inches='tight')

            grey = fullCropped[:,:,2]#cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

            grey[grey < darkTolerance] = 0.0
            grey[grey > darkTolerance+10.] = 255.
            if test == True:
                plt.imshow(grey)
                plt.axis('off')
                id += 1
                plt.savefig('./test'+str(id)+'.png', bbox_inches='tight')

            img = cv2.GaussianBlur(grey,(5,5),2)
            if test == True:
                plt.imshow(img)
                id += 1
                plt.savefig('./test'+str(id)+'.png', bbox_inches='tight')

            maxfilter = ndimage.maximum_filter(img, size=2)
            if test == True:
                plt.imshow(maxfilter)
                plt.axis('off')
                id += 1
                plt.savefig('./test'+str(id)+'.png', bbox_inches='tight')

            filtered = np.copy(maxfilter)
            filtered[filtered < 65.] = 0.0 #for removing extra shiney grass
            filtered[filtered > 0.] = 255.
            if test == True:
                plt.imshow(filtered)
                plt.axis('off')
                id += 1
                plt.savefig('./test'+str(id)+'.png', bbox_inches='tight')

            labels = measure.label(filtered, neighbors=8, background=0)

            objectLocations = []
            r = []
            pix = []
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

                if numPixels > sizeOfObject:
                    cnts = cv2.findContours(labelMask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1][0]
                    ((cX, cY), radius) = cv2.minEnclosingCircle(cnts)

                    if radius > radi:
                        if radius < 200:
                            if numPixels < 250:
                                objectLocations = objectLocations + [[cX, cY]]
                            else:
                                x,y,w,h = cv2.boundingRect(cnts)

                                miniImage = img[y: y + h, x: x + w]
                                if frameID > 0:
                                    lastT = np.array(sheepLocations[-1])
                                    leeway = 3
                                    prev =  len(np.where((lastT[:,0] > x+cropX-leeway) & (lastT[:,0] < x+w+cropX+leeway) & (lastT[:,1] < y+h+cropY+leeway) & (lastT[:,1] > y+cropY-leeway))[0])
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
                                elif num_new_objects_i == num_new_objects_k:
                                    C = cdist(new_objects, new_objects_K)
                                    _, assigment = linear_sum_assignment(C)
                                    sum_C = 0
                                    for i in range(num_new_objects_i):
                                        sum_C += C[i,  assigment[i]]
                                    if sum_C/num_new_objects_i < 3.5:
                                        check = 'Off'
                                        objectLocations = objectLocations + new_objects_K
                                elif (frameID > 0) & (num_new_objects_i != prev):
                                    check = 'Off'
                                    objectLocations += new_objects_K
                                if frameID == 5:
                                    check = 'On'
                                if check == 'On':
                                    plt.close()
                                    plt.figure(dpi = 300)
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
                                    plt.scatter(cX, cY, color = 'k', label = 'centre')
                                    plt.scatter(np.array(new_objects_K)[:,0], np.array(new_objects_K)[:,1], color = 'green', marker = '^', label = 'kmeans')
                                    if num_new_objects_i > 0:
                                        plt.scatter(np.array(new_objects)[:,0], np.array(new_objects)[:,1], color = 'red', label = 'iris', alpha = 0.5)
                                    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                                    plt.pause(0.5)
                                    if prev  < 0:
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




            print 'frameID: ', frameID, ', No. objects located: ', len(objectLocations)

            objectLocations = np.array(objectLocations)
            objectLocations[:, 0] += cropX
            objectLocations[:, 1] += cropY
            if plot != 'N':
                plt.close()
                if test == True:
                    fig = plt.figure(frameon=False)
                    fig.set_size_inches(5,5) #Set image to 1 inch by 1 inch, then control the resolution through tweaking "dots per inch"
                    ax = plt.Axes(fig, [0., 0., 1., 1.])
                    ax.set_axis_off()
                    ax.set_aspect('equal')
                    fig.add_axes(ax)
                    ax.imshow(fullCropped)
                    ax.scatter(np.array(objectLocations)[:, 0] - cropX, np.array(objectLocations)[:, 1] - cropY, s = 1.)
                    id += 1
                    fig.savefig('originalColour'+str(frameID-1)+'.png', dpi = 200)
                else:
                    plt.imshow(fullCropped)
                    plt.scatter(np.array(objectLocations)[:, 0] - cropX, np.array(objectLocations)[:, 1] - cropY, s = 1.)
                    plt.axis('off')
                    if plot == 's':
                        plt.savefig('/home/b1033128/Documents/throughFenceRL/'+str(frameID).zfill(4), bbox_inches='tight')
                    else:
                        plt.pause(15)


            objectLocations = objectLocations.tolist()
            sheepLocations = sheepLocations + [objectLocations]
            if len(objectLocations) < 141:
                print 'you lost a sheep'
                break
            frameID += 1
    cap.release()
    plt.clf()
    return np.array(sheepLocations)

locations = track('/home/b1033128/Documents/throughFenceRL.mp4', plot = 's', test = False, darkTolerance = 173.5, sizeOfObject = 60, radi = 5.)
np.save('locfull.npy',locations)

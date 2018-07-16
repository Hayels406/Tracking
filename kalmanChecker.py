import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import cv2
from sklearn.mixture import BayesianGaussianMixture as bgm
from trackingFunctions_ngs54_changes import extractDensityCoordinates
from myKalman import kalman
from skimage import measure
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

save = '/data/b1033128/Tracking/kalmanChecking/'

x, y = np.mgrid[0:7:.01, 0:7:.01]
pos = np.dstack((x, y))

m_x1 = 1.5
m_y1 = 1.0
rv1 = multivariate_normal([m_x1, m_y1], [[0.7, 0.3], [0.3, 0.25]])

m_x2 = 4.5
m_y2 = 5.0
rv2 = multivariate_normal([m_x2, m_y2], [[0.5, -0.3], [-0.3, 0.4]])

plt.close('all')
plt.contour(x, y, rv1.pdf(pos))
plt.contour(x, y, rv2.pdf(pos))
plt.savefig(save + 'fitNormalContours.png')

plt.figure()
image1 = np.transpose(rv1.pdf(pos))[::-1]
image1 = image1/np.max(image1)
image2 = np.transpose(rv2.pdf(pos))[::-1]
image2 = image2/np.max(image2)
image = image1+image2
image = image/np.max(image)
plt.imshow(image, cmap = 'gray')
plt.gca().set_axis_off()
plt.savefig(save + 'fitNormalImage.png')


frames = []
trueV = []
sheepLocations = []
sheepCov = []

for i in range(10):
    print i
    objectLocations = []
    frameCov = []
    m_y1 = 1.0+np.random.normal(0.25, 0.01**2)*i
    m_x1 = 1.5+np.random.normal(0.1, 0.01**2)*i
    m_y2 = 5.0-np.random.normal(0.25, 0.01**2)*i
    m_x2 = 4.5-np.random.normal(0.1, 0.01**2)*i
    rv1 = multivariate_normal([m_x1, m_y1], [[0.7, 0.3], [0.3, 0.25]])
    rv2 = multivariate_normal([m_x2, m_y2], [[0.5, -0.3], [-0.3, 0.4]])
    image1 = np.transpose(rv1.pdf(pos))[::-1]
    image1 = image1/np.max(image1)
    image2 = np.transpose(rv2.pdf(pos))[::-1]
    image2 = image2/np.max(image2)
    image = image1+image2
    image = image/np.max(image)
    binary = np.zeros(image.shape, dtype="uint8")
    binary[image < 0.5] = 0
    binary[image >= 0.5] = 1
    plt.figure()
    plt.imshow(binary, cmap='gray')
    m = np.array(np.where(image1 == np.max(image1)))[::-1].flatten()
    trueV += [m]
    plt.scatter(m[0], m[1], color = 'r', label = 'Actual Mean')
    m = np.array(np.where(image2 == np.max(image2)))[::-1].flatten()
    trueV += [m]
    plt.scatter(m[0], m[1], color = 'r')
    plt.gca().set_axis_off()
    frames += [binary]

    labels = measure.label(binary, neighbors=8, background=0)
    for label in np.unique(labels):
        if label == 0:
            continue
        else:
            if (label == 1) and (np.unique(labels)[-1] == 1):
                comp = 2
            else:
                comp = 1
            labelMask = np.zeros(binary.shape, dtype="uint8")
            labelMask[labels == label] = 1
            cnts = cv2.findContours(np.copy(labelMask), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1][0]
            rectangle = cv2.boundingRect(cnts)
            x,y,w,h = rectangle
            miniGrey = np.copy(image*labelMask)[y: y + h, x: x + w]
            c = extractDensityCoordinates(miniGrey[::10,::10])
            mm = bgm(n_components = comp, covariance_type='tied', random_state=1,max_iter=1000,tol=1e-6).fit(c.tolist())
            objectLocations += ((mm.means_*10+ [x,y])/100.).tolist()
            cov = mm.covariances_.flatten()[[0,1,-1]]
            s_x = np.sqrt(cov[0])
            s_y = np.sqrt(cov[2])
            rho = cov[1]/(s_x*s_y)
            frameCov += [[s_x, s_y, rho]]


    plt.scatter(np.array(objectLocations)[:, 0]*100, np.array(objectLocations)[:, 1]*100, color='b', marker = 'x', label = 'Fitted Mean')
    if (i > 0):
        finalDist = cdist(sheepLocations[-1], objectLocations)
        _, assignmentVec = linear_sum_assignment(np.max(finalDist)*(finalDist/np.max(finalDist))**3)
        finalLocations = np.array(objectLocations)[assignmentVec]
    else:
        finalLocations = objectLocations
    sheepLocations += [np.array(finalLocations).tolist()]
    sheepCov += [frameCov]

    if i > 1:
        for j in range(2):
            _, prediction, _ = kalman(np.array(sheepLocations)[:-1,j], cov = frameCov[-1])
            prediction = np.array(prediction[:2]).flatten()
            if j == 0:
                plt.scatter(prediction[0]*100, prediction[1]*100, color='g', marker = 'x', label = 'Predicted Mean')
            else:
                plt.scatter(prediction[0]*100, prediction[1]*100, color='g', marker = 'x')


    plt.legend(loc = 'best')
    plt.savefig(save + 'fitLocation'+str(i)+'.png')

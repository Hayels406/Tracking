import numpy as np
import matplotlib.pyplot as plt
import os

#if os.getcwd().rfind('hayley') > 0:
#    videoLocation = '/users/hayleymoore/Documents/PhD/Tracking/throughFenceRL.mp4'
#    save = '/users/hayleymoore/Documents/PhD/Tracking/throughFenceRL/'
#    plt.ion()
#elif os.getcwd().rfind('Uni') > 0:
#    videoLocation = '/home/b1033128/Documents/throughFenceRL.mp4'
#    save = '/home/b1033128/Documents/throughFenceRL/'
#    dell = True
#    brk = False
#else:#Kiel
#    videoLocation = '/data/b1033128/Videos/throughFenceRL.mp4'
#    save = '/data/b1033128/Tracking/throughFenceRL/'
#    dell = False
#    plt.ion()


#cap = cv2.VideoCapture(videoLocation)
#length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#print 'You have', length, 'frames', videoLocation
#frameID = 0
#if (videoLocation.rfind('data') > 0) and (videoLocation.rfind('throughFenceRL') > 0):
#    print 'Skipping first 15 frames'
#    while(frameID <= 15):
#        ret, frame = cap.read()
#        frameID += 1
#
#    if frameID > 0:
#        frameID = 0
#while(frameID <= 50):
#    ret, frame = cap.read()
#    print frameID
#    frameID +=1
#
#plt.close('all')
#plt.figure(figsize=(7,4))
#plt.imshow(frame)
#sheep = np.load(save+'loc50.npy')[0]

def xPrimeFunc(A, x):
    return A*x

def pPrimeFunc(A, P, Q):
    return A*P*np.transpose(A) + Q

def Kk(pPrime, R, H):
    return pPrime*np.transpose(H)*np.linalg.inv(R + H*pPrime*np.transpose(H))

def xFilteredFunc(xPrime, k, zk, H):
    return xPrime + k*(zk - H*xPrime)

def pFilteredFunc(pPrime, k, H):
    return pPrime - k*H*pPrime


def kalman(z):
    x = []
    p = []

    A = np.matrix(np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]]))
    H = np.matrix(np.array([[1,0,0,0],[0,1,0,0]]))
    R = np.matrix(np.eye(2)*0.01) #noise in the measurements
    Q = np.matrix(np.eye(4)*0.0001) #noise in the true values

    initVel = initVel = [z[1,0]-z[0,0], z[1,1]-z[0,1]]

    for zt in z:
        if np.all(zt == z[0]):
            xt = np.transpose(np.matrix(np.append(zt, initVel)))
            pt = np.ones((np.shape(xt)[0], np.shape(xt)[0]))
        else:
            xt = x[-1]
            pt = p[-1]

        xPrime = xPrimeFunc(A, xt)
        pPrime = pPrimeFunc(A, pt, Q)

        kt = Kk(pPrime, R, H)
        x.append(xFilteredFunc(xPrime, kt, np.transpose(np.matrix(zt)), H))
        p.append(pFilteredFunc(pPrime, kt, H))

    return (x, xPrimeFunc(A,x[-1]), pPrimeFunc(A,p[-1],Q))

#z = np.array([0.39, 0.50, 0.48, 0.29, 0.25, 0.32, 0.34, 0.48, 0.41, 0.45])
#z = sheep[:,-10,:]
#initVel = [z[1,0]-z[0,0], z[1,1]-z[0,1]]




#x, p, A = kalman(z, R, Q)

#x = np.array(x)


#plt.scatter(z[:,0], z[:,1], label = 'Observed')
#plt.scatter(x[:,0], x[:,1], label = 'My Kalman')
#plt.legend(loc = 'best')

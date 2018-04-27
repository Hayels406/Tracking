import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
import cv2
import os

if os.getcwd().rfind('hayley') > 0:
    videoLocation = '/users/hayleymoore/Documents/PhD/Tracking/throughFenceRL.mp4'
    save = '/users/hayleymoore/Documents/PhD/Tracking/throughFenceRL/'
elif os.getcwd().rfind('Uni') > 0:
    videoLocation = '/home/b1033128/Documents/throughFenceRL.mp4'
    save = '/home/b1033128/Documents/throughFenceRL/'
    dell = True
    brk = False
else:#Kiel
    videoLocation = '/data/b1033128/Videos/throughFenceRL.mp4'
    save = '/data/b1033128/Tracking/throughFenceRL/'
    dell = False
    plt.ion()


cap = cv2.VideoCapture(videoLocation)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print 'You have', length, 'frames', videoLocation
frameID = 0
if (videoLocation.rfind('data') > 0) and (videoLocation.rfind('throughFenceRL') > 0):
    print 'Skipping first 15 frames'
    while(frameID <= 15):
        ret, frame = cap.read()
        frameID += 1

    if frameID > 0:
        frameID = 0
while(frameID <= 50):
    ret, frame = cap.read()
    print frameID
    frameID +=1
plt.close('all')
plt.figure(figsize=(14,8))
plt.imshow(frame)
sheep = np.load(save+'loc50.npy')[0]
for i in [12,10,9,5]:
    MarkedMeasure = sheep[:,-i,:]
    Transition_Matrix=[[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]]
    Observation_Matrix=[[1,0,0,0],[0,1,0,0]]
    xinit=MarkedMeasure[0,0]
    yinit=MarkedMeasure[0,1]
    vxinit=MarkedMeasure[1,0]-MarkedMeasure[0,0]
    vyinit=MarkedMeasure[1,1]-MarkedMeasure[0,1]
    initstate=[xinit,yinit,vxinit,vyinit]
    initcovariance=1.0e-3*np.eye(4)
    transistionCov=1.0e-4*np.eye(4)
    observationCov=1.0e-2*np.eye(2)
    kf=KalmanFilter(transition_matrices=Transition_Matrix,
                observation_matrices =Observation_Matrix,
                initial_state_mean=initstate,
                initial_state_covariance=initcovariance,
                transition_covariance=transistionCov,
                observation_covariance=observationCov)
    (filtered_state_means, filtered_state_covariances) = kf.filter(MarkedMeasure)
    plt.scatter(MarkedMeasure[:,0],MarkedMeasure[:,1],marker = 'x',label='Measured', color='red')
    plt.scatter(filtered_state_means[:,0],filtered_state_means[:,1],marker = 'o',label='Kalman output',color='blue')
    prediction = np.array(kf.filter_update(filtered_state_means[-1], filtered_state_covariances[-1], MarkedMeasure[-1])[0][:2])
    plt.scatter(prediction[0], prediction[1], marker = 's', color='k', label = 'Prediction')
    p_vel = MarkedMeasure[-1] + (MarkedMeasure[-1]-MarkedMeasure[-6])/5
    plt.scatter(p_vel[0], p_vel[1], color='g', marker = '^', label = 'Previous Prediction')
    if i == 12:
        plt.legend(loc='best')
plt.xlim(1600,1700)
plt.ylim(1820,1760)
plt.title("Constant Velocity Kalman Filter")
plt.show()

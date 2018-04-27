import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def movingCropQuad(frameID, fullIm, quadLoc, cropV):
    cropX, cropY, cropXMax, cropYMax = cropV

    if frameID > 2:
        moveX, moveY = np.array(quadLoc)[-2] - np.array(quadLoc)[-1]
        cropX = int(cropX - moveX)
        cropY = int(cropY - moveY)
        cropXMax = int(cropXMax - moveX)
        cropYMax = int(cropYMax - moveY)

        cropV = [cropX, cropY, cropXMax, cropYMax]

    fullCropped = np.copy(full)[cropY:cropYMax, cropX:cropXMax, :]
    return (fullCropped, cropV)

if os.getcwd().rfind('Uni') > 0:
    videoLocation = '/home/b1033128/Documents/throughFenceRL.mp4'
    save = '/home/b1033128/Documents/throughFenceRL/'
    dell = True
    brk = False
elif os.getcwd().rfind('hayley') > 0:
    videoLocation = '/users/hayleymoore/Documents/PhD/Tracking/throughFenceRL.mp4'
    save = '/users/hayleymoore/Documents/PhD/Tracking/throughFenceRL/'
else:#Kiel
    videoLocation = '/data/b1033128/Videos/throughFenceRL.mp4'
    save = '/data/b1033128/Tracking/throughFenceRL/'
    dell = False

plot = 's'
darkTolerance = 100.
restart = 0

quadLocation = []
frameID = 0
cropVector = [2000,1500,2200,1700]


cap = cv2.VideoCapture(videoLocation)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print 'You have', length, 'frames', videoLocation

if (videoLocation.rfind('data') > 0) and (videoLocation.rfind('throughFenceRL') > 0):
    print 'Skipping first 15 frames'
    while(frameID <= 15):
        ret, frame = cap.read()
        frameID += 1

    if frameID > 0:
        frameID = 0

while(frameID <= 174):
    print frameID
    plt.close('all')
    ret, frame = cap.read()
    if ret == True:
        full = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        fullCropped, cropVector = movingCropQuad(frameID, np.copy(full), quadLocation, cropVector)
        grey = np.copy(fullCropped)[:,:,0] - np.copy(fullCropped)[:,:,1]
        binary = np.copy(grey)
        binary[binary < darkTolerance] = 0
        binary[binary > darkTolerance] = 255

        quad = np.array(np.where(binary == 0)).mean(axis = 1)[::-1]


        quadLocation += [(quad+[cropVector[0],cropVector[1]]).tolist()]
    frameID+=1

np.save(save+'quadLoc'+str(frameID-1), np.array(quadLocation))

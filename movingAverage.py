import numpy as np
import matplotlib.pyplot as plt
T = -50
def movingaverage(interval, window_size):
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'valid')

sheep = np.load('/Users/hayleymoore/Documents/PhD/Tracking/loc50.npy')[0]


def backwardFiniteDifference5(prev6time):
    return -prev6time[0]/5. + 5.*prev6time[1]/4. - 10.*prev6time[2]/3. + 5.*prev6time[3] - 5.*prev6time[4] + 137.*prev6time[5]/60.

def backwardFiniteDifference4(prev):
    return prev[0]/4. - 4.*prev[1]/3. + 3.*prev[2] - 4.*prev[3] + 25.*prev[4]/12.

def backwardFiniteDifference3(prev):
    return 3.*prev[-1]/2. - 2.*prev[-2] + prev[-3]/2.

ma_x = movingaverage(sheep[:,T,0], 2)
ma_y = movingaverage(sheep[:,T,1], 2)

plt.ion()
plt.scatter(sheep[:,T,0], sheep[:,T,1])
plt.plot(ma_x, ma_y)
v_x = backwardFiniteDifference5(ma_x[-6:])
v_y = backwardFiniteDifference5(ma_y[-6:])
p = np.array([sheep[-2,T,0], sheep[-2,T,1]])+np.array([v_x,v_y])
plt.scatter(p[0], p[1])

import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import rc
from glob import glob
import os

plt.close('all')

video = sys.argv[1]
videoLocation = '/data/b1033128/Videos/'+video
save = '/data/b1033128/Tracking/'+video[:-4] +'/'
ymax, xmax = (1520, 2704)

if (videoLocation.rfind('CaseH2') > 0) or (videoLocation.rfind('CaseJ2') > 0) or (videoLocation.rfind('CaseH3') > 0):
    video = 'video2'
if (video[:-4]=='CaseJ'):
    save = '/data/b1033128/Tracking/'+video[:-4] +'1/'
    video = 'CJ1'
if (video[:-4]=='CaseH'):
    save = '/data/b1033128/Tracking/'+video[:-4] +'1/'
    video = 'CH1'

data = glob(save+'Final-loc*')[-1]
sheep = np.load(data)
quad = np.load(glob(save+'Final-quad*')[-1])
sheep =  map(np.array,  sheep)
if video == 'video2':
    blackSheepData = glob(save+'Final-blackSheep*')[-1]
    blackSheep = np.load(blackSheepData)
print 'You have analysed', len(sheep), 'frames'


rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
if type(sheep) != list:
    sheep =  np.array(sheep.tolist())
else:
    sheep = np.array(sheep)
while len(sheep[-1]) != len(sheep[0]):
	sheep = np.array(sheep[:-1])
N = np.shape(sheep)[1]
S = 0
F = len(sheep)

if video == 'video2':
    corners = np.array([[2704,  463],
                        [2174,  517],
                        [2063,  371],
                        [1952,  364],
                        [1717,  374],
                        [1389,  390],
                        [1399,  566],
                        [1389,  390],
                        [1098,  401],
                        [903,   405],
                        [903,   349],
                        [723,   357],
                        [600,   363],
                        [600,   288],
                        [89,    307],
                        [11,    1502],
                        [494,   1439],
                        [894,   1485],
                        [1182,  1442],
                        [1275,  1419],
                        [1462,  1387],
                        [1421,  893],
                        [1462,  1387],
                        [1968,  1293],
                        [2036,  1515],
                        [1411,  731],
                        [1418,  826]  ])
    corners = np.abs(corners - [0, ymax])
    ax = plt.gca()
    ax.add_artist(plt.Circle(tuple(np.abs(np.array([1838, 369]) - [0,ymax])), 110, color='g'))
    ax.add_artist(plt.Circle(tuple(np.abs(np.array([ 903, 396]) - [0,ymax])), 190, color='g'))
    ax.add_artist(plt.Circle(tuple(np.abs(np.array([ 377, 290]) - [0,ymax])), 45, color='g'))
    ax.add_artist(plt.Circle(tuple(np.abs(np.array([ 200, 450]) - [0,ymax])), 120, color='g'))
    ax.add_artist(plt.Circle(tuple(np.abs(np.array([ 133, 665]) - [0,ymax])), 50, color='g'))
    ax.add_artist(plt.Circle(tuple(np.abs(np.array([ 196, 890]) - [0,ymax])), 85, color='g'))
    ax.add_artist(plt.Circle(tuple(np.abs(np.array([ 110,1100]) - [0,ymax])), 125, color='g'))
    ax.add_artist(plt.Circle(tuple(np.abs(np.array([ 173,1291]) - [0,ymax])), 80, color='g'))
    ax.add_artist(plt.Circle(tuple(np.abs(np.array([ 418,1485]) - [0,ymax])), 65, color='g'))
    ax.add_artist(plt.Circle(tuple(np.abs(np.array([ 795,1155]) - [0,ymax])), 125, color='g'))
    ax.add_artist(plt.Circle(tuple(np.abs(np.array([ 795,1440]) - [0,ymax])), 70, color='g'))
    ax.add_artist(plt.Circle(tuple(np.abs(np.array([1075,1450]) - [0,ymax])), 40, color='g'))
    ax.add_artist(plt.Circle(tuple(np.abs(np.array([1416,1378]) - [0,ymax])), 50, color='g'))
    ax.add_artist(plt.Circle(tuple(np.abs(np.array([2150,1290]) - [0,ymax])), 105, color='g'))
    ax.add_artist(plt.Circle(tuple(np.abs(np.array([2529,1335]) - [0,ymax])), 220, color='g'))
    ax.add_artist(plt.Circle(tuple(np.abs(np.array([2240,1455]) - [0,ymax])), 165, color='g'))

    ax.plot(corners[:-2,0], corners[:-2,1], color='k')
    ax.plot(corners[-2:,0], corners[-2:,1], color='k')
elif video == 'CJ1':
    ymax, xmax, _ = (1520, 2704, 3)
    fence1 = np.array([ [289,  0],
                        [406,  1125],
                        [265,  1365],
                        [114,  ymax],
                        [265,  1365],
                        [389,  ymax]  ])

    fence2 = np.array([ [2349,  0],
                        [2225,  1030],
                        [1975,  ymax]  ])
    fence1 = np.abs(fence1 - [0, ymax])
    fence2 = np.abs(fence2 - [0, ymax])

    ax = plt.gca()
    ax.add_artist(plt.Circle(tuple(np.abs(np.array([307, 143]) - [0,ymax])), 64, color='g'))
    ax.add_artist(plt.Circle(tuple(np.abs(np.array([330, 460]) - [0,ymax])), 64, color='g'))
    ax.add_artist(plt.Circle(tuple(np.abs(np.array([396, 788]) - [0,ymax])), 39, color='g'))
    ax.add_artist(plt.Circle(tuple(np.abs(np.array([355,1178]) - [0,ymax])), 39, color='g'))
    ax.add_artist(plt.Circle(tuple(np.abs(np.array([232,1510]) - [0,ymax])), 124, color='g'))

    ax.plot(fence1[:-2,0], fence1[:-2,1], color='k')
    ax.plot(fence1[-2:,0], fence1[-2:,1], color='k')
    ax.plot(fence2[:,0], fence2[:,1], color='k')

elif video == 'CH1':
    ymax, xmax, _ = (1520, 2704, 3)
    fence1 = np.array([ [0,     1383],
                        [2517,  1147],
                        [xmax,  1082]  ])
    fence1 = np.abs(fence1 - [0, ymax])

    ax = plt.gca()
    ax.add_artist(plt.Circle(tuple(np.abs(np.array([730,152]) - [0,ymax])), 110, color='g'))
    ax.plot(fence1[:,0], fence1[:,1], color='k')

for j in range(N):
    plt.plot(np.convolve(sheep[S:F,j,0], np.ones((5,))/5, mode='valid'), np.abs(np.convolve(sheep[S:F,j,1], np.ones((5,))/5, mode='valid') - ymax), color = 'blue', lw = 2, alpha = 0.5)
plt.scatter(sheep[0,:,0], np.abs(sheep[0,:,1]-ymax), color='green', alpha = 0.7)
plt.scatter(sheep[-1,:,0], np.abs(sheep[-1,:,1]-ymax), color='red', alpha = 0.7)

if video == 'video2':
    plt.plot(np.convolve(blackSheep[S:F,0], np.ones((5,))/5, mode='valid'), np.abs(np.convolve(blackSheep[S:F,1], np.ones((5,))/5, mode='valid') - ymax), color = 'blue', lw = 2, alpha = 0.5)
    plt.scatter(blackSheep[0,0], np.abs(blackSheep[0,1]-ymax), color='green', alpha = 0.7)
    plt.scatter(blackSheep[-1,0], np.abs(blackSheep[-1,1]-ymax), color='red', alpha = 0.7)


plt.plot(np.convolve(quad[S:F,0], np.ones((5,))/5, mode='valid'), np.abs(np.convolve(quad[S:F,1], np.ones((5,))/5, mode='valid') - ymax), color = 'red', lw = 2)
plt.scatter(quad[0,0], np.abs(quad[0,1]-ymax), color='green', alpha = 0.7)
plt.scatter(quad[-1,0], np.abs(quad[-1,1]-ymax), color='red', alpha = 0.7)

plt.axes().set_aspect('equal')
for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
    label.set_fontsize(12)
plt.xlim(0, 2704)
plt.ylim(0, 1520)
plt.savefig(save+'traj'+str(len(sheep) - 1)+'.pdf', format='pdf', dpi = 300)
os.system('pdfcrop ' +save+'traj'+str(len(sheep) - 1)+'.pdf '+save+'traj'+str(len(sheep) - 1)+'.pdf')

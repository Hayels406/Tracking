import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import rc
import os

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


sheep = np.load(save+'loc50.npy')
vel = np.load(save+'vel50.npy')
save = sys.argv[1] == 'True'

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
sheep =  np.array(sheep.tolist())
while len(sheep[-1]) != len(sheep[0]):
	sheep = np.array(sheep[:-1])
N = np.shape(sheep)[1]
S = 0
F = 201

colors = [ cm.prism(x) for x in np.linspace(0., 1., N)]
for j in range(N):
    plt.plot(np.convolve(sheep[S:F,j,0], np.ones((5,))/5, mode='valid'), np.convolve(sheep[S:F,j,1], np.ones((5,))/5, mode='valid'), color = colors[j], lw = 1)

plt.axes().set_aspect('equal')
plt.gca().invert_yaxis()
for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
	label.set_fontsize(12)
if save == True:
	plt.savefig('./traj/traj'+str(len(sheep) - 1)+'.png')
else:
	plt.ion()
	plt.show()

for i in range(len(vel)):
	if np.shape(vel[i])[0] == 0:
		vel[i] = np.nan*np.zeros((N,2))
	while np.shape(vel[i])[0] < 144:
		vel[i] = np.append(vel[i], [[np.nan, np.nan]], axis = 0)
vel = np.array(vel.tolist())
speed = (vel**2).sum(axis = 2)

index = np.where(sheep[0,:,0] > 1800)[0]

speed_good = np.nanmean(speed[:,index])

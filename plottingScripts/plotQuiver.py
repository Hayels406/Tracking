import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import rc
from scipy.spatial.distance import cdist
from scipy.optimize import curve_fit

if os.getcwd().rfind('Uni') > 0:
    videoLocation = '/home/b1033128/Documents/throughFenceRL.mp4'
    save = '/home/b1033128/Documents/throughFenceRL/'
    dell = True
    brk = False
elif os.getcwd().rfind('hayley') > 0:
    videoLocation = '/users/hayleymoore/Documents/PhD/Tracking/throughFenceRL.mp4'
    save = '/users/hayleymoore/Documents/PhD/Tracking/throughFenceRL/'
else:
    videoLocation = '/data/b1033128/Videos/throughFenceRL.mp4'
    save = '/data/b1033128/Tracking/throughFenceRL/'
    dell = False


def logistic(x, a, b):
	return 1/(1+np.exp(-a*(x-b)))


sheep = np.load('locfull.npy')
sheep = np.array(map(np.array,  sheep))
vel = np.load('velfull.npy')
vel = np.array(map(np.array,  vel))

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
plt.figure(figsize = (7, 8))
sheep =  np.array(sheep)
while len(sheep[-1]) != len(sheep[0]):
	sheep = np.array(sheep[:-1])
T, N, _ = np.shape(sheep)


for i in range(T):
	if i < 2:
		vel[i] = np.zeros((N,2))
	elif i < 7:
		vel[i] = (sheep[i-1] - sheep[max(i-6,  0)])/np.diff(i-1,  max(i-6, 0))
	while np.shape(vel[i])[0] < 144:
		vel[i] = np.append(vel[i], [[0, 0]], axis = 0)
vel = np.array(vel.tolist())


index = np.where(sheep[0,:,0] > 1800)[0]

#For frame 10
for frameID in range(T):

	ref_vel = np.mean(vel[frameID,index,:], axis = 0)
	norm_ref_vel = ref_vel/np.sqrt((ref_vel**2).sum())
	ref_loc = np.mean(sheep[frameID,index,:], axis = 0)
	norm_vel =  vel[frameID]/(np.transpose(np.tile(np.sqrt((vel[frameID]**2).sum(axis =  1)+1e-8),2).reshape(2,N)))
	
	plt.clf()
	plt.quiver(sheep[frameID,:,0],  sheep[frameID,:,1], norm_vel[:,0],  -norm_vel[:,1])
	plt.axes().set_aspect('equal')
	plt.gca().invert_yaxis()
	plt.savefig(save+'quiver'+str(frameID).zfill(4), bbox_inches='tight')
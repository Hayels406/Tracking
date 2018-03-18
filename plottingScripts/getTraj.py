import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import rc

sheep = np.load('locfull.npy')
vel = np.load('velfull.npy')
save = sys.argv[1] == 'True'

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
sheep =  np.array(sheep.tolist())
while len(sheep[-1]) != len(sheep[0]):
	sheep = np.array(sheep[:-1])
N = np.shape(sheep)[1]
S = 30
F = -1

colors = [ cm.prism(x) for x in np.linspace(0., 1., N)]
for j in range(N):
    plt.plot(sheep[S:F,j,0], sheep[S:F,j,1], color = colors[j], lw = 1)

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

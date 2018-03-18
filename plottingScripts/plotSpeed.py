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
plt.figure(figsize = (7, 8))
sheep =  np.array(sheep)
while len(sheep[-1]) != len(sheep[0]):
	sheep = np.array(sheep[:-1])
N = np.shape(sheep)[1]


for i in range(len(vel)):
	if np.shape(vel[i])[0] == 0:
		vel[i] = np.nan*np.zeros((N,2))
	while np.shape(vel[i])[0] < 144:
		vel[i] = np.append(vel[i], [[np.nan, np.nan]], axis = 0)
vel = np.array(vel.tolist())
speed = (vel**2).sum(axis = 2)

index = np.where(sheep[0,:,0] > 1800)[0]

speed_good = np.nanmean(speed[:,index])

Z = [[0,0],[0,0]]
levels = range(0,144,1)
CS3 = plt.contourf(Z, levels, cmap='magma')
colors = [ cm.magma(x) for x in np.linspace(0., 1., 144)]
for i in range(144):
	plt.plot(range(7,41),speed[7:,i], color=colors[i])
for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
	label.set_fontsize(12)
plt.xlim(0,40)
plt.colorbar(CS3)
if save == True:
	plt.savefig('./speed/speed'+str(len(sheep) - 1)+'.png')
else:
	plt.ion()
	plt.show()



#plt.plot(np.nanmean(speed[7:],axis=0))
#for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
#	label.set_fontsize(12)
#if save == True:
#	plt.savefig('./speed/avSpeed'+str(len(sheep) - 1)+'.png')
#else:
#	plt.ion()
#	plt.show()

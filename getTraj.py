import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import rc


def plotTraj(sheep,  save=False):
	rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
	rc('text', usetex=True)
	sheep =  np.array(sheep)
	while len(sheep[-1]) != len(sheep[0]):
		sheep = np.array(sheep[:-1])
	N = np.shape(sheep)[1]

	colors = [ cm.prism(x) for x in np.linspace(0., 1., N)]
	for j in range(N):
	    plt.plot(sheep[:,j,0], sheep[:,j,1], color = colors[j])

	plt.axes().set_aspect('equal')
	plt.gca().invert_yaxis()
	for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
		label.set_fontsize(16)
	if save == True:
		plt.savefig('./traj/traj'+str(len(sheep) - 1)+'.png')
	else:
		plt.show()

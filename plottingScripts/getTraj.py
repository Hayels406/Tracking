import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import rc
from glob import glob
import os

video = sys.argv[1]
videoLocation = '/data/b1033128/Videos/'+video
save = '/data/b1033128/Tracking/'+video[:-4] +'/'



data = glob(save+'Final-loc*')[-1]
sheep = np.load(data)
quad = np.load(glob(save+'Final-quad*')[-1])
sheep =  map(np.array,  sheep)
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

colors = [ cm.prism(x) for x in np.linspace(0., 1., N)]
for j in range(N):
    plt.plot(np.convolve(sheep[S:F,j,0], np.ones((5,))/5, mode='valid'), np.convolve(sheep[S:F,j,1], np.ones((5,))/5, mode='valid'), color = colors[j], lw = 1)

plt.axes().set_aspect('equal')
plt.gca().invert_yaxis()
for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
    label.set_fontsize(12)
plt.savefig(save+'traj'+str(len(sheep) - 1)+'.pdf', format='pdf')

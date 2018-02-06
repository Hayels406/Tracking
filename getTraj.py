import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

plt.close()
sheep = np.load('locfull.npy')

colors = [ cm.prism(x) for x in np.linspace(0., 1., 144)]
for j in range(144):
    plt.plot(sheep[:,j,0], sheep[:,j,1], color = colors[j])

plt.axes().set_aspect('equal')
plt.gca().invert_yaxis()
plt.savefig('./traj/traj'+str(len(sheep) - 1)+'.png')

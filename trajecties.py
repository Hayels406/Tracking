import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

locations = np.load('locfull.npy')

sheep = np.zeros((7, 141, 2))
sheep[0] = locations[0]

for t in range(6):
    print t
    C = cdist(sheep[t], locations[t+1])
    _, assigment = linear_sum_assignment(C)
    sheep[t+1] = locations[t+1][assigment]

colors = [ cm.prism(x) for x in np.linspace(0., 1., 141)]
for j in range(141):
    plt.plot(sheep[:,j,0], sheep[:,j,1], color = colors[j])
plt.gca().invert_yaxis()
plt.axes().set_aspect('equal')
plt.show()

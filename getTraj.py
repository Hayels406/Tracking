import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
plt.close()
locations = np.load('locfull.npy')
endTime = np.shape(locations)[0]
endTime -= 1
print endTime

sheep = np.zeros((endTime, 144, 2))
sheep[0] = np.array(locations[0])

for t in range(endTime-1):
    print t
    C = cdist(sheep[t], locations[t+1])
    _, assignment = linear_sum_assignment(C)
    sheep[t+1] = np.array(locations[t+1])[assignment]

colors = [ cm.prism(x) for x in np.linspace(0., 1., 144)]
for j in range(144):
    plt.plot(sheep[:,j,0], sheep[:,j,1], color = colors[j])

plt.axes().set_aspect('equal')
plt.gca().invert_yaxis()
plt.savefig('./traj/traj'+str(endTime)+'.png')

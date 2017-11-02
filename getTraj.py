import numpy as np
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt

locations = np.load('loc.npy')

print locations

for i in range(40):
    print np.shape(locations[i])

traj = [locations[0]]

print traj
t = []

for time in range(1):#range(len(locations)):
    if np.shape(locations[time]) < np.shape(locations[time+1]):
        print 'every element in 0 needs a partner in 1'
        tree = KDTree(locations[time+1])
        dist, idx = tree.query(locations[time], 1)
        for i in range(len(dist)):
            if dist[i] < 1000000:
                t += [locations[time+1][idx[i][0]]]
            else:
                t += [[np.nan, np.nan]]
    else:
        print 'every element in 1 needs a partner in 0'
        tree = KDTree(locations[time])
        dist, idx = tree.query(locations[time+1], 1)
        for i in range(len(dist)):
            if dist[i] < 1000000:
                t += [locations[time+1][idx[i][0]]]
            else:
                t += [[np.nan, np.nan]]

    traj += [t]

traj = np.array(traj)
plt.scatter(np.array(locations[1])[:,0], np.array(locations[1])[:,1], alpha = 0.5)
for i in range(np.shape(traj)[1]):
    plt.plot(traj[:,i,0], traj[:,i,1], color = 'k')
plt.scatter(traj[1,:,0], traj[1,:,1], marker = 'x')

plt.show()

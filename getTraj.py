import numpy as np
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt

locations = np.load('loc100.npy')

print locations

for i in range(99):
    print np.shape(locations[i])

traj = []


for time in range(1):#range(len(locations)):
    t = []
    t_end = []
    if np.shape(locations[time]) == np.shape(locations[time+1]):
        print 'every element in 0 has a partner in 1'
        tree = KDTree(locations[time+1])
        dist, idx = tree.query(locations[time], 1)
        for i in range(len(dist)):
            if dist[i] < 50:
                t += [locations[time]]
    elif np.shape(locations[time]) < np.shape(locations[time+1]):
        print 'every element in 0 needs a partner in 1'
        tree = KDTree(locations[time+1])
        dist, idx = tree.query(locations[time], 1)
        for i in range(len(dist)):
            if dist[i] < 50:
                t += [locations[time]]

    else:
        print 'every element in 1 needs a partner in 0'
        tree = KDTree(locations[time+1])
        dist, idx = tree.query(locations[time], 1)
        for i in range(len(dist)):
            if dist[i] < 1000000:
                t += [locations[time+1][idx[i][0]]]
            else:
                t += [[np.nan, np.nan]]

    if np.shape(traj)[1] == np.shape(t+t_end)[0]:
        traj += [t + t_end]
    elif np.shape(traj)[1] < np.shape(t+t_end)[0]:
        traj += [t + t_end]
    elif np.shape(traj)[1] > np.shape(t+t_end)[0]:
        traj += [t + t_end]
    else:
        break

traj = np.array(traj)
plt.scatter(np.array(locations[0])[:,0], np.array(locations[0])[:,1], alpha = 0.5, marker='+')
plt.scatter(np.array(locations[1])[:,0], np.array(locations[1])[:,1], alpha = 0.5)
for i in range(np.shape(traj)[1]):
    plt.plot(traj[:,i,0], traj[:,i,1], color = 'k')
plt.ylim(ymax=800,ymin=2000)
plt.show()

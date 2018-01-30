import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

time1 = np.array([[10, 10], [10, 30]])
time2 = np.array([[40, 30], [40, 10]])
vel = np.array([[30, 20], [30, -20]])

prediction_Objects = time1 + vel
x_r =  np.arange(0,  50)
y_r =  np.arange(0,  50)
xx, yy =  np.meshgrid(x_r, y_r)
z = []
s_x, s_y = [3, 3]
for point in prediction_Objects:
    m_x = point[0]
    m_y = point[1]
    z += [(1/(2*np.pi*s_x*s_y))*np.exp(-((xx-m_x)**2/(2*s_x**2))-((yy-m_y)**2/(2*s_y**2)))]


cropX, cropY = (0,0)
den2 = []
for predictionArea in z:
    den = []
    for point in np.floor(time2):
        den += [np.transpose(predictionArea)[int(point[0] - cropX), int(point[1] - cropY)]]
    den2 += [(np.max(den) - den)/np.max(den)]
_, assignment = linear_sum_assignment(den2)
allocation = time2[assignment]

z = np.array(z).sum(axis = 0)
plt.figure(figsize = (10, 7))
plt.imshow(z, cmap = 'gray')
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=16)
plt.scatter(time1[0,0], time1[0,1], color = 'red', label = 'Object 1')
plt.scatter(time2[0,0], time2[0,1], color = 'red')
plt.scatter(time1[1,0], time1[1,1], color = 'blue', label = 'Object 2')
plt.scatter(time2[1,0], time2[1,1], color = 'blue')
plt.scatter(allocation[0,0], allocation[0,1], marker = 'x', color = 'orange', label = 'Allocated 1')
plt.scatter(allocation[1,0], allocation[1,1], marker = 'x', color = 'green', label = 'Allocated 2')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5, fontsize = 18)
plt.title(r'Crossing', fontsize = 20)
for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
    label.set_fontsize(16)
plt.savefig('correct_allocation_when_crossing.png')


plt.close()


###################################################################################################################
time1 = np.array([[10, 30], [40, 10]])
time2 = np.array([[40, 30], [10, 10]])
vel = np.array([[30, 0], [-30, 0]])

prediction_Objects = time1 + vel
x_r =  np.arange(0,  50)
y_r =  np.arange(0,  50)
xx, yy =  np.meshgrid(x_r, y_r)
z = []
s_x, s_y = [3, 3]
for point in prediction_Objects:
    m_x = point[0]
    m_y = point[1]
    z += [(1/(2*np.pi*s_x*s_y))*np.exp(-((xx-m_x)**2/(2*s_x**2))-((yy-m_y)**2/(2*s_y**2)))]


cropX, cropY = (0,0)
den2 = []
for predictionArea in z:
    den = []
    for point in np.floor(time2):
        den += [np.transpose(predictionArea)[int(point[0] - cropX), int(point[1] - cropY)]]
    den2 += [(np.max(den) - den)/np.max(den)]
_, assignment = linear_sum_assignment(den2)
allocation = time2[assignment]

z = np.array(z).sum(axis = 0)
plt.figure(figsize = (10, 7))
plt.imshow(z, cmap = 'gray')
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=16)
plt.scatter(time1[0,0], time1[0,1], color = 'red', label = 'Object 1')
plt.scatter(time2[0,0], time2[0,1], color = 'red')
plt.scatter(time1[1,0], time1[1,1], color = 'blue', label = 'Object 2')
plt.scatter(time2[1,0], time2[1,1], color = 'blue')
plt.scatter(allocation[0,0], allocation[0,1], marker = 'x', color = 'orange', label = 'Allocated 1')
plt.scatter(allocation[1,0], allocation[1,1], marker = 'x', color = 'green', label = 'Allocated 2')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5, fontsize = 18)
plt.title(r'Passing', fontsize = 20)
for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
    label.set_fontsize(16)
plt.savefig('correct_allocation_when_passing.png', bbox_inches = 0)


plt.close()


###################################################################################################################
time1 = np.array([[25, 30], [25, 10]])
time2 = np.array([[25, 10], [25, 30]])
vel = np.array([[0, -20], [0, 20]])

prediction_Objects = time1 + vel
x_r =  np.arange(0,  50)
y_r =  np.arange(0,  50)
xx, yy =  np.meshgrid(x_r, y_r)
z = []
s_x, s_y = [3, 3]
for point in prediction_Objects:
    m_x = point[0]
    m_y = point[1]
    z += [(1/(2*np.pi*s_x*s_y))*np.exp(-((xx-m_x)**2/(2*s_x**2))-((yy-m_y)**2/(2*s_y**2)))]


cropX, cropY = (0,0)
den2 = []
for predictionArea in z:
    den = []
    for point in np.floor(time2):
        den += [np.transpose(predictionArea)[int(point[0] - cropX), int(point[1] - cropY)]]
    den2 += [(np.max(den) - den)/np.max(den)]
_, assignment = linear_sum_assignment(den2)
allocation = time2[assignment]

z = np.array(z).sum(axis = 0)
plt.figure(figsize = (10, 7))
plt.imshow(z, cmap = 'gray')
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=16)
plt.scatter(time1[0,0], time1[0,1], color = 'red', s = 400., alpha = 0.5)
plt.scatter(time2[0,0], time2[0,1], color = 'red', label = 'Object 1')
plt.scatter(time1[1,0], time1[1,1], color = 'blue', s = 400., alpha = 0.5)
plt.scatter(time2[1,0], time2[1,1], color = 'blue', label = 'Object 2')
plt.scatter(allocation[0,0], allocation[0,1], marker = 'x', color = 'orange', label = 'Allocated 1')
plt.scatter(allocation[1,0], allocation[1,1], marker = 'x', color = 'green', label = 'Allocated 2')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5, fontsize = 18)
plt.title(r'Swapping', fontsize = 20)
for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
    label.set_fontsize(16)
plt.savefig('correct_allocation_when_swapping.png')


plt.close()
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


prediction_Objects = np.array([[20., 20.], [60., 60.]])
x_r =  np.arange(0,  1000)
y_r =  np.arange(0,  1000)
xx, yy =  np.meshgrid(x_r, y_r)
z = []
s_x, s_y = [3, 3]
for point in prediction_Objects:
    m_x = point[0]
    m_y = point[1]
    z += [(1/(2*np.pi*s_x*s_y))*np.exp(-((xx-m_x)**2/(2*s_x**2))-((yy-m_y)**2/(2*s_y**2)))]

z = np.array(z).sum(axis = 0)
z = z/np.max(z)

plt.figure(figsize = (14, 14))

plt.subplot(3,2,1)
plt.imshow(z, cmap = 'magma')
plt.ylim(100, 0)
plt.xlim(0,100)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=16)
for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
    label.set_fontsize(16)

plt.subplot(3,2,2)
plt.hist(z[z > 0])
plt.ylim(0,100)
plt.title('Bar 1 has height of '+ str(np.histogram(z[z>0])[0][0]), fontsize = 18)
for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
    label.set_fontsize(16)


prediction_Objects = np.array([[46.5, 46.5], [52.5, 52.5]])
x_r =  np.arange(0,  1000)
y_r =  np.arange(0,  1000)
xx, yy =  np.meshgrid(x_r, y_r)
z = []
s_x, s_y = [3, 3]
for point in prediction_Objects:
    m_x = point[0]
    m_y = point[1]
    z += [(1/(2*np.pi*s_x*s_y))*np.exp(-((xx-m_x)**2/(2*s_x**2))-((yy-m_y)**2/(2*s_y**2)))]

z = np.array(z).sum(axis = 0)
z = z/np.max(z)

plt.subplot(3,2,3)
plt.imshow(z, cmap = 'magma')
plt.ylim(100, 0)
plt.xlim(0,100)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=16)
for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
    label.set_fontsize(16)

plt.subplot(3,2,4)
plt.hist(z[z > 0])
plt.ylim(0,100)
plt.title('Bar 1 has height of '+ str(np.histogram(z[z>0])[0][0]), fontsize = 18)
for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
    label.set_fontsize(16)



prediction_Objects = np.array([[46.5, 46.5], [52.5, 52.5], [20., 20.]])
x_r =  np.arange(0,  1000)
y_r =  np.arange(0,  1000)
xx, yy =  np.meshgrid(x_r, y_r)
z = []
s_x, s_y = [3, 3]
for point in prediction_Objects:
    m_x = point[0]
    m_y = point[1]
    z += [(1/(2*np.pi*s_x*s_y))*np.exp(-((xx-m_x)**2/(2*s_x**2))-((yy-m_y)**2/(2*s_y**2)))]

z = np.array(z).sum(axis = 0)
z = z/np.max(z)

plt.subplot(3,2,5)
plt.imshow(z, cmap = 'magma')
plt.ylim(100, 0)
plt.xlim(0,100)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=16)
for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
    label.set_fontsize(16)

plt.subplot(3,2,6)
plt.hist(z[z > 0])
plt.ylim(0,100)
plt.title('Bar 1 has height of '+ str(np.histogram(z[z>0])[0][0]), fontsize = 18)
for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
    label.set_fontsize(16)


plt.savefig('canonicalHist.png', bbox_inches = 0)
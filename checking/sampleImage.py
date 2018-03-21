import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from skimage import measure
import cv2
from trackingFunctions import kmeansClustering
from trackingFunctions import iris

plt.close()
plt.figure()
ax = plt.gca()
objectLocations = [[620., 250.], [720., 250.], [250., 600.], [700., 700.], [780., 840.]]
angles = np.array([0, 0, -0.4, 0.5, 0.5])
x_r =  np.arange(0,  1000)
y_r =  np.arange(0,  1000)
xx, yy =  np.meshgrid(x_r, y_r)
z =  xx*0.
s_x, s_y= [30,  50]
for i in range(len(objectLocations)):
	point = objectLocations[i]
	rho = angles[i]
	m_x = point[0]
	m_y = point[1]
	blob = (1/(2*np.pi*s_x*s_y*np.sqrt(1-rho**2)))*np.exp((-1/(2*(1-rho**2)))*((xx-m_x)**2/s_x**2 + (yy-m_y)**2/s_y**2 - 2*rho*(xx-m_x)*(yy-m_y)/(s_x*s_y)))
	z += blob/np.max(blob)
z = 255*z/np.max(z)
plt.imshow(z, cmap = 'gray')
plt.gca().set_aspect('equal')
plt.axis('off')
plt.savefig('sampleImage/original.png')

binary = z > 5
plt.imshow(binary, cmap = 'gray')
plt.gca().set_aspect('equal')
plt.axis('off')
plt.savefig('sampleImage/binary.png')

labels = measure.label(binary, neighbors=8, background=0)
cmap = plt.get_cmap('rainbow')
cmap.set_under('black')
plt.gca().set_aspect('equal')
plt.axis('off')
plt.imshow(labels, vmin =  0.5, cmap = cmap)
plt.gca().set_aspect('equal')
plt.axis('off')
plt.savefig('sampleImage/labels.png')

plt.imshow(binary, cmap = 'gray')
objectLocations = []
for label in np.unique(labels):
	if label == 0:
		continue
	labelMask = np.zeros(binary.shape, dtype="uint8")
	labelMask[labels == label] = 1

	numPixels = (labelMask > 0).sum()
	print numPixels
	cnts = cv2.findContours(labelMask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1][0]
	((cX, cY), radius) = cv2.minEnclosingCircle(cnts)
	if numPixels < 40000:
		objectLocations += [[cX, cY]]
		plt.scatter(cX, cY, marker = 's')
	else:
		x,y,w,h = cv2.boundingRect(cnts)
		miniImage = z[y: y + h, x: x + w]
		new_objects_K = np.array(kmeansClustering(miniImage, numPixels, x, y, previous = 2))
		new_objects = np.array(iris(miniImage, x, y))
		plt.scatter(new_objects_K[:,0], new_objects_K[:,1],color='red')
		plt.scatter(new_objects[:,0], new_objects[:,1], marker='^')
		objectLocations += new_objects.tolist()
plt.savefig('sampleImage/detected.png')

plt.close()
plt.imshow(binary, cmap = 'gray')
plt.scatter(np.array(objectLocations)[:,0], np.array(objectLocations)[:,1])
plt.gca().set_aspect('equal')
plt.axis('off')
plt.savefig('sampleImage/final.png', bbox_inches='tight')
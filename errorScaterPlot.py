import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def errorScatterNum(locations, pred):
	points1 = np.array(locations[-1])
	points2 = np.array(locations[-2])

	colors = [cm.prism(x) for x in np.linspace(0, 1, 20).tolist()*8]
	for j in range(len(points1)):
		plt.scatter(points2[j,0], points2[j,1], marker=r"$ {} $".format(j), color = 'red', s=200)
		plt.scatter(points1[j,0], points1[j,1], marker=r"$ {} $".format(j), color = 'forestGreen',s=200)
		plt.scatter(pred[j,0], pred[j,1], marker=r"$ {} $".format(j), color = 'purple',s=200)
	plt.gca().invert_yaxis()
	plt.show()

def errorScatterCol(locations, pred):
	points1 = np.array(locations[-1])
	points2 = np.array(locations[-2])

	colors = [cm.prism(x) for x in np.linspace(0, 1, 20).tolist()*8]
	for j in range(len(points1)):
		plt.scatter(points2[j,0], points2[j,1], marker='s', color = colors[j])
		plt.scatter(points1[j,0], points1[j,1], marker='o', color = colors[j])
		plt.scatter(pred[j,0], pred[j,1], marker='p', color = colors[j])
	plt.gca().invert_yaxis()
	plt.show()



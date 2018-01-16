import numpy as np
import matplotlib.pyplot as plt

def errorScatter(locations):
	points1 = np.array(locations[-1])
	points2 = np.array(locations[-2])

	plt.scatter(points2[:,0], points2[:,1])
	plt.scatter(points1[:,0], points1[:,1], marker = 'x')
	plt.gca().invert_yaxis()
	plt.show()

import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

matplotlib.rc('font', family='serif')
matplotlib.rc('text', usetex=True)

if os.getcwd().rfind('hayley') > 0:
    videoLocation = '/users/hayleymoore/Documents/PhD/Tracking/throughFenceRL.mp4'
    save = '/users/hayleymoore/Documents/PhD/Tracking/throughFenceRL/'
elif os.getcwd().rfind('Uni') > 0:
    videoLocation = '/home/b1033128/Documents/throughFenceRL.mp4'
    save = '/home/b1033128/Documents/throughFenceRL/'
    dell = True
    brk = False
else:#Kiel
    videoLocation = '/data/b1033128/Videos/throughFenceRL.mp4'
    save = '/data/b1033128/Tracking/throughFenceRL/'
    dell = False


grey = np.load(save + 'grey/grey.npy')/255.
gCor = np.load(save + 'grey/gammaCorrection.npy')
blue = np.load(save + 'grey/blue.npy')/255.
gr = np.load(save + 'grey/GR.npy')
gr = gr - np.min(gr)
gr = gr/np.max(gr)
prod = np.load(save + 'grey/product.npy')
mahal = np.load(save + 'grey/mahalanobis.npy')
mahal = mahal - np.min(mahal)
mahal = mahal/np.max(mahal)


methods =  [grey, blue, gr, gCor, prod, mahal]
labels = ['Weighted Average', 'Blue Channel', 'Green - Red', 'Gamma Correction', 'Product', 'Mahalanobis Distance']

for i in range(len(methods)):
    kernel = gaussian_kde(methods[i])
    X = np.linspace(0,1,1000)
    dens = kernel(X)
    plt.plot(X, dens, label = labels[i])


plt.legend(loc = 'best')
plt.yscale('log')
plt.ylim(10**-4, 10**3)
plt.savefig('kernelDens.pdf', format = 'pdf', dpi = 300)

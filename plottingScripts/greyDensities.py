import numpy as np
import os
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture as gm
from scipy.stats import norm

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
prod = np.load(save + 'grey/product.npy')/(255.**3)
mahal = np.load(save + 'grey/mahalanobis.npy')
mahal = mahal - np.min(mahal)
mahal = mahal/np.max(mahal)


methods =  [gr, gCor, prod, grey, blue, mahal]
labels = ['Green - Red', 'Gamma Correction', 'Product', 'Weighted Average', 'Blue Channel', 'Mahalanobis Distance']

plt.figure(figsize=(6,4))
for i in range(3):
    kernel = gaussian_kde(methods[i])
    X = np.linspace(0,1,1000)
    dens = kernel(X)
    plt.plot(X, dens, label = labels[i])


plt.legend(loc = 'best')
plt.yscale('log')
plt.ylim(10**-4, 10**2)
plt.savefig(save+'kernelDens1.pdf', format = 'pdf', dpi = 300)
os.system('pdfcrop ' +save+'kernelDens1.pdf '+save+'kernelDens1.pdf')

plt.close('all')
plt.figure(figsize=(6,4))
for i in range(3,5):
    print labels[i]
    kernel = gaussian_kde(methods[i])
    X = np.linspace(0,1,1000)
    dens = kernel(X)
    plt.plot(X, dens, label = labels[i])


plt.legend(loc = 'best')
plt.yscale('log')
plt.ylim(10**-4, 10**2)
plt.savefig(save+'kernelDens2.pdf', format = 'pdf', dpi = 300)
os.system('pdfcrop ' +save+'kernelDens2.pdf '+save+'kernelDens2.pdf')


plt.close('all')
plt.figure(figsize=(6,4))
for i in range(3,6):
    print labels[i]
    kernel = gaussian_kde(methods[i])
    X = np.linspace(0,1,1000)
    dens = kernel(X)
    plt.plot(X, dens, label = labels[i])


plt.legend(loc = 'best')
plt.yscale('log')
plt.ylim(10**-4, 10**2)
plt.savefig(save+'kernelDens3.pdf', format = 'pdf', dpi = 300)
os.system('pdfcrop ' +save+'kernelDens3.pdf '+save+'kernelDens3.pdf')



N = 3
plt.close('all')
plt.figure(figsize=(6,4))
plt.hist(gCor, bins = 30, normed = True,edgecolor='k',color='white')
plt.yscale('log')
plt.ylim(10**-4, 10**2)
mm = gm(n_components=N, covariance_type='full', weights_init=[0.95, 0.025, 0.025], means_init=[[0.05], [0.5], [0.9]]).fit(gCor.reshape(-1,1))
x = np.arange(0, 1, .01)
s = []
for i in range(N):
    rv = norm(loc = mm.means_[i], scale = np.sqrt(mm.covariances_[i]))
    plt.plot(x, mm.weights_[i]*rv.pdf(x).flatten(), label = 'N('+str(np.round(mm.means_[i], 3)[0])+', '+str(np.round(np.sqrt(mm.covariances_[i])[0,0], 3))+')')
    s += [(mm.weights_[i]*rv.pdf(x).flatten()).tolist()]
    if i == 0:
        plt.axvline(rv.interval(0.95)[1], color='C0', linestyle='--', label='95\%  Percentile')
    elif i == N-1:
        plt.axvline(rv.interval(0.95)[0], color='C2', linestyle='--', label='5\%  Percentile')
plt.plot(x, np.array(s).sum(axis =0), label = 'Gaussian Mixture Model')
plt.legend(loc = 'upper right')
plt.savefig(save+'gaussianMixture.pdf', format = 'pdf', dpi=300)
os.system('pdfcrop ' +save+'gaussianMixture.pdf '+save+'gaussianMixture.pdf')

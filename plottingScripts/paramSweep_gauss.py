import numpy as np
import matplotlib.pyplot as plt
from glob import glob

videoLocation = '/data/b1033128/Videos/CaseH2.mov'
save = '/data/b1033128/Tracking/CaseH2/'


plt.close('all')
plt.ion()
l = [3,5,7,9,11,13]
data = np.zeros([6, len(np.arange(1., 10, 0.5))])
for lG in l:
    i_l = np.where(np.array(l) == lG)[0][0]
    print i_l
    for sigmaG in np.arange(1., 10, 0.5):
        i_s  = np.where(np.arange(1, 10, 0.5) == sigmaG)[0][0]

        #setting up file directories
        sig = str(round(sigmaG,2)*10)
        l1 = sig[0]
        l2 = sig[1:]

        file = save+'ParamSweep/gaussianL'+str(lG)+'/sigma' +l1+'-'+l2+'/sheepDetected.csv'
        if len(glob(file)) > 0:
            data[i_l, i_s] = np.loadtxt(file)
        else:
            data[i_l, i_s] = 0


plt.imshow(data == 44)
ax = plt.gca()
ax.invert_yaxis()
ax.set_yticklabels(['', '3', '5', '7', '9', '11', '13'])
plt.xticks(np.arange(0, 18, 1.), np.arange(1,10,0.5))
plt.xlabel('$\sigma_G$')
plt.ylabel('$\ell_G$')
plt.savefig(save+'ParamSweep/gaussParamSweep.pdf', format='pdf')
np.savetxt(save+'pgfData/gaussIdentifies.txt', np.append(np.append(np.array(sorted(l*18)).reshape(18*6, 1), np.tile(np.arange(1, 10, 0.5), 6).reshape(18*6,1), axis = 1), (data == 44).flatten().reshape(18*6,1), axis = 1), fmt='%1.1e')

plt.close('all')
plt.plot(l,(data==44).sum(axis=1)*1./np.shape(data)[1])
plt.scatter(l,(data==44).sum(axis=1)*1./np.shape(data)[1], color='red')
plt.xlabel('$\ell_G$')
plt.ylabel('Proportion of $\sigma_G$ values that identify the correct number of sheep')
plt.xlim(1,15)
plt.savefig(save+'ParamSweep/proportionGauss.pdf', format = 'pdf')
np.savetxt(save+'pgfData/proportionGauss.txt', np.append(np.arange(3, 15, 2).reshape(6,1), ((data==44).sum(axis=1)*1./np.shape(data)[1]).reshape(6,1), axis = 1))

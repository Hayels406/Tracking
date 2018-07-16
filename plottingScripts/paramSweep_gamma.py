import numpy as np
import matplotlib.pyplot as plt
from glob import glob

videoLocation = '/data/b1033128/Videos/CaseH2.mov'
save = '/data/b1033128/Tracking/CaseH2/'


plt.close('all')

data = np.zeros([len(np.arange(1, 2.425, 0.025)), 6, 5])
prop = np.zeros([len(np.arange(1, 2.525, 0.025))])
for gamma in np.arange(1, 2.425, 0.025):
    i_g = np.where(np.arange(1, 2.425, 0.025) == gamma)[0][0]
    print i_g
    for tlPercent in [0.95, 0.99, 0.9925, 0.995, 0.9975, 0.999]:
        i_l  = np.where(np.array([0.95, 0.9925, 0.99, 0.995, 0.9975, 0.999]) == tlPercent)[0][0]
        for tuPercent in [0.01, 0.1, 0.2, 0.3, 0.4]:
            i_u = np.where(np.array([0.01, 0.1, 0.2, 0.3, 0.4]) == tuPercent)[0][0]
            #setting up file directories
            g = str(int(round(gamma*100)))
            g1 = g[0]
            g2 = g[1:]

            L = str(round(tlPercent,3)*100)
            l1 = L[:2]
            l2 = L[3:]

            u = str(round(tuPercent,3)*100)
            u1 = u[:2]
            u2 = u[3:]

            file = save+'ParamSweep/gamma'+g1+'-'+g2+'/lowerPercentile' +l1+'-'+l2+'/upperPercentile'+u1+'-'+u2+'/sheepDetected.csv'
            if len(glob(file)) > 0:
                data[i_g, i_l, i_u] = np.loadtxt(file)
            else:
                data[i_g, i_l, i_u] = 0

    if len(save+'ParamSweep/gamma'+g1+'-'+g2) > 0:
        plt.imshow(data[i_g, ::-1, :] == 44, cmap = 'ocean_r')
        ax = plt.gca()
        ax.set_yticklabels(['', '99.9', '99.75', '99.5', '99.25', '99.0', '95.0'])
        ax.set_xticklabels(['', '1.', '10', '20', '30', '40'])
        plt.xlabel('$p_{t_u}$')
        plt.ylabel('$p_{t_l}$')
        plt.savefig(save+'ParamSweep/gamma'+g1+'-'+g2+'/thresholds.pdf', format = 'pdf')
        np.savetxt(save+'ParamSweep/gamma'+g1+'-'+g2+'/proportionDetected.txt', [np.sum(data[i_g,:,:] == 44)*1./np.sum(data[i_g,:,:] > 0)])
    prop[i_g] = np.sum(data[i_g,:,:] == 44)*1./np.sum(data[i_g,:,:] > 0)

plt.close('all')
plt.plot(np.arange(1, 2.525, 0.025), prop, 'ro')
plt.plot(np.arange(1, 2.525, 0.025), prop)
plt.ylim(ymax=1)
plt.xlabel('$\gamma$')
plt.ylabel('Proportion')
plt.savefig(save+'ParamSweep/proportion.pdf', format = 'pdf')
np.savetxt(save+'ParamSweep/proportion.txt', np.append(np.arange(1, 2.525, 0.025).reshape(61,1), np.array([prop]).T, axis = 1))

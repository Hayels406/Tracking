import numpy as np
import matplotlib.pyplot as plt
from glob import glob

videoLocation = '/data/b1033128/Videos/CaseH2.mov'
save = '/data/b1033128/Tracking/CaseH2/'


plt.close('all')
plt.ion()
l = [1,3,5,7,9]
data = np.zeros([5])
for lG in l:
    i_l = np.where(np.array(l) == lG)[0][0]
    print i_l
    file = save+'ParamSweep/maximumL'+str(lG)+'/sheepDetected.csv'
    if len(glob(file)) > 0:
        data[i_l] = np.loadtxt(file)
    else:
        data[i_l] = 0


plt.plot(l,data)
plt.scatter(l,data, color='red')
plt.xlabel('$\ell_M$')
plt.ylabel('Number of Sheep Detected')
plt.savefig(save+'ParamSweep/proportionMax.pdf', format = 'pdf')
np.savetxt(save+'pgfData/numberDetectedMax.txt', np.append(np.array(l).reshape(5,1), data.reshape(5,1), axis = 1))

import numpy as np

script = 'paramSweepVideo_gaussian_check.py '

for lG in [3,5,7,9,11,13]:
    for sigma in np.arange(1, 10, 0.5):
        print 'python ' + script + str(lG) + ' ' + str(sigma)

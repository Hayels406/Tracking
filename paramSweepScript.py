import numpy as np
import sys
import os

for gamma in [1.0, 1.25, 1.5, 1.75, 2., 2.5, 3., 4.]:
    for lower in [0.95, 0.995, 0.9999]:
        for upper in [0.01, 0.1, 0.2]:
            sys.argv = ['paramSweepVideo.py', str(gamma), str(lower), str(upper)]
            execfile('paramSweepVideo.py')

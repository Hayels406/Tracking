import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.spatial.distance import mahalanobis
import cv2






sys.argv = ['getCoordsVideo.py', 10, 'N']
execfile('getCoordsVideo.py')


greyScale = np.copy(img)
plt.imshow(ndimage.maximum_filter(greyScale, size=3), cmap = 'gray')
plt.gca().set_axis_off()
plt.xlim(225,425)
plt.ylim(225,25)
plt.savefig(save+'maximumFilter/s2.pdf', bbox_inches='tight', pad_inches=0, format = 'pdf', dpi=300)
plt.close('all')


plt.imshow(ndimage.maximum_filter(greyScale, size=5), cmap = 'gray')
plt.gca().set_axis_off()
plt.xlim(225,425)
plt.ylim(225,25)
plt.savefig(save+'maximumFilter/s5.pdf', bbox_inches='tight', pad_inches=0, format = 'pdf', dpi=300)
plt.close('all')


plt.imshow(ndimage.maximum_filter(greyScale, size=11), cmap = 'gray')
plt.gca().set_axis_off()
plt.xlim(225,425)
plt.ylim(225,25)
plt.savefig(save+'maximumFilter/s10.pdf', bbox_inches='tight', pad_inches=0, format = 'pdf', dpi=300)
plt.close('all')

os.system('for a in '+save+'maximumFilter/*.pdf; do pdfcrop "$a" "$a"; done')

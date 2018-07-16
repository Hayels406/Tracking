import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.spatial.distance import mahalanobis
import cv2






sys.argv = ['CH2.py', '20', '0']
execfile('CH2.py')


greyScale = np.copy(img)
plt.imshow(ndimage.maximum_filter(greyScale, size=3), cmap = 'gray')
plt.gca().set_axis_off()
plt.xlim(225,425)
plt.ylim(800,600)
plt.savefig(save+'maximumFilter/s3.pdf', bbox_inches='tight', pad_inches=0, format = 'pdf', dpi=300)
plt.close('all')


plt.imshow(ndimage.maximum_filter(greyScale, size=5), cmap = 'gray')
plt.gca().set_axis_off()
plt.xlim(225,425)
plt.ylim(800,600)
plt.savefig(save+'maximumFilter/s5.pdf', bbox_inches='tight', pad_inches=0, format = 'pdf', dpi=300)
plt.close('all')


plt.imshow(ndimage.maximum_filter(greyScale, size=13), cmap = 'gray')
plt.gca().set_axis_off()
plt.xlim(225,425)
plt.ylim(800,600)
plt.savefig(save+'maximumFilter/s13.pdf', bbox_inches='tight', pad_inches=0, format = 'pdf', dpi=300)
plt.close('all')

os.system('for a in '+save+'maximumFilter/*.pdf; do pdfcrop "$a" "$a"; done')

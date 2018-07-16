import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import cv2






sys.argv = ['CH2.py', '19', '0']
execfile('CH2.py')


greyscale = np.copy(grey2)
plt.imshow(cv2.GaussianBlur(greyscale,(5,5),2), cmap = 'gray')
plt.gca().set_axis_off()
plt.xlim(225,425)
plt.ylim(800,600)
plt.savefig(save+'gauss/l5s2.png', bbox_inches='tight', pad_inches=0, dpi=300)
plt.close('all')


plt.imshow(cv2.GaussianBlur(greyscale,(5,5),20), cmap = 'gray')
plt.gca().set_axis_off()
plt.xlim(225,425)
plt.ylim(800,600)
plt.savefig(save+'gauss/l5s20.png', bbox_inches='tight', pad_inches=0, dpi=300)
plt.close('all')


plt.imshow(cv2.GaussianBlur(greyscale,(13,13),2), cmap = 'gray')
plt.gca().set_axis_off()
plt.xlim(225,425)
plt.ylim(800,600)
plt.savefig(save+'gauss/l13s2.png', bbox_inches='tight', pad_inches=0, dpi=300)
plt.close('all')

plt.imshow(cv2.GaussianBlur(greyscale,(13,13),20), cmap = 'gray')
plt.gca().set_axis_off()
plt.xlim(225,425)
plt.ylim(800,600)
plt.savefig(save+'gauss/l13s20.png', bbox_inches='tight', pad_inches=0, dpi=300)
plt.close('all')

os.system('for a in '+save+'gauss/*.png; do convert "$a" -trim "$a"; done')

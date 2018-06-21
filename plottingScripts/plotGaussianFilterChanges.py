import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import cv2






sys.argv = ['getCoordsVideo.py', 10, 'N']
execfile('getCoordsVideo.py')


greyScale = np.copy(grey2)
plt.imshow(cv2.GaussianBlur(greyscale,(5,5),2), cmap = 'gray')
plt.gca().set_axis_off()
plt.xlim(225,425)
plt.ylim(225,25)
plt.savefig(save+'gauss/l5s2.pdf', bbox_inches='tight', pad_inches=0, format = 'pdf', dpi=300)
plt.close('all')


plt.imshow(cv2.GaussianBlur(greyscale,(5,5),20), cmap = 'gray')
plt.gca().set_axis_off()
plt.xlim(225,425)
plt.ylim(225,25)
plt.savefig(save+'gauss/l5s20.pdf', bbox_inches='tight', pad_inches=0, format = 'pdf', dpi=300)
plt.close('all')


plt.imshow(cv2.GaussianBlur(greyscale,(13,13),2), cmap = 'gray')
plt.gca().set_axis_off()
plt.xlim(225,425)
plt.ylim(225,25)
plt.savefig(save+'gauss/l13s2.pdf', bbox_inches='tight', pad_inches=0, format = 'pdf', dpi=300)
plt.close('all')

plt.imshow(cv2.GaussianBlur(greyscale,(13,13),20), cmap = 'gray')
plt.gca().set_axis_off()
plt.xlim(225,425)
plt.ylim(225,25)
plt.savefig(save+'gauss/l13s20.pdf', bbox_inches='tight', pad_inches=0, format = 'pdf', dpi=300)
plt.close('all')

os.system('for a in '+save+'gauss/*.pdf; do pdfcrop "$a" "$a"; done')

import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import os

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

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

f, axes = plt.subplots(1, 3, sharey=True)
(ax1, ax2, ax3) = axes

z = np.zeros((7,7))

z[3,3] = 10
z[0,0] = 3
z[6,6] = 8
z[0,6] = 5
z[6,0] = 6
ax1.imshow(z, cmap='magma')
ax1.axhline(0.5)
ax1.axhline(1.5)
ax1.axhline(2.5)
ax1.axhline(3.5)
ax1.axhline(4.5)
ax1.axhline(5.5)

ax1.axvline(0.5)
ax1.axvline(1.5)
ax1.axvline(2.5)
ax1.axvline(3.5)
ax1.axvline(4.5)
ax1.axvline(5.5)

ax1.set_axis_off()
#plt.savefig(save+'maximumFilter/init.pdf', bbox_inches='tight', pad_inches=0, format = 'pdf', dpi=300)
#plt.close('all')
ax2.imshow(ndimage.maximum_filter(z, size=3), cmap='magma')
ax2.axhline(0.5)
ax2.axhline(1.5)
ax2.axhline(2.5)
ax2.axhline(3.5)
ax2.axhline(4.5)
ax2.axhline(5.5)

ax2.axvline(0.5)
ax2.axvline(1.5)
ax2.axvline(2.5)
ax2.axvline(3.5)
ax2.axvline(4.5)
ax2.axvline(5.5)

ax2.set_axis_off()
#plt.savefig(save+'maximumFilter/maxFilter_s3.pdf', bbox_inches='tight', pad_inches=0, format = 'pdf', dpi=300)
#plt.close('all')
im = ax3.imshow(ndimage.maximum_filter(z, size=5), cmap='magma')
ax3.axhline(0.5)
ax3.axhline(1.5)
ax3.axhline(2.5)
ax3.axhline(3.5)
ax3.axhline(4.5)
ax3.axhline(5.5)

ax3.axvline(0.5)
ax3.axvline(1.5)
ax3.axvline(2.5)
ax3.axvline(3.5)
ax3.axvline(4.5)
ax3.axvline(5.5)

ax3.set_axis_off()
f.subplots_adjust(right=0.8)
cbar = f.colorbar(im, ax=axes.ravel().tolist(), shrink=0.3)
cbar.ax.tick_params(labelsize=8)
plt.savefig(save+'maximumFilter/maxFilter.pdf', bbox_inches='tight', pad_inches=0, format = 'pdf', dpi=300)
plt.close('all')

z = np.zeros((7,7))

z[3,3] = 10
z[0,0] = 3
z[6,6] = 8
z[0,6] = 5
z[6,0] = 6
plt.imshow(z, cmap='magma')
plt.axhline(0.5)
plt.axhline(1.5)
plt.axhline(2.5)
plt.axhline(3.5)
plt.axhline(4.5)
plt.axhline(5.5)

plt.axvline(0.5)
plt.axvline(1.5)
plt.axvline(2.5)
plt.axvline(3.5)
plt.axvline(4.5)
plt.axvline(5.5)

plt.gca().set_axis_off()
plt.savefig(save+'maximumFilter/maxFilter_s1.pdf', bbox_inches='tight', pad_inches=0, format = 'pdf', dpi=300)
plt.close('all')

plt.imshow(ndimage.maximum_filter(z, size=3), cmap='magma')
plt.axhline(0.5)
plt.axhline(1.5)
plt.axhline(2.5)
plt.axhline(3.5)
plt.axhline(4.5)
plt.axhline(5.5)

plt.axvline(0.5)
plt.axvline(1.5)
plt.axvline(2.5)
plt.axvline(3.5)
plt.axvline(4.5)
plt.axvline(5.5)

plt.gca().set_axis_off()
plt.savefig(save+'maximumFilter/maxFilter_s3.pdf', bbox_inches='tight', pad_inches=0, format = 'pdf', dpi=300)
plt.close('all')

plt.imshow(ndimage.maximum_filter(z, size=5), cmap='magma')
plt.axhline(0.5)
plt.axhline(1.5)
plt.axhline(2.5)
plt.axhline(3.5)
plt.axhline(4.5)
plt.axhline(5.5)

plt.axvline(0.5)
plt.axvline(1.5)
plt.axvline(2.5)
plt.axvline(3.5)
plt.axvline(4.5)
plt.axvline(5.5)

plt.gca().set_axis_off()
plt.savefig(save+'maximumFilter/maxFilter_s5.pdf', bbox_inches='tight', pad_inches=0, format = 'pdf', dpi=300)
plt.close('all')

os.system('for a in '+save+'maximumFilter/*.pdf; do pdfcrop "$a" "$a"; done')

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

colour = plt.imread('./checking/beardguyWOTE.jpg')
grey = cv2.cvtColor(colour, cv2.COLOR_RGB2GRAY)

plt.imshow(grey, cmap = 'gray')
plt.gca().set_axis_off()
plt.savefig('checking/BWBeardGuy.pdf', format = 'pdf', dpi = 300)

for gamma in [10, 5, 2, 1, 0.5, 0.25, 0.1]:
    plt.imshow((grey/255.)**gamma, cmap ='gray')
    plt.gca().set_axis_off()
    plt.savefig('checking/'+str(int(gamma*10)).zfill(3)+'BeardGuy.pdf', format = 'pdf', dpi = 300)
    plt.close('all')
    
os.system('for a in checking/*BeardGuy.pdf; do pdfcrop "$a" "$a"; done')

import os
import numpy as np
import cv2
import open3d as o3d
from PIL import Image, ImageGrab
import pyscreenshot


load_path = '/home/krri/Desktop/sync_dist/images'
save_path = '/home/krri/Desktop/sync_dist'

p1 = cv2.imread(os.path.join(load_path, '1662516680600_fl.png'))
p2 = cv2.imread(os.path.join(load_path, '1662516680600_fr.png'))
p3 = cv2.imread(os.path.join(load_path, '1662516680600_rl.png'))
p4 = cv2.imread(os.path.join(load_path, '1662516680600_rr.png'))

h1 = np.hstack((p1, p2))
h2 = np.hstack((p3, p4))
v = np.vstack((h1, h2))

cv2.imwrite(save_path+'/1662516680600.png', v)
# cv2.imshow('1662516824800', v)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
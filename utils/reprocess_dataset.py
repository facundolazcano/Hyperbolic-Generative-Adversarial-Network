import numpy as np
import os 
import os.path as op
import cv2


# script to solve error due to Transform.ToTensor do a transponce HWC -> CHW, and before i did this process
# however i skip this part in the script create_lsun_cat_folder.py
path_dataset = '/home/jenny2/data/lsun/cats_256'

files = os.listdir(path_dataset)

for f in files:
    path_f = op.join(path_dataset, f)
    img = np.load(path_f)
    img = img.transpose([1, 2, 0])
    np.save(path_f, img)
    
    
print('DoNe!!!!!!!!!!!!!')
    
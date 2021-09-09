
import numpy as np
import lmdb # pip install lmdb # pylint: disable=import-error
import cv2 # pip install opencv-python
import io
import PIL.Image
import sys
import os
import os.path as op


lmdb_dir = '/home/jenny2/data/lsun/cat/'#$data.mdb'
path_to_save = '/home/jenny2/data/lsun/cats_256/'
resolution = 256
os.mkdir(path_to_save)


with lmdb.open(lmdb_dir, readonly=True).begin(write=False) as txn:
    total_images = txn.stat()['entries']
    for _idx, (_key, value) in enumerate(txn.cursor()):
        try:
            try:
                img = cv2.imdecode(np.fromstring(value, dtype=np.uint8), 1)
                if img is None:
                    raise IOError('cv2.imdecode failed')
                img = img[:, :, ::-1] # BGR => RGB
            except IOError:
                img = np.asarray(PIL.Image.open(io.BytesIO(value)))
            crop = np.min(img.shape[:2])
            img = img[(img.shape[0] - crop) // 2 : (img.shape[0] + crop) // 2, (img.shape[1] - crop) // 2 : (img.shape[1] + crop) // 2]
            img = PIL.Image.fromarray(img, 'RGB')
            img = img.resize((resolution, resolution), PIL.Image.ANTIALIAS)
            img = np.asarray(img)
            #img = img.transpose([2, 0, 1]) # HWC => CHW
        except:
            print(sys.exc_info()[1])
        np.save(op.join(path_to_save, str(_idx)), img) 
        
print('DoNe!!!!!!')
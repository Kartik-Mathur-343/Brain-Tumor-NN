import cv2
import os
import numpy as np

def get_imgarrays(filepath):
    files = []
    for file in os.listdir(filepath):  
        img = cv2.imread(os.path.join(filepath, file))
        img = cv2.resize(img, (224, 224))
        files.append(img)
    return np.array(files)

nos = get_imgarrays(os.path.join('.', 'brain_tumor_dataset', 'no'))
yess = get_imgarrays(os.path.join('.', 'brain_tumor_dataset', 'yes'))

data = np.concat([nos, yess], 0)
labels = np.array([0 for i in range(len(nos))]+[1 for i in range(len(yess))])

np.save('data', data)
np.save('labels', labels)


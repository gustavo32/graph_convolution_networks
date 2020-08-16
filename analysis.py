# vgg19,0.005,0.3,256
import os
from glob import glob
import numpy as np
import pandas as pd
from scipy import stats

test_indices = list(np.squeeze(np.loadtxt('../test_indices.txt', dtype=int)))
bbox_codes = np.squeeze(pd.read_csv('../images/mit67_subclass_bboxes.txt', sep=' ', usecols=[1,2], header=None).to_numpy())[test_indices]

n_bboxes = []
for code, bb_class in bbox_codes:
    n_bboxes.append(len(glob('dataset/test/{}/bboxes/{}_*'.format(bb_class, code))))


# print(n_bboxes.sum())
# print(labels)
allt = []
for filename in glob('gcn/checkpoints/firstchebnet/efficientnetb7/*'):
    labels = np.squeeze(np.loadtxt(filename + '/labels.txt', dtype=int))
    predicts = np.squeeze(np.loadtxt(filename + '/predicts.txt', dtype=int))

    counter = 0
    corrects = []
    
    for n_bbox in n_bboxes:
        if n_bbox > 0:
            corrects.append(stats.mode(predicts[counter:counter + n_bbox])[0][0] == labels[counter])
            counter += n_bbox
    
    corrects = np.asarray(corrects)
    allt.append(corrects.sum()/len(corrects))

allt = np.asarray(allt)
print(allt.max())
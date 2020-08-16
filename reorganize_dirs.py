import numpy as np
import pandas as pd
import re
import os
import shutil
from glob import glob


def organize_data_by_record(series, bboxes_folder, images_folder, test_indices):
    dest_folder = 'dataset/'
    if series[1] in test_indices:
        dest_folder += 'test/'
    else:
        dest_folder += 'train/'
        
    dest_folder += series[2]
    
    for current_bbox in glob(bboxes_folder + str(series[1]) + '_*'):
        image_in_new_folder = dest_folder + '/bboxes/' + re.findall(r"\\(.*?\.[A-z]{2,5})", current_bbox)[0]
        print(image_in_new_folder)
        if not os.path.isfile(image_in_new_folder):
            shutil.copy(current_bbox, dest_folder + '/bboxes')
    
    if not os.path.isfile(dest_folder + '/global/' + series[0]):
        shutil.copy(images_folder+series[2] + '/' + series[0], dest_folder + '/global')
    

bboxes_folder = '../images/boundingboxes/'
images_folder = '../images/global/'
map_bboxes_path = '../images/mit67_subclass_bboxes.txt'

if not os.path.isdir('dataset'):
    os.mkdir('dataset')
if not os.path.isdir('dataset/train'):
    os.mkdir('dataset/train')
if not os.path.isdir('dataset/test'):
    os.mkdir('dataset/test')
# print(np.squeeze(pd.read_csv('../images/mit67_subclass_globalClasses.txt').to_numpy()))
for classname in np.squeeze(pd.read_csv('../images/mit67_subclass_globalClasses.txt', header=None).to_numpy()):
    if not os.path.isdir('dataset/train/' + classname):
        os.mkdir('dataset/train/' + classname)
    if not os.path.isdir('dataset/test/' + classname):
        os.mkdir('dataset/test/' + classname)
        
    if not os.path.isdir('dataset/train/' + classname + '/bboxes'):
        os.mkdir('dataset/train/' + classname + '/bboxes')
        os.mkdir('dataset/train/' + classname + '/global')
        
    if not os.path.isdir('dataset/test/' + classname + '/bboxes'):
        os.mkdir('dataset/test/' + classname + '/bboxes')
        os.mkdir('dataset/test/' + classname + '/global')
    

test_indices = np.squeeze(pd.read_csv('../new_test_indices.txt', header=None).to_numpy())
map_bboxes = pd.read_csv(map_bboxes_path, usecols=[0,1,2], sep=' ', header=None)
map_bboxes.apply(organize_data_by_record, axis=1, bboxes_folder=bboxes_folder, images_folder=images_folder, test_indices=test_indices)





# vgg19,0.005,0.3,256

import os
from glob import glob
import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict, OrderedDict
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
from matplotlib import image as img_handler, patches
import re
import seaborn as sns

from architectures import FirstChebNet

import torch
from prepare_data import GNNHandler

import torch.nn.functional as F

#####################################


# test_indices = list(np.squeeze(np.loadtxt('../test_indices.txt', dtype=int)))
# bbox_codes = np.squeeze(pd.read_csv('../images/mit67_subclass_bboxes.txt', sep=' ', usecols=[1,2], header=None).to_numpy())

# print(bbox_codes[:, 1])
# counts = np.unique(bbox_codes[:, 1], return_counts=True)
# df = pd.DataFrame()

# df['n_images'] = counts[1]
# df.index = counts[0]
# df.index = df.index.str.upper()

# n_bboxes = []
# for bb_class in np.unique(bbox_codes[:, 1]):
#     n_bboxes.append(len(glob('dataset/*/{}/bboxes/*'.format(bb_class))))

# df['n_bboxes'] = np.asarray(n_bboxes)

# print(df['n_images'].describe())
# print()
# print(df['n_bboxes'].describe())
# print()



# print(df)
# df.to_csv('bboxes_distribuition.csv')


##############################

# test_indices = list(np.squeeze(np.loadtxt('../test_indices.txt', dtype=int)))
# bbox_codes = np.squeeze(pd.read_csv('../images/mit67_subclass_bboxes.txt', sep=' ', usecols=[1,2], header=None).to_numpy())[test_indices]

# n_bboxes = []
# for code, bb_class in bbox_codes:
#     n_bboxes.append(len(glob('dataset/test/{}/bboxes/{}_*'.format(bb_class, code))))


# # print(n_bboxes.sum())
# # print(labels)

# allt = []
# for filename in glob('gcn/checkpoints/firstchebnet/efficientnetb7/*'):
#     labels = np.squeeze(np.loadtxt(filename + '/labels.txt', dtype=int))
#     predicts = np.squeeze(np.loadtxt(filename + '/predicts.txt', dtype=int))

#     counter = 0
#     corrects = []
    
#     for n_bbox in n_bboxes:
#         if n_bbox > 0:
#             corrects.append(stats.mode(predicts[counter:counter + n_bbox])[0][0] == labels[counter])
#             counter += n_bbox
    
#     corrects = np.asarray(corrects)
#     allt.append(corrects.sum()/len(corrects))

# allt = np.asarray(allt)
# print(allt.max())

###########################

# test_indices = list(np.squeeze(np.loadtxt('../test_indices.txt', dtype=int)))
# bbox_codes = np.squeeze(pd.read_csv('../images/mit67_subclass_bboxes.txt', sep=' ', usecols=[0,1,2], header=None).to_numpy())[test_indices]

# n_bboxes = []
# fnames = []
# for fname, code, bb_class in bbox_codes:
#     n_bboxes.append(len(glob('dataset/test/{}/bboxes/{}_*'.format(bb_class, code))))
#     fnames.append(fname)
    
#     if n_bboxes[-1] == 0:
#         n_bboxes = n_bboxes[:-1]
#         fnames = fnames[:-1]

# # print(n_bboxes.sum())
# # print(labels)

# filename = 'gcn/checkpoints/firstchebnet/inceptionv3/256hd-0.8dp-0.001lr'
# labels = np.squeeze(np.loadtxt(filename + '/labels.txt', dtype=int))
# predicts = np.squeeze(np.loadtxt(filename + '/predicts.txt', dtype=int))

# # allt = []

# n_bboxes = np.asarray(n_bboxes)
# print(n_bboxes)
# print('mean: ', n_bboxes.mean())
# print('25%:', np.percentile(n_bboxes, 25))
# print('median: ', np.median(n_bboxes))
# print('75%:', np.percentile(n_bboxes, 75))
# print('std: ', n_bboxes.std())
# print('min: ', n_bboxes.min())
# print('max: ', n_bboxes.max())

# print()

# t = defaultdict(list)
# corrects = []
# counter = 0
# for n_bbox, fname in zip(n_bboxes, fnames):
#     if n_bbox == 47 or n_bbox == 57 or n_bbox == 83:
#         print(n_bbox, ':', fname)
#         print('labels:\n', labels[counter: counter + n_bbox])
#         print('predicts:\n', predicts[counter:counter + n_bbox])
#         print()
        
#     t[n_bbox].extend(predicts[counter:counter + n_bbox] == labels[counter: counter + n_bbox])
#     # corrects.extend(predicts[counter:counter + n_bbox] == labels[counter: counter + n_bbox])
#     counter += n_bbox

# result = {}
# for n_bbox, preds in t.items():
#     p = np.asarray(preds)
#     result[n_bbox] = p.sum()/len(preds)

# result = dict(OrderedDict(sorted(result.items())))
# print(result)

# # pd.DataFrame.from_dict(result, orient='index').to_csv('teste.csv')


# #########################
# ### PLOT THE OUTLIERS ###

# def add_bbox(x, y, width, height, ax):
#     rect = patches.Rectangle((x, y), width-x, height-y, edgecolor='r', facecolor="none")
#     ax.add_patch(rect)

# def return_bboxes_from_unfixed_file(fname, desired_img):
#     with open(fname, 'r') as f:
#         lines = f.read().split('\n')
    
#     bboxes = []
#     for line in lines:
#         line = line.strip()
#         if line.split(' ')[0] == desired_img:
#             des_bboxes = line.split(r' ')[4:]
#             des_bboxes = re.split(r'\s?\D+\s', ' '.join(des_bboxes))[1:]

#             bboxes = [bbox.split(' ') for bbox in des_bboxes]
    
#     return np.asarray(bboxes).astype(int)

            
# outliers = ['casino_0021.jpg', 'kitchen143.jpg', 'conference3.jpg']
# filename_template = 'dataset/test/{}/{}/{}'
# # bbox_codes = np.squeeze(pd.read_csv('../images/mit67_subclass_bboxes.txt', sep=' ', header=None))

# for outlier in outliers:
#     bb_class = re.findall(r'^([a-z]+)', outlier)[0]
#     if bb_class == 'conference':
#         bb_class = 'meeting_room'
    
#     figure, ax = plt.subplots(1)
#     img = img_handler.imread(filename_template.format(bb_class, 'global', outlier))
#     ax.imshow(img)
    
#     for x, y, w, h in return_bboxes_from_unfixed_file('../images/mit67_subclass_bboxes.txt', outlier):
#         add_bbox(x, y, w, h, ax)
#     # print(bboxes_codes[bboxes_codes.iloc[0] == outlier])
#     # add_bbox(x, y, width, height, ax)
#     plt.show()


#########################################
### GET THE MAX ACC AND THEIR METRICS ###

# for pretrained_cnn in ['vgg19', 'resnet50', 'inceptionv3', 'efficientnetb7']:
#     filename_template = 'gcn/checkpoints/firstchebnet/{}/*'
#     max_acc = 0
#     for filename in glob(filename_template.format(pretrained_cnn)):
#         labels = np.squeeze(np.loadtxt(filename + '/labels.txt', dtype=int))
#         predicts = np.squeeze(np.loadtxt(filename + '/predicts.txt', dtype=int))
#         if accuracy_score(labels, predicts) > max_acc:
#             max_acc = accuracy_score(labels, predicts)
#             max_recall = recall_score(labels, predicts, average='macro')
#             max_precision = precision_score(labels, predicts, average='macro')
#             max_f1 = f1_score(labels, predicts, average='macro')
    
#     print(pretrained_cnn)
#     print('{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t'.format(max_acc*100, max_recall*100, max_precision*100, max_f1*100))
#     print()

########################################

#############################################
### PLOT CONFUSION_MATRIX OF GREATEST ACC ###

max_acc = 0
max_confusion_matrix = None
for pretrained_cnn in ['vgg19', 'resnet50', 'inceptionv3', 'efficientnetb7']:
    filename_template = 'gcn/checkpoints/firstchebnet/{}/*'
    for filename in glob(filename_template.format(pretrained_cnn)):
        labels = np.squeeze(np.loadtxt(filename + '/labels.txt', dtype=int))
        predicts = np.squeeze(np.loadtxt(filename + '/predicts.txt', dtype=int))
        if accuracy_score(labels, predicts) > max_acc:
            max_confusion_matrix = np.squeeze(np.loadtxt(filename + '/confusion_matrix.txt', dtype=int))
            max_acc = accuracy_score(labels, predicts)
            print(filename)
ax = plt.subplot()
sns.heatmap(max_confusion_matrix, annot=False, ax = ax); #annot=True to annotate cells
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
plt.show()

########################################


####################
### PYTORCH TEST ###


# obj = 'gcn/storage/inceptionv3/graph.bin'
# g, features, labels, train_mask, val_mask, test_mask, in_feats, n_classes = GNNHandler.get_info_from_graph(obj)

# cuda = True
# torch.cuda.set_device(0)
# features = features.cuda()
# labels = labels.cuda()
# train_mask = train_mask.cuda()
# val_mask = val_mask.cuda()
# test_mask = test_mask.cuda()
# n_edges = g.number_of_edges()
# # normalization
# degs = g.in_degrees().float()
# norm = torch.pow(degs, -0.5)
# norm[torch.isinf(norm)] = 0
# if cuda:
#     norm = norm.cuda()
# g.ndata['norm'] = norm.unsqueeze(1)

# model = FirstChebNet(g,
#                     in_feats,
#                     256,
#                     n_classes,
#                     2,
#                     F.relu,
#                     0.8)

# model.load_state_dict(torch.load('gcn/checkpoints/firstchebnet/inceptionv3/256hd-0.8dp-0.001lr/cp.pt'))

# indices = g.ndata['indice'][test_mask].numpy()
# features = features.cpu()
# labels = labels[test_mask].cpu().numpy()

# _, predicts = torch.max(model(features)[test_mask], dim=1)
# predicts = predicts.numpy()


# corrects = (labels == predicts)
# count_indices = np.unique(indices, return_counts=True)

# t = defaultdict(list)
# corrects = []
# counter = 0
# print(count_indices)
# for ind, n_bbox in zip(*count_indices):
#     print(labels[counter: counter + n_bbox])
#         # print(n_bbox, ':', fname)
#         # print('labels:\n', labels[counter: counter + n_bbox])
#         # print('predicts:\n', predicts[counter:counter + n_bbox])
#         # print()
        
#     t[n_bbox].extend(predicts[counter:counter + n_bbox] == labels[counter: counter + n_bbox])
#     # corrects.extend(predicts[counter:counter + n_bbox] == labels[counter: counter + n_bbox])
#     counter += n_bbox

# result = {}
# for n_bbox, preds in t.items():
#     p = np.asarray(preds)
#     result[n_bbox] = p.sum()/len(preds)

# result = dict(OrderedDict(sorted(result.items())))
# print(result)

# pd.DataFrame.from_dict(result, orient='index').to_csv('teste.csv')


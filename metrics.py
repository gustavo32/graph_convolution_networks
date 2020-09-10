import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np
import os

import tensorflow as tf
from prepare_data import NPDataset

class Metrics():
    def __init__(self, ds, labels=None, mask=None, backend='pytorch'):
        self.backend = backend
        self.ds = ds
        if isinstance(ds, tf.data.Dataset):
            self.features = ds.map(lambda feat, lbl: feat)
            self.labels = np.asarray(list(ds.map(lambda feat, lbl: lbl).as_numpy_iterator()))
        elif isinstance(ds, NPDataset):
            self.features = ds.X
            labels = ds.y
        else:
            self.features = ds
            
        if backend == 'pytorch':
            self.labels = labels[mask]
        else:
            print(labels)
            self.labels = np.argmax(labels, axis=1)
                
        self.mask = mask
    
    def get_pytorch_indices(self, model):
        logits = model(self.features)
        logits = logits[self.mask]
        _, indices = torch.max(logits, dim=1)
        return indices
        
    def get_tensorflow_indices(self, model):
        return np.argmax(model.predict(self.features), axis=1)
    
    def evaluate(self, model):
        if self.backend == 'pytorch':
            model.eval()
            with torch.no_grad():
                indices = self.get_pytorch_indices(model)
                correct = torch.sum(indices == self.labels)
                return correct.item() * 1.0 / len(self.labels)
        else:
            indices = self.get_tensorflow_indices(model)
            return accuracy_score(self.labels, indices)
            
    
    def save_metrics(self, checkpoint, model):
        if self.backend == 'pytorch':
            labels = self.labels.cpu().numpy()
            indices = self.get_pytorch_indices(model).cpu().numpy()
        else:
            labels = self.labels
            indices = self.get_tensorflow_indices(model)
            
        cnf = confusion_matrix(labels, indices)
        clf_report = classification_report(labels, indices)
        
        np.savetxt(checkpoint + '/labels.txt', labels, fmt="%d")
        np.savetxt(checkpoint + '/predicts.txt', indices, fmt="%d")
        np.savetxt(checkpoint + '/confusion_matrix.txt', cnf, fmt="%d")
        with open(checkpoint + '/classification_report.txt', 'w') as f:
            f.write(clf_report)

    @staticmethod
    def save_record(checkpoint, architecture, pretrained_cnn, lr, dropout, n_hidden, acc):
        if not os.path.isfile(checkpoint):
            with open(checkpoint, 'w') as f:
                f.write('structure,pretrained_cnn,learning_rate,dropout,n_hidden,accuracy\n')
        with open(checkpoint, 'a') as f:
            f.write("{},{},{},{},{},{}\n".format(architecture, pretrained_cnn, lr, dropout, n_hidden, acc))
                    
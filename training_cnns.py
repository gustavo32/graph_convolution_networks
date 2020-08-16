import tensorflow as tf
import pandas as pd
import numpy as np
import re
import os
from glob import glob
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from time import time, ctime
import prepare_data as pdata
from architectures import create_model
from metrics import Metrics
from itertools import product

if __name__ == "__main__":
    batch_size = 32
    root = ''
    dataset = 'dataset'
    level = 'global'
    extract_format = 'cnn'
    handler_mode = 'np'
    
    if extract_format == 'cnn':
        n_last_conv = 3
    else:
        n_last_conv = ''
        
    checkpoint_format = root + 'cnn/checkpoints/{}/{}/' + extract_format + str(n_last_conv) + '-{}hd-{}dp-{}lr' 

    for pretrained_cnn in ['vgg19', 'resnet50', 'inceptionv3', 'efficientnetb7']:
        print(pretrained_cnn)
        handler = pdata.CNNHandler(dataset, pretrained_cnn, level, root=root, extract_format=extract_format, n_last_conv=n_last_conv)
        train_ds, test_ds = handler.create_datasets(handler_mode, batch_size)
        print(train_ds.X.shape)

        storage = root + 'cnn/storage/' + pretrained_cnn
        for n_hidden, lr, dropout in product([32, 128, 256], [1e-2, 5e-2, 1e-3, 5e-3], [0.3, 0.5, 0.8, 0.9]):
            checkpoint = checkpoint_format.format(pretrained_cnn, level, n_hidden, dropout, lr)
            if os.path.isfile(checkpoint + '/cp.index'):
                if extract_format == 'cnn':
                    break
                continue
            
            if not os.path.isdir(checkpoint):
                os.makedirs(checkpoint)
            if not os.path.isdir(storage):
                os.makedirs(storage)
                
            model = create_model(len(np.unique(handler.classes_path)), n_hidden, dropout, handler.pretrained_model, extract_format)
            with tf.device('/GPU:0'):
                model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=lr))
                
                if handler_mode == 'np':
                    model.fit(train_ds.X, train_ds.y, epochs=2000, batch_size=batch_size)
                elif handler_mode == 'tf':
                    print(len(self.classes_path))
                    model.fit(train_ds, epochs=2000, steps_per_epoch=len(self.classes_path) // batch_size)
            
            model.save_weights(checkpoint + '/cp')
            
            mtc = Metrics(test_ds, backend='tensorflow')
            mtc.save_record(root + 'cnn/' + level + '_results.txt', extract_format, pretrained_cnn, lr, dropout, n_hidden, mtc.evaluate(model))
            
            if extract_format == 'cnn':
                break
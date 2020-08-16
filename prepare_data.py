import dgl
from dgl.data import utils
import torch

import tensorflow as tf
from tensorflow.keras.applications import vgg19, resnet, inception_v3, efficientnet
from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import OneHotEncoder

import numpy as np
import pandas as pd
import networkx as nx

import os
from glob import glob
from collections import defaultdict
from tqdm import tqdm
import tempfile

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

class NPDataset():
    def __init__(self, X, y):
        self.X = X
        self.y = y

def get_model_from_string(pretrained_cnn):
    if pretrained_cnn == 'vgg19':
        structure = vgg19
        pretrained_model = vgg19.VGG19
    elif pretrained_cnn == 'inceptionv3':
        structure = inception_v3
        pretrained_model = inception_v3.InceptionV3
    elif pretrained_cnn == 'resnet50':
        structure = resnet
        pretrained_model = resnet.ResNet50
    elif pretrained_cnn == 'efficientnetb7':
        structure = efficientnet
        pretrained_model = efficientnet.EfficientNetB7
    else:
        raise KeyError('Please, implement ' + pretrained_cnn + ' on prepare_data.py!')
    
    return structure, pretrained_model

class GNNHandler():
    
    def __init__(self, dataset, pretrained_cnn_str, input_shape=[224, 224, 3]):
        self.dataset = dataset
        self.input_shape = input_shape
        
        self.structure, self.pretrained_model = get_model_from_string(pretrained_cnn_str)
        self.pretrained_model = self.pretrained_model(input_shape = input_shape, include_top = False, pooling='max')
        classes = np.unique([path.split(os.sep)[-1] for path in glob(self.dataset + '/train/*')])
        self.map_label = dict(zip(classes, np.arange(len(classes))))
    
    def _get_indice_from_path(self, path):
        splitted_path = path.split(os.sep)
        return splitted_path[-3] + ';' + splitted_path[-1].split("_")[0]
        
    def group_bboxes_by_indice(self, paths):
        bboxes_set = defaultdict(list)
        root_path = (os.sep).join(paths[0].split(os.sep)[:1])
        str_template = '{}/{}/bboxes/{}_*'
        
        vectorized_indice_extractor = np.vectorize(self._get_indice_from_path)
        for class_and_indice in np.unique(vectorized_indice_extractor(paths)):
            bbox_class, indice = class_and_indice.split(';')
            bboxes_set[indice] = pd.Series(glob(str_template.format(root_path, bbox_class, indice)))
        
        return bboxes_set
    
    @staticmethod
    def full_graph(n_nodes):
        new_graph = dgl.DGLGraph()
        new_graph.add_nodes(n_nodes)
        nodes = np.arange(n_nodes)
        for current_node in nodes:
            new_graph.add_edges(current_node, nodes)
        return new_graph
            
    @staticmethod
    def concat_graph(list_graphs):
        new_graph = dgl.batch(list_graphs)
        new_graph.flatten()
        return new_graph
    
    @staticmethod
    def plot_graph(self, graph, with_labels=False):
        nx.draw(graph.to_networkx(), with_labels=with_labels)
        plt.show()
        
    def build_subgraph(self, paths, batch_size=32):
        graph = dgl.DGLGraph()
        counter = 0
        f = tempfile.TemporaryDirectory(dir = os.path.dirname(os.path.abspath(__file__)))
        for indice, bboxes in tqdm(list(self.group_bboxes_by_indice(paths).items())):
            splitted_path = (os.sep).join(bboxes[0].split('/'))
            splitted_path = splitted_path.split(os.sep)
            
            bboxes_class, subset = splitted_path[-3], splitted_path[-4]
            total_nodes = len(bboxes)
            subset = True if subset == 'train' else False 

            current_graph = self.full_graph(total_nodes) # create graph
            # set the node attributes
            current_graph.ndata['features'] = np.concatenate(bboxes.apply(CNNHandler.extract_features_from_file,
                                                                          structure=self.structure,
                                                                          pretrained_model=self.pretrained_model,
                                                                          input_shape=self.input_shape))
            
            current_graph.ndata['class'] = np.asarray([self.map_label[bboxes_class]] * total_nodes).astype(int)
            current_graph.ndata['indice'] = np.asarray([indice] * total_nodes).astype(int)
            current_graph.ndata['isTraining'] = np.asarray([subset] * total_nodes)
            graph = self.concat_graph([graph, current_graph])
            
            counter += 1
            
            if counter % batch_size == 0:
                self.save_graph(f.name + '/' + str(counter) + '_graph.bin', graph)
                graph = dgl.DGLGraph()
        
        self.save_graph(f.name + '/last_graph.bin', graph)
        graph = self.concat_graph([self.load_graph(gpath) for gpath in glob(f.name + '/*')])
        f.cleanup()
        return graph
    
    def build_graph(self):
        train_graph = self.build_subgraph(glob(self.dataset + '/train/*/bboxes/*'))
        test_graph = self.build_subgraph(glob(self.dataset + '/test/*/bboxes/*'))
        return self.concat_graph([train_graph, test_graph])
    
    @staticmethod
    def save_graph(filename, graph):
        utils.save_graphs(filename, graph)
    
    @staticmethod
    def load_graph(filename):
        graph, _ = utils.load_graphs(filename)
        return graph[0]

    @staticmethod
    def get_info_from_graph(obj):
        if not isinstance(obj, dgl.DGLGraph):
            graph = GNNHandler.load_graph(obj)
        else:
            graph = obj
            
        features = graph.ndata['features']
        labels = graph.ndata['class'].type(torch.LongTensor)
        train_mask = graph.ndata['isTraining'].type(torch.BoolTensor)
        test_mask = (train_mask == False).type(torch.BoolTensor)
        val_mask = test_mask
        size = features.shape[1]
        n_class = (torch.max(labels) + 1).numpy()
        
        return graph, features, labels, train_mask, val_mask, test_mask, size, n_class
        
        
class CNNHandler():
    
    def __init__(self, dataset, pretrained_cnn_str, level, input_shape=[224, 224, 3], root='', extract_format='dense', n_last_conv=None):
        self.dataset = dataset
        self.input_shape = input_shape
        self.pretrained_cnn_str = pretrained_cnn_str
        self.structure, self.pretrained_model = get_model_from_string(pretrained_cnn_str)
        self.extract_format = extract_format
        self.n_last_conv = n_last_conv
        self.pretrained_model = self.pretrained_model(input_shape = input_shape, include_top = False, pooling='max')
        if extract_format == 'cnn':
            self.pretrained_model = self.freeze_layers(self.pretrained_model, n_last_conv)
             
        self.level = level
        self.root = root
        self.classes_path = np.asarray([path.split(os.sep)[-1] for path in glob(root + dataset + '/train/*')])
        self.one_hot = OneHotEncoder()
        self.one_hot.fit(self.classes_path.reshape(-1, 1))
        
    @staticmethod
    def show(image, label):
        plt.figure()
        plt.imshow(image)
        plt.title(label.numpy())
        plt.axis('off')
        plt.show()
    
    @staticmethod
    def freeze_layers(model_f, n_last_conv):
        for layer in model_f.layers:
            layer.trainable = False
        
        counter = 0
        for layer in model_f.layers[::-1]:
            if isinstance(layer, tf.keras.layers.Conv2D):
                layer.trainable = True
                counter += 1
            if counter == n_last_conv:
                break
    
        return model_f
    # @staticmethod
    # def get_mid_conv(model_f, n_last_conv):
    #     counter = 0
    #     unfrozen_layers = []
    #     frozen_layers = []
    #     print(model_f)
    #     for layer in model_f.layers[::-1]:
    #         if counter < n_last_conv:
    #             unfrozen_layers.append(layer)
    #         else:
    #             frozen_layers.append(layer)
                
    #         if isinstance(layer, tf.keras.layers.Conv2D):
    #             counter += 1

    #     frozen_layers = frozen_layers[::-1]
    #     unfrozen_layers = unfrozen_layers[::-1]
        
        # print(tf.keras.Model(inputs=frozen_layers[0].input, outputs=frozen_layers[-1].output).summary())
        # print(tf.keras.Sequential(unfrozen_layers).summary())
        
        # return tf.keras.Model(inputs=frozen_layers[0].input, outputs=frozen_layers[-1].output), unfrozen_layers
                
    @staticmethod
    def extract_features_from_file(path, structure, pretrained_model, input_shape, extract_format='dense'):
        img = image.load_img(path, target_size=input_shape[:2], color_mode="rgb")
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = structure.preprocess_input(x)
        if extract_format == 'dense':
            x = pretrained_model.predict(x)
        return x
    
    @staticmethod
    def extract_features_from_file_tf(path, structure, pretrained_model, input_shape, extract_format='dense'):
        x = tf.io.read_file(path)
        x = tf.image.decode_png(x, channels=3)
        x = tf.image.resize(x, input_shape[:2])
        x = tf.expand_dims(x, axis=0)
        x = structure.preprocess_input(x)
        if extract_format == 'dense':
            x = pretrained_model(x)
        return x
    
    def create_dataset_np(self, subset, paths):
        filename_template = self.root + 'cnn/storage/' + self.pretrained_cnn_str + '/' + \
                            self.extract_format + str(self.n_last_conv) + '_' + self.level + '_{}_' + subset + '.npy'
             
        if not os.path.isfile(filename_template.format('X')):
            rand_indices = np.random.choice(len(paths), len(paths), replace=False)
            paths = paths[rand_indices]
            
            labels = self.one_hot.transform(
                        np.asarray([path.split(os.sep)[-3] for path in paths]).reshape(-1, 1)
                     ).toarray().astype(int)
            
            images = np.concatenate(
                        paths.apply(self.extract_features_from_file, structure=self.structure,
                                    pretrained_model=self.pretrained_model, input_shape=self.input_shape,
                                    extract_format=self.extract_format)
                        .to_numpy()
                    )
            
            np.save(filename_template.format('y'), labels)
            np.save(filename_template.format('X'), images)
        else:
            images = np.load(filename_template.format('X'))
            labels = np.load(filename_template.format('y'))
        
        return images, labels
        
    def create_dataset_tf(self, subset, paths):
        filename_template = self.root + 'cnn/storage/' + self.pretrained_cnn_str + '/' + \
                            self.extract_format + str(self.n_last_conv) + '_' + self.level + '_{}_' + subset + '.tfrecord'
        
        if not os.path.isdir(filename_template.format('X')):
            rand_indices = np.random.choice(len(paths), len(paths), replace=False)
            paths = paths[rand_indices]
            
            labels = tf.data.Dataset.from_tensor_slices((self.one_hot.transform(
                        np.asarray([path.split(os.sep)[-3] for path in paths]).reshape(-1, 1)
                    ).toarray().astype(int)))

            images = tf.data.Dataset.from_tensor_slices(paths).map(lambda x: self.extract_features_from_file_tf(x, self.structure,
                                                                   self.pretrained_model, self.input_shape, self.extract_format),
                                                                   num_parallel_calls=tf.data.experimental.AUTOTUNE)
            
            tf.data.experimental.save(labels, filename_template.format('y'))
            tf.data.experimental.save(images, filename_template.format('X'))
        else:
            ds_shape = list(paths[:1].apply(self.extract_features_from_file, structure=self.structure,
                                            pretrained_model=self.pretrained_model, input_shape=self.input_shape,
                                            extract_format=self.extract_format)[0].shape)
            
            images = tf.data.experimental.load(filename_template.format('X'), tf.TensorSpec(shape=ds_shape[1:], dtype=tf.float64))
            labels = tf.data.experimental.load(filename_template.format('y'), 
                                                    tf.TensorSpec(shape=(len(self.one_hot.categories_[0])), dtype=tf.int64)
                                               )
        return images, labels
        
        
    def create_datasets(self, mode='np', batch_size=None): 
        path_template = self.root + self.dataset + '/{}/*/{}/*'
        
        if mode == 'np':
            X, y = self.create_dataset_np('train', pd.Series(glob(path_template.format('train', self.level))))
            train_ds = NPDataset(X, y)
            
            X, y = self.create_dataset_np('test', pd.Series(glob(path_template.format('test', self.level))))
            test_ds = NPDataset(X, y)
            
        elif mode == 'tf':
            X_train, y_train = self.create_dataset_tf('train', pd.Series(glob(path_template.format('train', self.level))))
            X_test, y_test = self.create_dataset_tf('test', pd.Series(glob(path_template.format('test', self.level))))
            
            train_ds = tf.data.Dataset.zip((X_train, y_train)).batch(batch_size)
            test_ds = tf.data.Dataset.zip((X_test, y_test)).batch(batch_size)
            
        return train_ds, test_ds
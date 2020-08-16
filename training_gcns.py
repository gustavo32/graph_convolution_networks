import time
import numpy as np
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F

from architectures import FirstChebNet
from metrics import  Metrics

import sys
import pickle as pkl
from datetime import datetime
import os

import scipy.sparse as sp
import pandas as pd
from prepare_data import GNNHandler
from itertools import product

class Parameters():

    def __init__(self, storage=None, dataset=None, arch_gcn=None, gpu=None, self_loop=None, n_hidden=None,
                 n_layers=None, dropout=None, lr=None, weight_decay=None, n_epochs=None, save_path=None,
                 pretrained_cnn=None):
        
        self.storage = storage
        self.dataset = dataset
        self.arch_gcn = arch_gcn
        self.gpu = gpu
        self.self_loop = self_loop
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.save_path = save_path
        self.pretrained_cnn = pretrained_cnn
    
    
def main(args):
    assert args.arch_gcn in ['firstchebnet'], '[ERROR] Architecture not implemented!'
    assert args.dataset == 'mit67', '[ERROR] Dataset not supported yet!'
    
    obj = args.storage + '/graph.bin'
    # load and preprocess dataset
    if not os.path.isfile(obj):
        print('Graph not found!')
        print('Creating graph...')
        gh = GNNHandler('dataset', args.pretrained_cnn)
        graph = gh.build_graph()
        gh.save_graph(obj, graph)
        obj = graph
        
    g, features, labels, train_mask, val_mask, test_mask, in_feats, n_classes = GNNHandler.get_info_from_graph(obj)
    n_edges = g.number_of_edges()
    
    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
              train_mask.sum().item(),
              val_mask.sum().item(),
              test_mask.sum().item()))

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        features = features.cuda()
        labels = labels.cuda()
        train_mask = train_mask.cuda()
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()

    # graph preprocess and calculate normalization factor
    # add self loop
    # if args.self_loop:
    #     g.remove_edges_from(g.selfloop_edges())
    #     g.add_edges_from(zip(g.nodes(), g.nodes()))
    
    n_edges = g.number_of_edges()
    # normalization
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    if cuda:
        norm = norm.cuda()
    g.ndata['norm'] = norm.unsqueeze(1)

    if args.arch_gcn == 'firstchebnet':
        model = FirstChebNet(g,
                            in_feats,
                            args.n_hidden,
                            n_classes,
                            args.n_layers,
                            F.relu,
                            args.dropout)
    
    else:
        print('ARCHITECTURE NOT IMPLEMENT! EXITING...')
        exit(1)

    if cuda:
        model.cuda()
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    mtc = Metrics(features, labels, val_mask, backend='pytorch')
    # initialize graph
    dur = []
    for epoch in range(args.n_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        acc = mtc.evaluate(model)
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
              "ETputs(KTEPS) {:.2f}". format(epoch, np.mean(dur), loss.item(),
                                             acc, n_edges / np.mean(dur) / 1000))

    print("Test Accuracy {:.2%}".format(mtc.evaluate(model)))
    
    mtc.save_metrics(args.save_path, model)
    torch.save(model.state_dict(), args.save_path + '/cp.pt')
    
    return mtc, model

if __name__ == "__main__":
    arch_gcn = 'firstchebnet'
    checkpoint_format = 'gcn/checkpoints/{}/{}/{}hd-{}dp-{}lr' 

    for pretrained_cnn in ['vgg19', 'resnet50', 'inceptionv3', 'efficientnetb7']:
        print(pretrained_cnn)
        storage = 'gcn/storage/' + pretrained_cnn
        for n_hidden, lr, dropout in product([32, 128, 256], [1e-2, 5e-2, 1e-3, 5e-3], [0.3, 0.5, 0.8, 0.9]):
            checkpoint = checkpoint_format.format(arch_gcn, pretrained_cnn, n_hidden, dropout, lr)
            if os.path.isfile(checkpoint + '/cp.pt'):
                continue 
            
            if not os.path.isdir(checkpoint):
                os.makedirs(checkpoint)
            
            if not os.path.isdir(storage):
                os.makedirs(storage)
                
            args = Parameters(storage=storage, dataset='mit67', arch_gcn=arch_gcn, gpu=0, self_loop=False, n_hidden=n_hidden, n_layers=2,
                                dropout=dropout, lr=lr, weight_decay=0, n_epochs=2000, save_path=checkpoint, pretrained_cnn=pretrained_cnn)
            mtc, model = main(args)
            mtc.save_record('gcn/results.txt', arch_gcn, pretrained_cnn, lr, dropout, n_hidden, mtc.evaluate(model))
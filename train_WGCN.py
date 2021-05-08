#  MIT License
#
#  Copyright (c) 2020 WGCN Authors
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

import argparse
import json
import os
import time

import scipy.sparse as sp
import dgl.init
import numpy as np
import torch
import torch as th
import torch.nn.functional as F
import utils_data
from utils_layers import WGCNNet
from utils_structural import generate_dijkstra, load_adj, compute_structural_infot

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="WGCN_TwoLayers")
    parser.add_argument('--dataset', type=str, default="cora_ml")
    parser.add_argument('--directed', type=bool, default=True)
    parser.add_argument('--dataset_embedding', type=str, default="isomap")
    parser.add_argument('--num_hidden', type=int, default=48)
    parser.add_argument('--num_heads_layer_one', type=int, default=1)
    parser.add_argument('--num_heads_layer_two', type=int, default=1)
    parser.add_argument('--layer_one_ggcn_merge', type=str, default='cat')
    parser.add_argument('--layer_two_ggcn_merge', type=str, default='cat')
    parser.add_argument('--layer_one_channel_merge', type=str, default='cat')
    parser.add_argument('--layer_two_channel_merge', type=str, default='mean')
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--weight_decay_layer_one', type=float, default=5e-6)
    parser.add_argument('--weight_decay_layer_two', type=float, default=5e-6)
    parser.add_argument('--num_epochs_max', type=int, default=1000)
    parser.add_argument('--run_id', type=str, default="000")
    parser.add_argument('--dataset_split', type=str, default="random")
    parser.add_argument('--attention', type=bool, default=False)
    parser.add_argument('--learning_rate_decay_patience', type=int, default=50)
    parser.add_argument('--learning_rate_decay_factor', type=float, default=0.8)
    parser.add_argument('--num_epochs_patience', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=0.05)
    parser.add_argument('--nfold', type=int, default=4)
    parser.add_argument('--in_out_ratio', type=float, default=3)
    parser.add_argument('--restart_rate', type=float, default=0.0)
    parser.add_argument('--in_out_peak', type=float, default=0.4)
    parser.add_argument('--dijkstra_k', type=int, default=1)
    parser.add_argument('--latent', type=bool, default=True)
    parser.add_argument('--ng', type=bool, default=False)
    parser.add_argument('--layers', type=int, default=2)
    args = parser.parse_args()
#    args.dataset_embedding='poincare'
    if os.path.exists('data/{}/{}_structural_{}_{}_{}_{}.npz'.format(args.dataset,args.dataset,args.in_out_ratio,args.restart_rate,args.in_out_peak,args.dijkstra_k)):
        structural_info = sp.load_npz('data/{}/{}_structural_{}_{}_{}_{}.npz'.format(args.dataset,args.dataset,args.in_out_ratio,args.restart_rate,args.in_out_peak,args.dijkstra_k))
        structural_info = structural_info.toarray()
    else:
        if not os.path.exists("data/{}/{}_{}_dijkstra.npz".format(args.dataset,args.dataset,args.dijkstra_k)):
            generate_dijkstra(args.dataset, args.dijkstra_k)
        origin_adj = load_adj(args.dataset)
        structural_info = compute_structural_infot(args.dataset, args.directed, args.dijkstra_k, args.in_out_ratio,args.restart_rate,args.in_out_peak)
        #structural_info = compute_structural_info(args.dataset,origin_adj, args.directed, args.dijkstra_k, args.in_out_ratio,args.restart_rate,args.in_out_peak)
        structural_info = structural_info.toarray()

    if args.dataset_split == 'random':
#        g, features, labels, train_mask, val_mask, test_mask, num_features, num_labels, num_devisions, pairs = utils_data.load_data_custome(
#            args.dataset, args.dijkstra_k, args.nfold, None, 20,500, 'WGCN', args.dataset_embedding, structural_info,args.latent,args.ng)

        g, features, labels, train_mask, val_mask, test_mask, num_features, num_labels, num_devisions, pairs = utils_data.load_data(
            args.dataset, args.dijkstra_k, args.nfold, None, 0.6, 0.2, 'WGCN', args.dataset_embedding, structural_info,args.latent,args.ng)
    else:
        g, features, labels, train_mask, val_mask, test_mask, num_features, num_labels, num_devisions, pairs = utils_data.load_data(
            args.dataset, args.dijkstra_k, args.nfold, args.dataset_split, None, None, 'WGCN', args.dataset_embedding, structural_info,args.latent,args.ng)

    g.set_n_initializer(dgl.init.zero_initializer)
    g.set_e_initializer(dgl.init.zero_initializer)
    net = WGCNNet(g=g, num_input_features=num_features, num_output_classes=num_labels, num_hidden=args.num_hidden,
                     num_divisions=num_devisions, pairs = pairs, dropout_rate=args.dropout_rate,
                     num_heads_layer_one=args.num_heads_layer_one, num_heads_layer_two=args.num_heads_layer_two,
                     layer_one_ggcn_merge=args.layer_one_ggcn_merge,
                     layer_one_channel_merge=args.layer_one_channel_merge,
                     layer_two_ggcn_merge=args.layer_two_ggcn_merge,
                     layer_two_channel_merge=args.layer_two_channel_merge, attention=args.attention,layers=args.layers)

    optimizer = th.optim.Adam([{'params': net.wgcn1.parameters(), 'weight_decay': args.weight_decay_layer_one},
                               {'params': net.wgcn2.parameters(), 'weight_decay': args.weight_decay_layer_two}],
                              lr=args.learning_rate)
    learning_rate_scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                                      factor=args.learning_rate_decay_factor,
                                                                      patience=args.learning_rate_decay_patience)

    net.cuda()
    features = features.cuda()
    structural_info = torch.tensor(structural_info)
    structural_info = structural_info.cuda()
    labels = labels.cuda()
    train_mask = train_mask.cuda()
    val_mask = val_mask.cuda()
    test_mask = test_mask.cuda()

    # Adapted from https://github.com/PetarV-/GAT/blob/master/execute_cora.py
    vlss_mn = np.inf
    vacc_mx = 0.0
    vacc_early_model = None
    vlss_early_model = None
    tacc_early_model = None
    tlss_early_model = None
    state_dict_early_model = None
    curr_step = 0
    result_epoch = 0
    # Adapted from https://docs.dgl.ai/tutorials/models/1_gnn/1_gcn.html
    dur = []
    results = ''
    sum_dur = 0
    for epoch in range(args.num_epochs_max):
        t0 = time.time()

        net.train()
        train_logits = net(features)
        train_logp = F.log_softmax(train_logits, 1)
        train_loss = F.nll_loss(train_logp[train_mask], labels[train_mask])
        train_pred = train_logp.argmax(dim=1)
        train_acc = th.eq(train_pred[train_mask], labels[train_mask]).float().mean().item()

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        net.eval()
        with th.no_grad():
            val_logits = net(features)
            val_logp = F.log_softmax(val_logits, 1)
            val_loss = F.nll_loss(val_logp[val_mask], labels[val_mask]).item()
            val_pred = val_logp.argmax(dim=1)
            val_acc = th.eq(val_pred[val_mask], labels[val_mask]).float().mean().item()
            test_logp = F.log_softmax(val_logits, 1)
            test_loss = F.nll_loss(test_logp[test_mask], labels[test_mask]).item()
            test_pred = test_logp.argmax(dim=1)
            test_acc = th.eq(test_pred[test_mask], labels[test_mask]).float().mean().item()
        learning_rate_scheduler.step(val_loss)

        dur.append(time.time() - t0)

        print(
            "Epoch {:05d} | Train Loss {:.4f} | Train Acc {:.4f} | Val Loss {:.4f} | Val Acc {:.4f} | Time(s) {:.4f}".format(
                epoch, train_loss.item(), train_acc, val_loss, val_acc, sum(dur) / len(dur)))

        sum_dur = sum_dur + (sum(dur)/len(dur))
        if (epoch+1)%100 == 0:
            results+= 'epoch: ' + str(epoch+1) + ' (accuracy: ' + str(test_acc) +')  '
        # Adapted from https://github.com/PetarV-/GAT/blob/master/execute_cora.py
        if (val_acc >= vacc_mx or val_loss <= vlss_mn):
            if val_acc >= vacc_mx and val_loss <= vlss_mn:
                vacc_early_model = val_acc
                vlss_early_model = val_loss
                tacc_early_model = test_acc
                tlss_early_model = test_loss
                state_dict_early_model = net.state_dict()
                result_epoch = epoch

            vacc_mx = np.max((val_acc, vacc_mx))
            vlss_mn = np.min((val_loss, vlss_mn))

            curr_step = 0
        else:
            curr_step += 1
            if curr_step >= args.num_epochs_patience:
                break

    printres = 'Epoch: ' + str(result_epoch+1) + ' (accuracy: ' + str(tacc_early_model) + ')' + '   ' + results + '   ' + str(sum_dur/args.num_epochs_max)
    with open(os.path.join('results', '{}_{}_{}_results.txt'.format(args.dataset,args.dataset_embedding, args.run_id)), 'w') as outfile:
        outfile.write(json.dumps(printres) + '\n')

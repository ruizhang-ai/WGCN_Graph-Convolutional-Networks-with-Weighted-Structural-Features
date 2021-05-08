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

import os
from collections import defaultdict

import numpy as np
import networkx as nx
import scipy.sparse as sp
import torch
import pickle

from utils_dijkstra import Graph


from pyrwr.rwr import RWR


def compute_structural_infot(dataset_str, directed, dijkstra_k, in_out_ratio, restart_rate, in_out_peak):
    dijkstra = sp.load_npz('data/{}/{}_{}_dijkstra.npz'.format(dataset_str, dataset_str, dijkstra_k))
    dijkstra = dijkstra.toarray()
    if type(dijkstra) is not np.ndarray:
        dijkstra = dijkstra.numpy()
    graph_type = 'undirected'
    max_iters = 100
    epsilon = 1e-6
    input_graph = os.path.join('data', dataset_str, 'out1_graph_edges_rwr.txt')
    ri_all = []
    for seed in range(len(dijkstra)):
        rwr = RWR()
        rwr.read_graph(input_graph, graph_type)
        r = rwr.compute(seed, restart_rate, epsilon, max_iters,True,True,in_out_peak,in_out_ratio)
        index_i = np.where((dijkstra[seed] <= dijkstra_k) & (dijkstra[seed] > 0))
        #r = r[index_i]
        for i in range(len(dijkstra)):
            if i not in index_i[0]:
                r[i]=0
        ri_all.append(r)
    structural_info = structural_interactiont(ri_all)
    structural_info = sp.csr_matrix(structural_info)
    sp.save_npz(
        'data/{}/{}_structural_{}_{}_{}_{}.npz'.format(dataset_str, dataset_str, in_out_ratio, restart_rate, in_out_peak,
                                                       dijkstra_k), structural_info)
    return structural_info

def structural_interactiont(ri_all):
    g = np.zeros([len(ri_all),len(ri_all)])
    for i in range(len(ri_all)):
        for j in range(len(ri_all)):
            g[i][j] = sum(np.minimum(ri_all[i],ri_all[j])) /sum(np.maximum(ri_all[i],ri_all[j]))
    return g


def structural_interaction(ri_index, ri_all, directed):
    """structural interaction between the structural fingerprints for citeseer"""
    g = np.zeros([len(ri_index),len(ri_index)])
    for i in range(0, len(ri_index)):
        if directed:
            start = 0
        else:
            start = i
        a = ri_index[i].tolist()
        for idxj in range(start,len(ri_index[i])):
            j = ri_index[i][idxj]
            intersection = set(ri_index[i]).intersection(set(ri_index[j]))
            union = set(ri_index[i]).union(set(ri_index[j]))
            intersection = list(intersection)
            union = list(union)
            union_ri_alli = []
            union_ri_allj = []
            b = ri_index[j].tolist()
            if not len(intersection) == 0:
                k_min = []
                k_max = []
                for k in range(len(intersection)):
                    ta = ri_all[i][a.index(intersection[k])]
                    tb = ri_all[j][b.index(intersection[k])]
                    if ta > tb:
                        k_min.append(tb)
                        k_max.append(ta)
                    else:
                        k_min.append(ta)
                        k_max.append(tb)
                union_rest = set(union).difference(set(intersection))
                union_rest = list(union_rest)
                for k in range(len(union_rest)):
                    if union_rest[k] in ri_index[i]:
                        union_ri_alli.append(ri_all[i][a.index(union_rest[k])])
                    else:
                        union_ri_allj.append(ri_all[j][b.index(union_rest[k])])
                union_ri_allj = k_max + union_ri_allj
                union_num = np.sum(np.array(union_ri_allj), axis=0)
                inter_num = np.sum(np.array(k_min), axis=0)
                if len(union_ri_allj) != 0 and union_num != 0:
                    g[i][j] = inter_num / union_num
                    if not directed:
                        g[j][i] = inter_num / union_num
        if i%1000 == 0:
            print(i)
    return g


def generate_dijkstra(dataset_str, k):
    # """Load data."""
    if dataset_str in {'cora', 'citeseer', 'pubmed'}:
        fr = open("data/{}/ind.{}.graph".format(dataset_str,dataset_str), 'rb')
        graph = pickle.load(fr)
        fr.close()
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        adj=adj.astype(np.float32)
        adj = normalize_adj(adj + sp.eye(adj.shape[0]))
        adj = torch.FloatTensor(np.array(adj.todense()))
        dijkstra = np.zeros([len(adj), len(adj)])
        marked_adj = np.zeros([len(adj), len(adj)])
        inf = graph
        # caculate n-hop neighbors
        g = Graph(len(marked_adj),inf)

        for i in range(len(inf)):
            for j in range(len(inf[i])):
                g.addEdge(i, inf[i][j], 1)
        for i in range(0, len(adj)):
            # make the in edge not devided by 1
            res = g.findLkNeighboursUndir(i, k)
            for idx in range(0, k):
                for j in res[idx]:
                    marked_adj[i, j] = 1
                    dijkstra[i][j] = (idx + 1)
        marked_adj = sp.csr_matrix(marked_adj)
        dijkstra = sp.csr_matrix(dijkstra)
        sp.save_npz('data/{}/{}_{}_dijkstra.npz'.format(dataset_str,dataset_str,k),dijkstra)
        sp.save_npz('data/{}/{}_{}_marked_adj.npz'.format(dataset_str,dataset_str,k),marked_adj)
    else:
        graph_adjacency_list_file_path = os.path.join('data', dataset_str, 'out1_graph_edges.txt')
        graph_node_features_and_labels_file_path = os.path.join('data', dataset_str, 'out1_node_feature_label.txt')
        graph = nx.DiGraph()
        graph_node_features_dict = {}
        graph_labels_dict = {}

        if dataset_str == 'film':
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                    feature_blank = np.zeros(932, dtype=np.uint8)
                    feature_blank[np.array(line[1].split(','), dtype=np.uint16)] = 1
                    graph_node_features_dict[int(line[0])] = feature_blank
                    graph_labels_dict[int(line[0])] = int(line[2])
        else:
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                    if len(line[1]) > 0:
                        graph_node_features_dict[int(line[0])] = np.array(line[1].split(','), dtype=np.uint8)
                    graph_labels_dict[int(line[0])] = int(line[2])

        inf = defaultdict(list)
        for i in range(0,len(graph_labels_dict)):
            inf[i].__init__()
        with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
            graph_adjacency_list_file.readline()
            for line in graph_adjacency_list_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 2)
                if int(line[0]) not in graph  and int(line[0]) in graph_node_features_dict.keys():
                    graph.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                               label=graph_labels_dict[int(line[0])])
                if int(line[1]) not in graph  and int(line[1]) in graph_node_features_dict.keys():
                    graph.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                               label=graph_labels_dict[int(line[1])])
                graph.add_edge(int(line[0]), int(line[1]))
                inf[int(line[0])].append(int(line[1]))
        adj = nx.adjacency_matrix(graph, sorted(graph.nodes()))
        adj=adj.astype(np.float32)
        adj = normalize_adj(adj + sp.eye(adj.shape[0]))
        adj = torch.FloatTensor(np.array(adj.todense()))
        dijkstra = np.zeros([len(adj), len(adj)])
        marked_adj = np.zeros([len(adj), len(adj)])
        inf = graph
        # caculate n-hop neighbors
        g = Graph(len(marked_adj),inf)
        for i in range(len(inf)):
            tmpInf = list(inf[i])
            for j in range(len(tmpInf)):
                g.addEdge(i, tmpInf[j], 1)
                g.addEdge(tmpInf[j], i, 1)
        for i in range(0, len(adj)):
            # make the in edge not devided by 1
            res, marked = g.findLkNeighboursDir(i, k)
            for idx in range(0, k):
                for j in res[idx]:
                    dijkstra[i][j] = (idx + 1)
            for j in range(0, len(adj)):
                marked_adj[i][j] = marked[j]
        marked_adj = sp.csr_matrix(marked_adj)
        dijkstra = sp.csr_matrix(dijkstra)
        sp.save_npz('data/{}/{}_{}_dijkstra.npz'.format(dataset_str,dataset_str,k),dijkstra)
        sp.save_npz('data/{}/{}_{}_marked_adj.npz'.format(dataset_str,dataset_str,k),marked_adj)

def compute_structural_info(dataset_str, origin_adj, directed, dijkstra_k, in_out_ratio, restart_rate, in_out_peak):
    dijkstra = sp.load_npz('data/{}/{}_{}_dijkstra.npz'.format(dataset_str,dataset_str,dijkstra_k))
    marked_adj = sp.load_npz('data/{}/{}_{}_marked_adj.npz'.format(dataset_str,dataset_str,dijkstra_k))
    dijkstra = dijkstra.toarray()
    marked_adj=marked_adj.toarray()
    if type(dijkstra) is not np.ndarray:
        dijkstra = dijkstra.numpy()
    ri_all = []
    ri_index = []
    # You may replace 3327 with the size of dataset
    in_edge_sum = origin_adj.sum(axis=0)
    out_edge_sum = origin_adj.sum(axis=1)
    for i in range(len(origin_adj)):
        # You may replace 1,4 with the .n-hop neighbors you want
        index_i = np.where((dijkstra[i] <= dijkstra_k) & (dijkstra[i] > 0))
        I = np.eye((len(index_i[0]) + 1), dtype=int)
        ei = []
        if in_edge_sum[i] + out_edge_sum[i] < 10:
            in_edge_weight = 0.0
        else:
            in_edge_weight = (out_edge_sum[i] - in_edge_sum[i]) / (in_edge_sum[i] + out_edge_sum[i])  # [-1 , 1]
            in_edge_weight = in_out_peak * in_edge_weight ** in_out_ratio
        for q in range((len(index_i[0]) + 1)):
            if q == 0:
                ei.append([1])
            else:
                ei.append([0])
        W = []
        for j in range((len(index_i[0])) + 1):
            w = []
            for k in range((len(index_i[0])) + 1):
                if j == 0:
                    if k == 0:
                        w.append(float(0))
                    else:
                        if marked_adj[i, index_i[0][k - 1]] == 1:
                            #w.append(float(1 + in_edge_weight))
                            w.append(1 / dijkstra[i][index_i[0][k - 1]] * float(1 + in_edge_weight))
                        elif marked_adj[i, index_i[0][k - 1]] == -1:
                            w.append(1 / dijkstra[i][index_i[0][k - 1]] * float(1 - in_edge_weight))
                        else: #mutual 2
                            #w.append(1 / dijkstra[i][index_i[0][k - 1]] * float(1 + abs(in_edge_weight)))
                            w.append(1 / dijkstra[i][index_i[0][k - 1]] * 2)
                        # w.append(float(1))
                else:
                    if k == 0:
                        if marked_adj[i, index_i[0][k - 1]] == 1:
                            w.append(1 / dijkstra[i][index_i[0][k - 1]] * float(1 + in_edge_weight))
                        elif marked_adj[i, index_i[0][k - 1]] == -1:
                            w.append(1 / dijkstra[i][index_i[0][k - 1]] * float(1 - in_edge_weight))
                        else: #mutual 2
                            #w.append(1 / dijkstra[i][index_i[0][k - 1]] * float(1 + abs(in_edge_weight)))
                            w.append(1 / dijkstra[i][index_i[0][k - 1]] * 2)
                        # w.append(float(1))
                    else:
                        w.append(float(0))
            W.append(w)

        W = np.array(W)
        if len(W) > 1:
            W = W / np.linalg.norm(W)
        rw_left = (I - restart_rate * W)
        try:
            rw_left = np.linalg.inv(rw_left)
        except:
            rw_left = rw_left
        else:
            rw_left = rw_left
        ei = np.array(ei)
        rw_left = torch.tensor(rw_left, dtype=torch.float32)
        ei = torch.tensor(ei, dtype=torch.float32)
        ri = torch.mm(rw_left, ei)
        ri = torch.transpose(ri, 1, 0)
        ri = abs(ri[0]).numpy().tolist()
        ri_index.append(np.concatenate(([i], index_i[0])))
        ri_all.append(ri)
    # Evaluate structural interaction between the structural fingerprints of node i and j
    structural_info = structural_interaction(ri_index, ri_all, directed)
    structural_info = sp.csr_matrix(structural_info)
    sp.save_npz('data/{}/{}_structural_{}_{}_{}_{}.npz'.format(dataset_str, dataset_str,in_out_ratio,restart_rate,in_out_peak,dijkstra_k),structural_info)
    return structural_info


def load_adj(dataset_str):
    graph_adjacency_list_file_path = os.path.join('data', dataset_str, 'out1_graph_edges.txt')
    graph_node_features_and_labels_file_path = os.path.join('data', dataset_str, 'out1_node_feature_label.txt')
    graph = nx.DiGraph()
    graph_node_features_dict = {}
    graph_labels_dict = {}

    if dataset_str == 'film':
        with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
            graph_node_features_and_labels_file.readline()
            for line in graph_node_features_and_labels_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 3)
                assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                feature_blank = np.zeros(932, dtype=np.uint8)
                feature_blank[np.array(line[1].split(','), dtype=np.uint16)] = 1
                graph_node_features_dict[int(line[0])] = feature_blank
                graph_labels_dict[int(line[0])] = int(line[2])
    else:
        with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
            graph_node_features_and_labels_file.readline()
            for line in graph_node_features_and_labels_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 3)
                assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                if len(line[1])>0:
                    graph_node_features_dict[int(line[0])] = np.array(line[1].split(','), dtype=np.uint8)
                else:
                    graph_node_features_dict[int(line[0])] = []
                graph_labels_dict[int(line[0])] = int(line[2])

    inf = defaultdict(list)
    for i in range(0,len(graph_labels_dict)):
        inf[i].__init__()
    with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
        graph_adjacency_list_file.readline()
        for line in graph_adjacency_list_file:
            line = line.rstrip().split('\t')
            assert (len(line) == 2)
            if int(line[0]) not in graph:
                graph.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                           label=graph_labels_dict[int(line[0])])
            if int(line[1]) not in graph:
                graph.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                           label=graph_labels_dict[int(line[1])])
            graph.add_edge(int(line[0]), int(line[1]))
            inf[int(line[0])].append(int(line[1]))
    adj = nx.adjacency_matrix(graph, sorted(graph.nodes()))
    return adj.toarray()

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

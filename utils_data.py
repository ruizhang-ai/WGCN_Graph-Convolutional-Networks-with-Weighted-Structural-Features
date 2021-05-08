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

# Adapted from https://github.com/graphdml-uiuc-jlu/geom-gcn
import os
import re
import json
import random
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch as th
from dgl import DGLGraph
from sklearn.model_selection import ShuffleSplit
import utils_basic


def load_data_custome(dataset_name, dijskra_k, nfold, splits_file_path=None, num_eachclass=None,
                      val_number=None, embedding_mode=None,
              embedding_method=None, structural_info=None,latent=True, ng=False, size_patience=None,
              embedding_method_graph=None, embedding_method_space=None):

    graph_adjacency_list_file_path = os.path.join('data', dataset_name, 'out1_graph_edges.txt')
    graph_node_features_and_labels_file_path = os.path.join('data', dataset_name,
                                                            'out1_node_feature_label.txt')
    G = nx.DiGraph()
    graph_node_features_dict = {}
    graph_labels_dict = {}

    if dataset_name == 'cora_ml' or dataset_name == 'citeseer_ml':
        with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
            header = graph_node_features_and_labels_file.readline()
            feature_dim = int((header.rstrip().split('	'))[4])
            for line in graph_node_features_and_labels_file:
                line = line.rstrip().split('	')
                assert (len(line) == 3)
                assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                features = np.zeros(feature_dim, dtype=np.uint8)
                if len(line[1]) > 0:
                    values = np.array(line[1].split(','), dtype=np.uint8)
                    for i in range(len(values)):
                        features[values[i]] = 1
                graph_node_features_dict[int(line[0])] = features
                graph_labels_dict[int(line[0])] = int(line[2])
    else:
        with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
            header = graph_node_features_and_labels_file.readline()
            feature_dim = int((header.rstrip().split('	'))[4])
            for line in graph_node_features_and_labels_file:
                line = line.rstrip().split('	')
                assert (len(line) == 3)
                assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                features = np.zeros(feature_dim, dtype=np.uint8)
                if len(line[1]) > 0:
                    values = np.array(line[1].split(','), dtype=np.uint8)
                    for i in range(len(values)):
                        features[values[i]] = 1
                graph_node_features_dict[int(line[0])] = features
                graph_labels_dict[int(line[0])] = int(line[2])

    with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
        graph_adjacency_list_file.readline()
        for line in graph_adjacency_list_file:
            line = line.rstrip().split('\t')
            assert (len(line) == 2)
            if int(line[0]) not in G:
                G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                           label=graph_labels_dict[int(line[0])])
            if int(line[1]) not in G:
                G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                           label=graph_labels_dict[int(line[1])])
            G.add_edge(int(line[0]), int(line[1]))

    adj = nx.adjacency_matrix(G, sorted(G.nodes()))
    features = np.array(
        [features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
    labels = np.array(
        [label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])


    if not embedding_mode:
        g = DGLGraph(adj + sp.eye(adj.shape[0]))
    else:
        marked_adj = sp.load_npz('data/{}/{}_{}_marked_adj.npz'.format(dataset_name, dataset_name,dijskra_k))
        if embedding_mode == 'ExperimentTwoAll':
            embedding_file_path = os.path.join('embedding_method_combinations_all','outf_nodes_relation_{}all_embedding_methods.txt'.format(dataset_name))
        elif embedding_mode == 'ExperimentTwoPairs':
            embedding_file_path = os.path.join('embedding_method_combinations_in_pairs','outf_nodes_relation_{}_graph_{}_space_{}.txt'.format(dataset_name,embedding_method_graph,embedding_method_space))
        else:
            embedding_file_path = os.path.join('structural_neighborhood', 'space_relation_{}_{}.txt'.format(dataset_name,embedding_method))
        space_and_relation_type_to_idx_dict = {}

        space_and_relation_type_to_idx_dict['self_loop'] = len(space_and_relation_type_to_idx_dict)
        for node in sorted(G.nodes()):
            if G.has_edge(node, node):
                G.remove_edge(node, node)
            G.add_edge(node, node, subgraph_idx=space_and_relation_type_to_idx_dict['self_loop'])

        with open(embedding_file_path) as embedding_file:
            for line in embedding_file:
                if line.rstrip() == 'node1,node2	space	relation_type':
                    continue
                line = re.split(r'[\t,]', line.rstrip())
                assert (len(line) == 4)
                assert (int(line[0]) in G and int(line[1]) in G)
                if line[2] == 'graph':
                    if marked_adj[int(line[0]),int(line[1])] == 2:
                        '''
                        line[2] = 'graph_in'
                        if (line[2], int(line[3])) not in space_and_relation_type_to_idx_dict:
                            space_and_relation_type_to_idx_dict[(line[2], int(line[3]))] = len(
                            space_and_relation_type_to_idx_dict)
                        G.add_edge(int(line[0]), int(line[1]), subgraph_idx=space_and_relation_type_to_idx_dict[
                            (line[2], int(line[3]))])
                        '''
                        line[2] = 'graph_out'
                        if (line[2], int(line[3])) not in space_and_relation_type_to_idx_dict:
                            space_and_relation_type_to_idx_dict[(line[2], int(line[3]))] = len(
                            space_and_relation_type_to_idx_dict)
                        if G.has_edge(int(line[0]), int(line[1])):
                            G.remove_edge(int(line[0]), int(line[1]))
                        G.add_edge(int(line[0]), int(line[1]), subgraph_idx=space_and_relation_type_to_idx_dict[
                            (line[2], int(line[3]))])
                    elif marked_adj[int(line[0]),int(line[1])] == 1:
                        line[2] = 'graph_in'
                        if (line[2], int(line[3])) not in space_and_relation_type_to_idx_dict:
                            space_and_relation_type_to_idx_dict[(line[2], int(line[3]))] = len(
                            space_and_relation_type_to_idx_dict)
                        if G.has_edge(int(line[0]), int(line[1])):
                            G.remove_edge(int(line[0]), int(line[1]))
                        G.add_edge(int(line[0]), int(line[1]), subgraph_idx=space_and_relation_type_to_idx_dict[
                            (line[2], int(line[3]))])
                    else:
                        line[2] = 'graph_out'
                        if (line[2], int(line[3])) not in space_and_relation_type_to_idx_dict:
                            space_and_relation_type_to_idx_dict[(line[2], int(line[3]))] = len(
                            space_and_relation_type_to_idx_dict)
                        if G.has_edge(int(line[0]), int(line[1])):
                            G.remove_edge(int(line[0]), int(line[1]))
                        G.add_edge(int(line[0]), int(line[1]), subgraph_idx=space_and_relation_type_to_idx_dict[
                            (line[2], int(line[3]))])
                else:
                    continue
                    if (line[2], int(line[3])) not in space_and_relation_type_to_idx_dict:
                        space_and_relation_type_to_idx_dict[(line[2], int(line[3]))] = len(
                            space_and_relation_type_to_idx_dict)
                    if G.has_edge(int(line[0]), int(line[1])):
                        G.remove_edge(int(line[0]), int(line[1]))
                    G.add_edge(int(line[0]), int(line[1]), subgraph_idx=space_and_relation_type_to_idx_dict[
                        (line[2], int(line[3]))])
        pairs = []
        for i in range(nfold):
            if ("graph_in", i) in space_and_relation_type_to_idx_dict and ("graph_out", i) in space_and_relation_type_to_idx_dict:
                if space_and_relation_type_to_idx_dict[("graph_in", i)] > space_and_relation_type_to_idx_dict[("graph_out", i)]:
                    pairs.append([space_and_relation_type_to_idx_dict[("graph_out", i)], space_and_relation_type_to_idx_dict[("graph_in", i)]])
                else:
                    pairs.append([space_and_relation_type_to_idx_dict[("graph_in", i)], space_and_relation_type_to_idx_dict[("graph_out", i)]])

        adj = nx.adjacency_matrix(G, sorted(G.nodes()))
        g = DGLGraph(adj)

        #max = torch.max(torch.tensor(structural_info))
        #tmp_structural_info = utils.preprocess_features(structural_info)
        for u, v, feature in G.edges(data='subgraph_idx'):
            g.edges[g.edge_id(u, v)].data['subgraph_idx'] = th.tensor([feature])
            #a = tmp_structural_info[u,v]/max + 1
            #g.edges[g.edge_id(u, v)].data['struc_info'] = th.tensor([a])


    if dataset_name in {'cora_ml', 'citeseer_ml', 'cora_all'}:
        disconnected_node_file_path = os.path.join('unconnected_nodes', '{}_unconnected_nodes.txt'.format(dataset_name))
        with open(disconnected_node_file_path) as disconnected_node_file:
            disconnected_node_file.readline()
            disconnected_nodes = []
            for line in disconnected_node_file:
                line = line.rstrip()
                disconnected_nodes.append(int(line))

        disconnected_nodes = np.array(disconnected_nodes)
        connected_nodes = np.setdiff1d(np.arange(features.shape[0]), disconnected_nodes)

        random.shuffle(connected_nodes)

        connected_labels = labels[connected_nodes]

        train_and_val_index, test_index = next(
            ShuffleSplit(n_splits=1, train_size=0.01).split(
                np.empty_like(connected_labels), connected_labels))

    else:
        train_and_val_index, test_index = next(
        ShuffleSplit(n_splits=1, train_size=0.01).split(
            np.empty_like(labels), labels))

    train_index = []
    val_index = []

    train_mask = np.zeros_like(labels)
    val_mask = np.zeros_like(labels)
    test_mask = np.zeros_like(labels)

    i = 0
    while i < len(test_index):
        tmpLabel = labels[train_index]
        tmpLabel = tmpLabel.tolist()
        if tmpLabel.count(labels[test_index[i]])<num_eachclass:
            train_index.append(test_index[i])
            test_index = np.delete(test_index,i)
        elif len(val_index)<val_number:
            val_index.append((test_index[i]))
            test_index = np.delete(test_index,i)
        else:
            i = i+1
    assert (np.array_equal(np.unique(labels), np.arange(len(np.unique(labels)))))
    for i in train_index:
        train_mask[i] = 1
    for i in val_index:
        val_mask[i] = 1
    for i in test_index:
        test_mask[i] = 1

    num_features = features.shape[1]
    num_labels = len(np.unique(labels))
    assert (np.array_equal(np.unique(labels), np.arange(len(np.unique(labels)))))

    features = th.FloatTensor(features)
    labels = th.LongTensor(labels)
    train_mask = th.BoolTensor(train_mask)
    val_mask = th.BoolTensor(val_mask)
    test_mask = th.BoolTensor(test_mask)

    degs = g.in_degrees().float()
    norm = th.pow(degs, -0.5).cuda()
    norm[th.isinf(norm)] = 0
    g.ndata['norm'] = norm.unsqueeze(1)

    return g, features, labels, train_mask, val_mask, test_mask, num_features, num_labels, len(
        space_and_relation_type_to_idx_dict), pairs


def load_data(dataset_name, dijskra_k, nfold, splits_file_path=None, train_percentage=None, val_percentage=None,
              embedding_mode=None,
              embedding_method=None, structural_info=None, latent=True, ng=False, size_patience=None,
              embedding_method_graph=None, embedding_method_space=None):
    graph_adjacency_list_file_path = os.path.join('data', dataset_name, 'out1_graph_edges.txt')
    graph_node_features_and_labels_file_path = os.path.join('data', dataset_name, 'out1_node_feature_label.txt')
    G = nx.DiGraph()
    graph_node_features_dict = {}
    graph_labels_dict = {}

    with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
        header = graph_node_features_and_labels_file.readline()
        feature_dim = int((header.rstrip().split('\t'))[4])
        for line in graph_node_features_and_labels_file:
            line = line.rstrip().split('\t')
            assert (len(line) == 3)
            assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
            features = np.zeros(feature_dim, dtype=np.uint8)
            if len(line[1]) > 0:
                values = np.array(line[1].split(','), dtype=np.uint8)
                for i in range(len(values)):
                    features[values[i]] = 1
            graph_node_features_dict[int(line[0])] = features
            graph_labels_dict[int(line[0])] = int(line[2])

    with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
        graph_adjacency_list_file.readline()
        for line in graph_adjacency_list_file:
            line = line.rstrip().split('\t')
            assert (len(line) == 2)
            if int(line[0]) not in G:
                G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                           label=graph_labels_dict[int(line[0])])
            if int(line[1]) not in G:
                G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                           label=graph_labels_dict[int(line[1])])
            G.add_edge(int(line[0]), int(line[1]))

    adj = nx.adjacency_matrix(G, sorted(G.nodes()))
    features = np.array(
        [features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
    labels = np.array(
        [label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])

    if not embedding_mode:
        g = DGLGraph(adj + sp.eye(adj.shape[0]))
    else:
        marked_adj = sp.load_npz('data/{}/{}_{}_marked_adj.npz'.format(dataset_name, dataset_name, dijskra_k))
        if latent:
            embedding_file_path = os.path.join('structural_neighborhood_l',
                                               'space_relation_{}_{}.txt'.format(dataset_name, embedding_method))
        else:
            embedding_file_path = os.path.join('structural_neighborhood',
                                               'space_relation_{}_{}.txt'.format(dataset_name, embedding_method))
        if ng:
            embedding_file_path = os.path.join('structural_neighborhood_ng',
                                               'space_relation_{}_{}.txt'.format(dataset_name, embedding_method))
        space_and_relation_type_to_idx_dict = {}
        space_and_relation_type_to_idx_dict['self_loop'] = len(space_and_relation_type_to_idx_dict)
        for node in sorted(G.nodes()):
            if G.has_edge(node, node):
                G.remove_edge(node, node)
            G.add_edge(node, node, subgraph_idx=space_and_relation_type_to_idx_dict['self_loop'])

        with open(embedding_file_path) as embedding_file:
            for line in embedding_file:
                if line.rstrip() == 'node1,node2	space	relation_type':
                    continue
                line = re.split(r'[\t,]', line.rstrip())
                assert (len(line) == 4)
                assert (int(line[0]) in G and int(line[1]) in G)
                if line[2] == 'graph':
                    if marked_adj[int(line[0]), int(line[1])] == 2:
                        line[2] = 'graph_out'
                        if (line[2], int(line[3])) not in space_and_relation_type_to_idx_dict:
                            space_and_relation_type_to_idx_dict[(line[2], int(line[3]))] = len(
                                space_and_relation_type_to_idx_dict)
                        if G.has_edge(int(line[0]), int(line[1])):
                            G.remove_edge(int(line[0]), int(line[1]))
                        G.add_edge(int(line[0]), int(line[1]), subgraph_idx=space_and_relation_type_to_idx_dict[
                            (line[2], int(line[3]))])
                    elif marked_adj[int(line[0]), int(line[1])] == 1:
                        line[2] = 'graph_in'
                        if (line[2], int(line[3])) not in space_and_relation_type_to_idx_dict:
                            space_and_relation_type_to_idx_dict[(line[2], int(line[3]))] = len(
                                space_and_relation_type_to_idx_dict)
                        if G.has_edge(int(line[0]), int(line[1])):
                            G.remove_edge(int(line[0]), int(line[1]))
                        G.add_edge(int(line[0]), int(line[1]), subgraph_idx=space_and_relation_type_to_idx_dict[
                            (line[2], int(line[3]))])
                    else:
                        line[2] = 'graph_out'
                        if (line[2], int(line[3])) not in space_and_relation_type_to_idx_dict:
                            space_and_relation_type_to_idx_dict[(line[2], int(line[3]))] = len(
                                space_and_relation_type_to_idx_dict)
                        if G.has_edge(int(line[0]), int(line[1])):
                            G.remove_edge(int(line[0]), int(line[1]))
                        G.add_edge(int(line[0]), int(line[1]), subgraph_idx=space_and_relation_type_to_idx_dict[
                            (line[2], int(line[3]))])
                else:
                    continue
                    if (line[2], int(line[3])) not in space_and_relation_type_to_idx_dict:
                        space_and_relation_type_to_idx_dict[(line[2], int(line[3]))] = len(
                            space_and_relation_type_to_idx_dict)
                    if G.has_edge(int(line[0]), int(line[1])):
                        G.remove_edge(int(line[0]), int(line[1]))
                    G.add_edge(int(line[0]), int(line[1]), subgraph_idx=space_and_relation_type_to_idx_dict[
                        (line[2], int(line[3]))])
        pairs = []
        for i in range(nfold):
            if ("graph_in", i) in space_and_relation_type_to_idx_dict and (
            "graph_out", i) in space_and_relation_type_to_idx_dict:
                if space_and_relation_type_to_idx_dict[("graph_in", i)] > space_and_relation_type_to_idx_dict[
                    ("graph_out", i)]:
                    pairs.append([space_and_relation_type_to_idx_dict[("graph_out", i)],
                                  space_and_relation_type_to_idx_dict[("graph_in", i)]])
                else:
                    pairs.append([space_and_relation_type_to_idx_dict[("graph_in", i)],
                                  space_and_relation_type_to_idx_dict[("graph_out", i)]])

        adj = nx.adjacency_matrix(G, sorted(G.nodes()))
        g = DGLGraph(adj)
        for u, v, feature in G.edges(data='subgraph_idx'):
            g.edges[g.edge_id(u, v)].data['subgraph_idx'] = th.tensor([feature])

    features = utils_basic.preprocess_features(features)
    structural_info = utils_basic.preprocess_features(structural_info)
    features = np.concatenate([np.array(features), structural_info], axis=1)

    if splits_file_path:
        with np.load(splits_file_path) as splits_file:
            train_mask = splits_file['train_mask']
            val_mask = splits_file['val_mask']
            test_mask = splits_file['test_mask']
    else:
        assert (train_percentage is not None and val_percentage is not None)
        assert (train_percentage < 1.0 and val_percentage < 1.0 and train_percentage + val_percentage < 1.0)
        if dataset_name in {'cora_ml', 'citeseer_ml', 'cora_tt'}:

            disconnected_node_file_path = os.path.join('unconnected_nodes',
                                                       '{}_unconnected_nodes.txt'.format(dataset_name))
            with open(disconnected_node_file_path) as disconnected_node_file:
                disconnected_node_file.readline()
                disconnected_nodes = []
                for line in disconnected_node_file:
                    line = line.rstrip()
                    disconnected_nodes.append(int(line))

            disconnected_nodes = np.array(disconnected_nodes)
            connected_nodes = np.setdiff1d(np.arange(features.shape[0]), disconnected_nodes)

            random.shuffle(connected_nodes)

            connected_labels = labels[connected_nodes]


            #if dataset_name in {'cora_ml', 'cora_tt'}:
            #    train_percentage = 140/len(connected_nodes)#
            #else:
            #    train_percentage = 120 / len(connected_nodes)  #
            #val_percentage = 500/len(connected_nodes)#


            train_and_val_index, test_index = next(
                ShuffleSplit(n_splits=1, train_size=train_percentage + val_percentage).split(
                    np.empty_like(connected_labels), connected_labels))
            train_index, val_index = next(
                ShuffleSplit(n_splits=1, train_size=train_percentage / (train_percentage + val_percentage)).split(
                    np.empty_like(connected_labels[train_and_val_index]), connected_labels[train_and_val_index]))
            train_index = train_and_val_index[train_index]
            val_index = train_and_val_index[val_index]

            #test_index = test_index[0:1000]#


            train_mask = np.zeros_like(labels)
            train_mask[connected_nodes[train_index]] = 1
            val_mask = np.zeros_like(labels)
            val_mask[connected_nodes[val_index]] = 1
            test_mask = np.zeros_like(labels)
            test_mask[connected_nodes[test_index]] = 1
        else:
            train_and_val_index, test_index = next(
                ShuffleSplit(n_splits=1, train_size=train_percentage + val_percentage).split(
                    np.empty_like(labels), labels))
            train_index, val_index = next(ShuffleSplit(n_splits=1, train_size=train_percentage).split(
                np.empty_like(labels[train_and_val_index]), labels[train_and_val_index]))
            train_index = train_and_val_index[train_index]
            val_index = train_and_val_index[val_index]

            train_mask = np.zeros_like(labels)
            train_mask[train_index] = 1
            val_mask = np.zeros_like(labels)
            val_mask[val_index] = 1
            test_mask = np.zeros_like(labels)
            test_mask[test_index] = 1

    num_features = features.shape[1]
    num_labels = len(np.unique(labels))
    assert (np.array_equal(np.unique(labels), np.arange(len(np.unique(labels)))))

    features = th.FloatTensor(features)
    labels = th.LongTensor(labels)
    train_mask = th.BoolTensor(train_mask)
    val_mask = th.BoolTensor(val_mask)
    test_mask = th.BoolTensor(test_mask)

    degs = g.in_degrees().float()
    norm = th.pow(degs, -0.5).cuda()
    norm[th.isinf(norm)] = 0
    g.ndata['norm'] = norm.unsqueeze(1)

    return g, features, labels, train_mask, val_mask, test_mask, num_features, num_labels, len(
        space_and_relation_type_to_idx_dict), pairs

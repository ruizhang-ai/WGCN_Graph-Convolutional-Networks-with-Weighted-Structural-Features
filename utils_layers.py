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

import dgl.function as fn
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
# Adapted from https://docs.dgl.ai/tutorials/models/1_gnn/1_gcn.html
# Adapted from https://github.com/graphdml-uiuc-jlu/geom-gcn

class WGCNSingleChannel(nn.Module):
    def __init__(self, g, in_feats, out_feats, num_divisions, pairs, activation, dropout_prob, merge, isFinal, attention):
        super(WGCNSingleChannel, self).__init__()
        if isFinal:
            self.num_divisions = num_divisions - len(pairs)
        else:
            self.num_divisions = num_divisions
        self.pairs = pairs
        self.in_feats_dropout = nn.Dropout(dropout_prob)

        self.linear_for_each_division = nn.ModuleList()
        for i in range(self.num_divisions):
            self.linear_for_each_division.append(nn.Linear(in_feats, out_feats, bias=False))

        for i in range(self.num_divisions):
            nn.init.xavier_uniform_(self.linear_for_each_division[i].weight)

        self.activation = activation
        self.g = g
        self.subgraph_edge_list_of_list= self.get_subgraphs(self.g, isFinal)
        self.merge = merge
        self.out_feats = out_feats
        self.W_si = nn.Parameter(torch.zeros(size=(1, 1)))
        self.isFinal = isFinal
        nn.init.constant_(self.W_si.data, 0)

        self.attention = attention
        if self.attention and not isFinal:
            self.currentI = 0
            self.atten_linear = nn.Linear(in_feats, out_feats, bias=False)
            self.atten_feats_dropout = nn.Dropout(dropout_prob)
            self.attention_linear = nn.Linear(2 * out_feats, 1, bias=False)
            nn.init.xavier_uniform_(self.attention_linear.weight)
            self.attention_head_dropout = nn.Dropout(dropout_prob)
            self.linear_feats_dropout = nn.Dropout(dropout_prob)

    def calculate_node_pairwise_attention(self, edges):
        h_concat = th.cat([edges.src['Wh_{}'.format(self.currentI)], edges.dst['Wh_{}'.format(self.currentI)]], dim=1)
        e = self.attention_linear(h_concat)
        e = F.leaky_relu(e, negative_slope=0.2)
        return {'e': e}

    def reduce_func(self, nodes):
        a = F.softmax(nodes.mailbox['e_{}'.format(self.currentI)], dim=1)
        a_dropout = self.attention_head_dropout(a)
        Wh_dropout = self.linear_feats_dropout(nodes.mailbox['Wh_{}'.format(self.currentI)])
        return {'h_{}'.format(self.currentI): th.sum(a_dropout * Wh_dropout, dim=1)}

    def message_func(self, edges):
        return {'Wh_{}'.format(self.currentI): edges.src['Wh_{}'.format(self.currentI)], 'e_{}'.format(self.currentI): edges.data['e']}

    def forward(self, feature):
        # here add structural information
        # use weighted sum for node info and str info
        in_feats_dropout = self.in_feats_dropout(feature)
        self.g.ndata['h'] = in_feats_dropout
        for i in range(self.num_divisions):
            subgraph = self.g.edge_subgraph(self.subgraph_edge_list_of_list[i])
            subgraph.copy_from_parent()
            subgraph.ndata['Wh_{}'.format(i)] = self.linear_for_each_division[i](subgraph.ndata['h']) * subgraph.ndata['norm']
            if self.isFinal or not self.attention:
                subgraph.update_all(message_func=fn.copy_u(u='Wh_{}'.format(i), out='m_{}'.format(i)), reduce_func=fn.sum(msg='m_{}'.format(i), out='h_{}'.format(i)))
            else:
                self.currentI = i
                subgraph.apply_edges(self.calculate_node_pairwise_attention)
                subgraph.update_all(self.message_func, self.reduce_func)
            subgraph.ndata.pop('Wh_{}'.format(i))
            subgraph.copy_to_parent()

        self.g.ndata.pop('h')
        results_from_subgraph_list = []
        if self.isFinal:
            for i in range(self.num_divisions):
                if 'h_{}'.format(i) in self.g.node_attr_schemes():
                    results_from_subgraph_list.append(self.g.ndata.pop('h_{}'.format(i)))
                else:
                    results_from_subgraph_list.append(
                        th.zeros((feature.size(0), self.out_feats), dtype=th.float32, device=feature.device))
        else:
            visited = []
            for i in range(self.num_divisions):
                for j in range(len(self.pairs)):
                    if (i == self.pairs[j][0] or i == self.pairs[j][1]) and i not in visited:
                        results_from_subgraph_list.append(th.add(self.g.ndata.pop('h_{}'.format(self.pairs[j][0])),self.g.ndata.pop('h_{}'.format(self.pairs[j][1]))))
                        visited.append(self.pairs[j][1])
                        visited.append(self.pairs[j][0])
                        break
                if i not in visited:
                    results_from_subgraph_list.append(self.g.ndata.pop('h_{}'.format(i)))

        if self.merge == 'cat':
            h_new = th.cat(results_from_subgraph_list, dim=-1)
        else:
            h_new = th.mean(th.stack(results_from_subgraph_list, dim=-1), dim=-1)
        h_new = h_new * self.g.ndata['norm']
        h_new = self.activation(h_new)
        return h_new

    def get_subgraphs(self, g, isFinal):
        subgraph_edge_list = [[] for _ in range(self.num_divisions)]
        u, v, eid = g.all_edges(form='all')
        if isFinal:
            for i in range(g.number_of_edges()):
                isVisited = False
                for j in range(len(self.pairs)):
                    if g.edges[u[i], v[i]].data['subgraph_idx'] == self.pairs[j][1]:
                        subgraph_edge_list[self.pairs[j][0]].append(eid[i])
                        isVisited = True
                if not isVisited:
                    subgraph_edge_list[g.edges[u[i], v[i]].data['subgraph_idx']].append(eid[i])
        else:
            for i in range(g.number_of_edges()):
                subgraph_edge_list[g.edges[u[i], v[i]].data['subgraph_idx']].append(eid[i])
            #subgraph_edge_weight_list[g.edges[u[i], v[i]].data['subgraph_idx']].append(g.edges[u[i], v[i]].data['struc_info'])
        return subgraph_edge_list


class WGCN(nn.Module):
    def __init__(self, g, in_feats, out_feats, num_divisions, pairs, activation, num_heads, dropout_prob, ggcn_merge,
                 channel_merge, isFinal,attention):
        super(WGCN, self).__init__()
        self.attention_heads = nn.ModuleList()
        for _ in range(num_heads):
            self.attention_heads.append(
                WGCNSingleChannel(g, in_feats, out_feats, num_divisions, pairs, activation, dropout_prob, ggcn_merge, isFinal,attention))
        self.channel_merge = channel_merge
        self.g = g
        self.isFinal = isFinal

    def forward(self, feature):
        all_attention_head_outputs = [head( feature) for head in self.attention_heads]
        if self.channel_merge == 'cat':
            return th.cat(all_attention_head_outputs, dim=1)
        else:
            return th.mean(th.stack(all_attention_head_outputs), dim=0)


class WGCNNet(nn.Module):
    def __init__(self, g, num_input_features, num_output_classes, num_hidden, num_divisions, pairs, num_heads_layer_one,
                 num_heads_layer_two,
                 dropout_rate, layer_one_ggcn_merge, layer_one_channel_merge, layer_two_ggcn_merge,
                 layer_two_channel_merge,attention,layers):
        super(WGCNNet, self).__init__()
        #self.num_structural_features = (int)(num_structural_features/num_divisions)

        #self.wgcn0 = WGCN(g, num_input_features, self.num_structural_features, num_divisions, F.relu, 1, dropout_rate, layer_one_ggcn_merge, layer_one_channel_merge)
        isFinal = False
        self.wgcn1 = WGCN(g, num_input_features, num_hidden, num_divisions, pairs, F.relu, num_heads_layer_one,
                                dropout_rate,
                                layer_one_ggcn_merge, layer_one_channel_merge, isFinal,attention)

        if layer_one_ggcn_merge == 'cat':
            layer_one_ggcn_merge_multiplier = num_divisions-len(pairs)
        else:
            layer_one_ggcn_merge_multiplier = 1

        if layer_one_channel_merge == 'cat':
            layer_one_channel_merge_multiplier = num_heads_layer_one
        else:
            layer_one_channel_merge_multiplier = 1

        isFinal = True
        self.wgcnM = WGCN(g, num_hidden * layer_one_ggcn_merge_multiplier * layer_one_channel_merge_multiplier,
                          num_hidden * layer_one_ggcn_merge_multiplier * layer_one_channel_merge_multiplier, num_divisions, pairs, lambda x: x,
                          num_heads_layer_two, dropout_rate, layer_two_ggcn_merge, layer_two_channel_merge, isFinal,
                          attention)
        isFinal = False
        self.wgcn2 = WGCN(g, num_hidden * layer_one_ggcn_merge_multiplier * num_heads_layer_one, num_hidden, num_divisions, pairs,
                          F.relu, num_heads_layer_one,
                          dropout_rate,
                          layer_one_ggcn_merge, layer_one_channel_merge, isFinal, attention)

        isFinal = True
        self.wgcnM1 = WGCN(g, num_hidden * layer_one_ggcn_merge_multiplier * layer_one_channel_merge_multiplier,
                          num_hidden * layer_one_ggcn_merge_multiplier * layer_one_channel_merge_multiplier, num_divisions, pairs, lambda x: x,
                          num_heads_layer_two, dropout_rate, layer_two_ggcn_merge, layer_two_channel_merge, isFinal,
                          attention)
        isFinal = False
        self.wgcn3 = WGCN(g, num_hidden * layer_one_ggcn_merge_multiplier * num_heads_layer_one, num_hidden, num_divisions, pairs,
                          F.relu, num_heads_layer_one,
                          dropout_rate,
                          layer_one_ggcn_merge, layer_one_channel_merge, isFinal, attention)


        isFinal = True
        self.wgcnF = WGCN(g, num_hidden * layer_one_ggcn_merge_multiplier * layer_one_channel_merge_multiplier,
                                num_output_classes, num_divisions, pairs, lambda x: x,
                                num_heads_layer_two, dropout_rate, layer_two_ggcn_merge, layer_two_channel_merge, isFinal,attention)
        self.g = g
        self.layers = layers

    def forward(self, features):
        x = self.wgcn1(features)
        x = self.wgcnM(x)
        x = self.wgcn2(x)
        x = self.wgcnM1(x)
        x = self.wgcn3(x)
        x = self.wgcnF(x)
        return x


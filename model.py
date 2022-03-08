import torch
import dgl
import dgl.nn.pytorch as dglnn
from torch import nn
from torch.nn import init

class NetModel(torch.nn.Module):

    def __init__(self, num_layer, dim, is_graph, is_hetero, is_dynamic, num_snapshot):
        super(NetModel, self).__init__()
        self.num_layer = num_layer
        self.is_graph = is_graph
        self.is_hetero = is_hetero
        self.is_dynamic = is_dynamic
        self.is_jk = True if is_hetero else False
        self.num_snapshot = num_snapshot

        mods = dict()
        if not self.is_graph:
            dim_node_in = 21
            if not self.is_graph:
                dim_node_in += 4
            for l in range(self.num_layer):
                mods['n' + str(l)] = torch.nn.BatchNorm1d(dim_node_in, track_running_stats=False)
                mods['l' + str(l)] = Perceptron(dim_node_in, dim)
                dim_node_in = dim
        else:
            dim_node_in = 21
            dim_edge_in = 4
            dim_tot = 25
            if not self.is_hetero:
                for l in range(self.num_layer):
                    conv = EGCNConv(dim_node_in, dim_edge_in, dim, dim)
                    conv_dict = dict()
                    conv_dict['ap_ap'] = conv
                    conv_dict['ap_sta'] = conv
                    conv_dict['sta_ap'] = conv
                    mods['nn' + str(l)] = torch.nn.BatchNorm1d(dim_node_in, track_running_stats=False)
                    mods['ne' + str(l)] = torch.nn.BatchNorm1d(dim_edge_in, track_running_stats=False)
                    mods['l' + str(l)] = EHeteroGraphConv(conv_dict)
                    dim_node_in = dim
                    dim_edge_in = dim
                    dim_tot += dim
                mods['comb'] = Perceptron(dim_tot, dim)
            else:
                for l in range(self.num_layer):
                    conv_dict = dict()
                    conv_dict['ap_ap'] = EGATConv(dim_node_in, dim_edge_in, dim, dim, 1)
                    conv_dict['ap_sta'] = EGATConv(dim_node_in, dim_edge_in, dim, dim, 1)
                    conv_dict['sta_ap'] = EGATConv(dim_node_in, dim_edge_in, dim, dim, 1)
                    mods['nn' + str(l)] = torch.nn.BatchNorm1d(dim_node_in, track_running_stats=False)
                    mods['ne' + str(l)] = torch.nn.BatchNorm1d(dim_edge_in, track_running_stats=False)
                    mods['l' + str(l)] = EHeteroGraphConv(conv_dict)
                    dim_node_in = dim
                    dim_edge_in = dim
        if self.is_dynamic:
            # mods['rnn'] = torch.nn.RNN(dim, dim, 2)
            # mods['rnn'] = torch.nn.GRU(dim, dim, 2)
            mods['rnn'] = torch.nn.LSTM(dim, dim, 2)
        mods['predict'] = Perceptron(dim, 1, act=False)
        mods['softplus'] = torch.nn.Softplus()
        self.mods = torch.nn.ModuleDict(mods)

    def forward(self, g):
        # if not self.training:
        #     import pdb; pdb.set_trace()
        if not self.is_graph:
            h = torch.cat([g.nodes['sta'].data['feat'], g.edges['sta_ap'].data['feat']], dim=1)
            for l in range(self.num_layer):
                h = self.mods['n' + str(l)](h)
                h = self.mods['l' + str(l)](h)
        else:
            h = [g.nodes['sta'].data['feat'], g.edges['sta_ap'].data['feat']]
            h_node = {'ap':g.nodes['ap'].data['feat'], 'sta':g.nodes['sta'].data['feat']}
            h_edge = {'ap_ap':g.edges['ap_ap'].data['feat'], 'ap_sta':g.edges['ap_sta'].data['feat'], 'sta_ap':g.edges['sta_ap'].data['feat']}
            for l in range(self.num_layer):
                h_node['ap'] = self.mods['nn' + str(l)](h_node['ap'])
                h_node['sta'] = self.mods['nn' + str(l)](h_node['sta'])
                h_edge['ap_ap'] = self.mods['ne' + str(l)](h_edge['ap_ap'])
                h_edge['ap_sta'] = self.mods['ne' + str(l)](h_edge['ap_sta'])
                h_edge['sta_ap'] = self.mods['ne' + str(l)](h_edge['sta_ap'])
                h_node, h_edge = self.mods['l' + str(l)](g, h_node, h_edge)
                h.append(h_node['sta'])
            h = self.mods['comb'](torch.cat(h, dim=1)) if self.is_jk else h[-1]
        if self.is_dynamic:
            h = h.view((self.num_snapshot, -1, h.shape[-1]))
            h = self.mods['rnn'](h)[0].view(-1, h.shape[-1])
        h = self.mods['predict'](h)
        # to ensure positive prediction
        h = self.mods['softplus'](h)
        return h

class Perceptron(torch.nn.Module):

    def __init__(self, in_dim, out_dim, dropout=0, norm=False, act=True):
        super(Perceptron, self).__init__()
        self.weight = torch.nn.Parameter(torch.empty(in_dim, out_dim))
        torch.nn.init.xavier_uniform_(self.weight.data)
        self.bias = torch.nn.Parameter(torch.empty(out_dim))
        torch.nn.init.zeros_(self.bias.data)
        self.norm = norm
        if norm:
            self.norm = torch.nn.BatchNorm1d(out_dim, eps=1e-9, track_running_stats=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.act = act

    def forward(self, f_in):
        f_in = self.dropout(f_in)
        f_in = torch.mm(f_in, self.weight) + self.bias
        if self.act:
            f_in = torch.nn.functional.relu(f_in)
        if self.norm:
            f_in = self.norm(f_in)
        return f_in

    def reset_parameters():
        torch.nn.init.xavier_uniform_(self.weight.data)
        torch.nn.init.zeros_(self.bias.data)

class EGCNConv(nn.Module):

    def __init__(self,
                 in_node_feats,
                 in_edge_feats,
                 out_node_feats,
                 out_edge_feats,
                 bias=True):

        super().__init__()
        self._out_node_feats = out_node_feats
        self._out_edge_feats = out_edge_feats
        self.fc_node = nn.Linear(in_node_feats, out_node_feats, bias=True)
        self.fc_ni = nn.Linear(in_node_feats, out_edge_feats, bias=False)
        self.fc_fij = nn.Linear(in_edge_feats, out_edge_feats, bias=False)
        self.fc_nj = nn.Linear(in_node_feats, out_edge_feats, bias=False)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(size=(out_edge_feats,)))
        else:
            self.register_buffer('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        gain = init.calculate_gain('relu')
        init.xavier_normal_(self.fc_node.weight, gain=gain)
        init.xavier_normal_(self.fc_ni.weight, gain=gain)
        init.xavier_normal_(self.fc_fij.weight, gain=gain)
        init.xavier_normal_(self.fc_nj.weight, gain=gain)
        init.constant_(self.bias, 0)

    def forward(self, graph, nfeats, efeats):
        
        with graph.local_scope():
            f_ni = self.fc_ni(nfeats)
            f_nj = self.fc_nj(nfeats)
            f_fij = self.fc_fij(efeats)
            graph.srcdata.update({'f_ni': f_ni})
            graph.dstdata.update({'f_nj': f_nj})
            # add ni, nj factors
            graph.apply_edges(dgl.function.u_add_v('f_ni', 'f_nj', 'f_tmp'))
            # add fij to node factor
            f_out = graph.edata.pop('f_tmp') + f_fij
            if self.bias is not None:
                f_out = f_out + self.bias
            f_out = nn.functional.leaky_relu(f_out)
            f_out = f_out.view(-1, self._out_edge_feats)
            graph.edata['a'] = torch.ones((f_out.shape[0], 1)).cuda()
            graph.ndata['h_out'] = self.fc_node(nfeats).view(-1, self._out_node_feats)
            # calc weighted sum
            graph.update_all(dgl.function.u_mul_e('h_out', 'a', 'm'),
                             dgl.function.mean('m', 'h_out'))

            h_out = graph.ndata['h_out'].view(-1, self._out_node_feats)
            return h_out, f_out

class EGATConv(nn.Module):

    def __init__(self,
                 in_node_feats,
                 in_edge_feats,
                 out_node_feats,
                 out_edge_feats,
                 num_heads,
                 bias=True):

        super().__init__()
        self._num_heads = num_heads
        self._out_node_feats = out_node_feats
        self._out_edge_feats = out_edge_feats
        self.fc_node = nn.Linear(in_node_feats, out_node_feats*num_heads, bias=True)
        self.fc_ni = nn.Linear(in_node_feats, out_edge_feats*num_heads, bias=False)
        self.fc_fij = nn.Linear(in_edge_feats, out_edge_feats*num_heads, bias=False)
        self.fc_nj = nn.Linear(in_node_feats, out_edge_feats*num_heads, bias=False)
        self.attn = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_edge_feats)))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(size=(num_heads * out_edge_feats,)))
        else:
            self.register_buffer('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        gain = init.calculate_gain('relu')
        init.xavier_normal_(self.fc_node.weight, gain=gain)
        init.xavier_normal_(self.fc_ni.weight, gain=gain)
        init.xavier_normal_(self.fc_fij.weight, gain=gain)
        init.xavier_normal_(self.fc_nj.weight, gain=gain)
        init.xavier_normal_(self.attn, gain=gain)
        init.constant_(self.bias, 0)

    def forward(self, graph, nfeats, efeats, get_attention=False):
        with graph.local_scope():
            f_ni = self.fc_ni(nfeats)
            f_nj = self.fc_nj(nfeats)
            f_fij = self.fc_fij(efeats)
            graph.srcdata.update({'f_ni': f_ni})
            graph.dstdata.update({'f_nj': f_nj})
            # add ni, nj factors
            graph.apply_edges(dgl.function.u_add_v('f_ni', 'f_nj', 'f_tmp'))
            # add fij to node factor
            f_out = graph.edata.pop('f_tmp') + f_fij
            if self.bias is not None:
                f_out = f_out + self.bias
            f_out = nn.functional.leaky_relu(f_out)
            f_out = f_out.view(-1, self._num_heads, self._out_edge_feats)
            # compute attention factor
            e = (f_out * self.attn).sum(dim=-1).unsqueeze(-1)
            graph.edata['a'] = dglnn.edge_softmax(graph, e)
            graph.ndata['h_out'] = self.fc_node(nfeats).view(-1, self._num_heads,
                                                             self._out_node_feats)
            # calc weighted sum
            graph.update_all(dgl.function.u_mul_e('h_out', 'a', 'm'),
                             dgl.function.sum('m', 'h_out'))

            h_out = graph.ndata['h_out'].view(-1, self._num_heads, self._out_node_feats)
            if get_attention:
                return h_out, f_out, graph.edata.pop('a')
            else:
                return h_out.view(-1, self._out_node_feats), f_out.view(-1, self._out_edge_feats)

class EHeteroGraphConv(nn.Module):

    def __init__(self, mods):
        super(EHeteroGraphConv, self).__init__()
        self.mods = nn.ModuleDict(mods)
        for _, v in self.mods.items():
            set_allow_zero_in_degree_fn = getattr(v, 'set_allow_zero_in_degree', None)
            if callable(set_allow_zero_in_degree_fn):
                set_allow_zero_in_degree_fn(True)

    def forward(self, g, n_inputs, e_inputs):
        n_outputs = {nty : [] for nty in g.dsttypes}
        e_outputs = {}
        for stype, etype, dtype in g.canonical_etypes:
            rel_graph = g[stype, etype, dtype]
            if rel_graph.number_of_edges() == 0:
                continue
            if stype not in n_inputs:
                continue
            if stype != dtype:
                h_n = torch.cat([n_inputs[stype], n_inputs[dtype]], dim=0)
            else:
                h_n = n_inputs[stype]
            dstdata, e_output = self.mods[etype](dgl.to_homogeneous(rel_graph), h_n, e_inputs[etype])
            if stype != dtype:
                dstdata = dstdata[n_inputs[stype].shape[0]:]
            n_outputs[dtype].append(dstdata)
            e_outputs[etype] = e_output
        n_rsts = {}
        for nty, alist in n_outputs.items():
            if len(alist) != 0:
                n_rsts[nty] = torch.mean(torch.stack(alist), dim=0)
        return n_rsts, e_outputs


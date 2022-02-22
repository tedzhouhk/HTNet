import torch
import dgl
import dgl.nn.pytorch as dglnn

class NetModel(torch.nn.Module):

    def __init__(self, num_layer, dim, is_graph, is_hetero, is_dynamic, num_snapshot):
        super(NetModel, self).__init__()
        self.num_layer = num_layer
        self.is_graph = is_graph
        self.is_hetero = is_hetero
        self.is_dynamic = is_dynamic
        self.num_snapshot = num_snapshot

        mods = dict()
        dim_node_in = 21
        if not self.is_graph:
            dim_node_in += 4
        for l in range(self.num_layer):
            if not self.is_graph:
                mods['n' + str(l)] = torch.nn.BatchNorm1d(dim_node_in, track_running_stats=False)
                mods['l' + str(l)] = Perceptron(dim_node_in, dim)
            dim_node_in = dim
        if self.is_dynamic:
            mods['rnn'] = torch.nn.RNN(dim, dim, 2)
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
import torch
import torch.nn as nn
import math
from timm.models.layers import DropPath

class GraphConv(nn.Module):

    def __init__(self, in_features, out_features, adj, bias=True):
        super(GraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.zeros(size=(2, in_features, out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.adj = adj  

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        h0 = torch.matmul(input, self.W[0])
        h1 = torch.matmul(input, self.W[1])

        if self.adj.device != input.device:
            self.adj = self.adj.to(input.device)

        N = self.adj.shape[0]
        adj = self.adj + torch.eye(N).to(self.adj.device)
        d = adj.sum(1)
        adj = adj / (d.view(-1, 1) + 1e-6)

        output = torch.matmul(adj, h0) + torch.matmul(adj.transpose(0, 1), h1)

        if self.bias is not None:
            return output + self.bias
        return output

class ModulatedGraphConv(nn.Module):

    def __init__(self, in_features, out_features, adj, bias=True):
        super(ModulatedGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.zeros(size=(2, in_features, out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.adj = adj

        self.M = nn.Parameter(torch.zeros_like(adj, dtype=torch.float))
        nn.init.constant_(self.M.data, 1e-6)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        h0 = torch.matmul(input, self.W[0])
        h1 = torch.matmul(input, self.W[1])

        if self.adj.device != input.device:
            self.adj = self.adj.to(input.device)

        adj_total = self.adj + self.M

        N = adj_total.shape[0]
        adj_total = adj_total + torch.eye(N).to(adj_total.device)
        d = adj_total.sum(1)
        adj_norm = adj_total / (d.view(-1, 1) + 1e-6)

        output = torch.matmul(adj_norm, h0) + torch.matmul(adj_norm.transpose(0, 1), h1)

        if self.bias is not None:
            return output + self.bias
        return output


class ResGCNBlock(nn.Module):

    def __init__(self, hidden_dim, adj, drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(hidden_dim)

        self.gcn1 = GraphConv(hidden_dim, hidden_dim, adj)
        self.act = nn.GELU()
        self.gcn2 = GraphConv(hidden_dim, hidden_dim, adj)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.gcn1(x)
        x = self.act(x)
        x = self.gcn2(x)
        x = shortcut + self.drop_path(x)
        return x

class ModulatedGCNBlock(nn.Module):

    def __init__(self, hidden_dim, adj, drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(hidden_dim)

        self.gcn1 = ModulatedGraphConv(hidden_dim, hidden_dim, adj)
        self.act = nn.GELU()
        self.gcn2 = ModulatedGraphConv(hidden_dim, hidden_dim, adj)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        nn.init.constant_(self.gcn2.W.data, 0)
        if self.gcn2.bias is not None:
            nn.init.constant_(self.gcn2.bias.data, 0)

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.gcn1(x)
        x = self.act(x)
        x = self.gcn2(x) 
        x = shortcut + self.drop_path(x)
        return x
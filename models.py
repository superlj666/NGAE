import math
import random
import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init
import numpy as np
import igraph
import pdb
from util import Newton_method, sgd_method, is_same_DAG

# This file implements the NAS method with graph VAE.
class NGAE(nn.Module):
    def __init__(self, max_n, nvt, START_TYPE, END_TYPE, hs=501, nz=56, bidirectional=False, vid=True): 
        super(NGAE, self).__init__()
        self.max_n = max_n  # maximum number of vertices
        self.nvt = nvt  # number of vertex types
        self.START_TYPE = START_TYPE
        self.END_TYPE = END_TYPE
        self.hs = hs  # hidden state size of each vertex
        self.nz = nz  # size of latent representation z
        self.gs = hs  # size of graph state
        self.bidir = bidirectional  # whether to use bidirectional encoding
        self.vid = vid
        self.device = None

        if self.vid:
            self.vs = hs + max_n  # vertex state size = hidden state + vid
        else:
            self.vs = hs

        # 0. encoding-related
        self.grue_forward = nn.GRUCell(nvt, hs)  # encoder GRU
        self.grue_backward = nn.GRUCell(nvt, hs)  # backward encoder GRU
        self.fc1 = nn.Linear(self.gs, nz)  # latent mean
        self.fc2 = nn.Linear(self.gs, nz)  # latent logvar
            
        # 1. decoding-related
        self.grud = nn.GRUCell(nvt, hs)  # decoder GRU
        self.fc3 = nn.Linear(nz, hs)  # from latent z to initial hidden state h0
        self.add_vertex = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.ReLU(),
                nn.Linear(hs * 2, nvt)
                )  # which type of new vertex to add f(h0, hg)
        self.add_edge = nn.Sequential(
                nn.Linear(hs * 2, hs * 4), 
                nn.ReLU(), 
                nn.Linear(hs * 4, 1)
                )  # whether to add edge between v_i and v_new, f(hvi, hnew)

        # 2. gate-related
        self.gate_forward = nn.Sequential(
                nn.Linear(self.vs, hs), 
                nn.Sigmoid()
                )
        self.gate_backward = nn.Sequential(
                nn.Linear(self.vs, hs), 
                nn.Sigmoid()
                )
        self.mapper_forward = nn.Sequential(
                nn.Linear(self.vs, hs, bias=False),
                )  # disable bias to ensure padded zeros also mapped to zeros
        self.mapper_backward = nn.Sequential(
                nn.Linear(self.vs, hs, bias=False), 
                )

        # 3. bidir-related, to unify sizes
        if self.bidir:
            self.hv_unify = nn.Sequential(
                    nn.Linear(hs * 2, hs), 
                    )
            self.hg_unify = nn.Sequential(
                    nn.Linear(self.gs * 2, self.gs), 
                    )

        # 4. other
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.logsoftmax1 = nn.LogSoftmax(1)

    def get_device(self):
        if self.device is None:
            self.device = next(self.parameters()).device
        return self.device
    
    def _get_zeros(self, n, length):
        return torch.zeros(n, length).to(self.get_device()) # get a zero hidden state

    def _get_zero_hidden(self, n=1):
        return self._get_zeros(n, self.hs) # get a zero hidden state

    def _one_hot(self, idx, length):
        if type(idx) in [list, range]:
            if idx == []:
                return None
            idx = torch.LongTensor(idx).unsqueeze(0).t()
            x = torch.zeros((len(idx), length)).scatter_(1, idx, 1).to(self.get_device())
        else:
            idx = torch.LongTensor([idx]).unsqueeze(0)
            x = torch.zeros((1, length)).scatter_(1, idx, 1).to(self.get_device())
        return x

    def _gated(self, h, gate, mapper): # gated sum of 
        return gate(h) * mapper(h)

    def _collate_fn(self, G):
        return [g.copy() for g in G]

    def _propagate_to(self, G, v, propagator, H=None, reverse=False):
        # propagate messages to vertex index v for all graphs in G
        # return the new messages (states) at v
        G = [g for g in G if g.vcount() > v]
        if len(G) == 0:
            return
        if H is not None:
            idx = [i for i, g in enumerate(G) if g.vcount() > v]
            H = H[idx]
        v_types = [g.vs[v]['type'] for g in G]
        X = self._one_hot(v_types, self.nvt)
        if reverse:
            H_name = 'H_backward'  # name of the hidden states attribute
            H_pred = [[g.vs[x][H_name] for x in g.successors(v)] for g in G]
            if self.vid:
                vids = [self._one_hot(g.successors(v), self.max_n) for g in G]
            gate, mapper = self.gate_backward, self.mapper_backward
        else:
            H_name = 'H_forward'  # name of the hidden states attribute
            H_pred = [[g.vs[x][H_name] for x in g.predecessors(v)] for g in G] #
            if self.vid:
                vids = [self._one_hot(g.predecessors(v), self.max_n) for g in G]
            gate, mapper = self.gate_forward, self.mapper_forward
        if self.vid:
            H_pred = [[torch.cat([x[i], y[i:i+1]], 1) for i in range(len(x))] for x, y in zip(H_pred, vids)]  # batch size * the number of predecessors * vs
        # if h is not provided, use gated sum of v's predecessors' states as the input hidden state
        if H is None:
            max_n_pred = max([len(x) for x in H_pred])  # maximum number of predecessors
            if max_n_pred == 0:
                H = self._get_zero_hidden(len(G))
            else:
                H_pred = [torch.cat(h_pred + [self._get_zeros(max_n_pred - len(h_pred), self.vs)], 0).unsqueeze(0) for h_pred in H_pred]  # pad all to same length
                H_pred = torch.cat(H_pred, 0)  # batch * max_n_pred * vs
                H = self._gated(H_pred, gate, mapper).sum(1)  # batch * hs
        Hv = propagator(X, H)  ### Update function of GRU cell, H's shape is same to Hv
        for i, g in enumerate(G):
            g.vs[v][H_name] = Hv[i:i+1]
        return Hv

    def _propagate_from(self, G, v, propagator, H0=None, reverse=False):
        # perform a series of propagation_to steps starting from v following a topo order
        # assume the original vertex indices are in a topological order
        if reverse:
            prop_order = range(v, -1, -1)
        else:
            prop_order = range(v, self.max_n)
        Hv = self._propagate_to(G, v, propagator, H0, reverse=reverse)  # the initial vertex
        for v_ in prop_order[1:]:
            self._propagate_to(G, v_, propagator, reverse=reverse)
        return Hv

    def _update_v(self, G, v, H0=None):
        # perform a forward propagation step at v when decoding to update v's state
        self._propagate_to(G, v, self.grud, H0, reverse=False)
        return
    
    def _get_vertex_state(self, G, v):
        # get the vertex states at v
        Hv = []
        for g in G:
            if v >= g.vcount():
                hv = self._get_zero_hidden()
            else:
                hv = g.vs[v]['H_forward']
            Hv.append(hv)
        Hv = torch.cat(Hv, 0)
        return Hv

    def _get_graph_state(self, G, decode=False):
        # get the graph states
        # when decoding, use the last generated vertex's state as the graph state
        # when encoding, use the ending vertex state or unify the starting and ending vertex states
        Hg = []
        for g in G:
            hg = g.vs[g.vcount()-1]['H_forward']
            if self.bidir and not decode:  # decoding never uses backward propagation
                hg_b = g.vs[0]['H_backward']
                hg = torch.cat([hg, hg_b], 1)
            Hg.append(hg)
        Hg = torch.cat(Hg, 0)
        if self.bidir and not decode:
            Hg = self.hg_unify(Hg)
        return Hg

    def encode(self, G):
        # encode graphs G into latent vectors
        if type(G) != list:
            G = [G]
        self._propagate_from(G, 0, self.grue_forward, H0=self._get_zero_hidden(len(G)),
                             reverse=False)
        if self.bidir:
            self._propagate_from(G, self.max_n-1, self.grue_backward, 
                                 H0=self._get_zero_hidden(len(G)), reverse=True)
        Hg = self._get_graph_state(G)
        mu, logvar = self.fc1(Hg), self.fc2(Hg) 
        return mu, logvar

    def reparameterize(self, mu, logvar, eps_scale=0.01):
        # return z ~ N(mu, std)
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std) * eps_scale
            return eps.mul(std).add_(mu)
        else:
            return mu

    def _get_edge_score(self, Hvi, H, H0):
        # compute scores for edges from vi based on Hvi, H (current vertex) and H0
        # in most cases, H0 need not be explicitly included since Hvi and H contain its information
        return self.sigmoid(self.add_edge(torch.cat([Hvi, H], -1)))

    def decode(self, z, stochastic=True):
        # decode latent vectors z back to graphs
        # if stochastic=True, stochastically sample each action from the predicted distribution;
        # otherwise, select argmax action deterministically.
        H0 = self.tanh(self.fc3(z))  # or relu activation, similar performance
        G = [igraph.Graph(directed=True) for _ in range(len(z))]
        for g in G:
            g.add_vertex(type=self.START_TYPE)
        self._update_v(G, 0, H0)
        finished = [False] * len(G)
        for idx in range(1, self.max_n):
            # decide the type of the next added vertex
            if idx == self.max_n - 1:  # force the last node to be end_type
                new_types = [self.END_TYPE] * len(G)
            else:
                Hg = self._get_graph_state(G, decode=True)
                type_scores = self.add_vertex(Hg)
                if stochastic:
                    type_probs = F.softmax(type_scores, 1).cpu().detach().numpy()
                    new_types = [np.random.choice(range(self.nvt), p=type_probs[i]) 
                                 for i in range(len(G))]
                else:
                    new_types = torch.argmax(type_scores, 1)
                    new_types = new_types.flatten().tolist()
            for i, g in enumerate(G):
                if not finished[i]:
                    g.add_vertex(type=new_types[i])
            self._update_v(G, idx)

            # decide connections
            edge_scores = []
            for vi in range(idx-1, -1, -1):
                Hvi = self._get_vertex_state(G, vi)
                H = self._get_vertex_state(G, idx)
                ei_score = self._get_edge_score(Hvi, H, H0)
                if stochastic:
                    random_score = torch.rand_like(ei_score)
                    decisions = random_score < ei_score
                else:
                    decisions = ei_score > 0.5
                for i, g in enumerate(G):
                    if finished[i]:
                        continue
                    if new_types[i] == self.END_TYPE: 
                    # if new node is end_type, connect it to all loose-end vertices (out_degree==0)
                        end_vertices = set([v.index for v in g.vs.select(_outdegree_eq=0) 
                                            if v.index != g.vcount()-1])
                        for v in end_vertices:
                            g.add_edge(v, g.vcount()-1)
                        finished[i] = True
                        continue
                    if decisions[i, 0]:
                        g.add_edge(vi, g.vcount()-1)
                self._update_v(G, idx)

        for g in G:
            del g.vs['H_forward']  # delete hidden states to save GPU memory
        return G

    def loss(self, mu, logvar, G_true, beta=0.005):
        # compute the loss of decoding mu and logvar to true graphs using teacher forcing
        # ensure when computing the loss of step i, steps 0 to i-1 are correct
        z = self.reparameterize(mu, logvar)
        H0 = self.tanh(self.fc3(z))  # or relu activation, similar performance
        G = [igraph.Graph(directed=True) for _ in range(len(z))]
        for g in G:
            g.add_vertex(type=self.START_TYPE)
        self._update_v(G, 0, H0)
        res = 0  # log likelihood
        for v_true in range(1, self.max_n):
            # calculate the likelihood of adding true types of nodes
            # use start type to denote padding vertices since start type only appears for vertex 0 
            # and will never be a true type for later vertices, thus it's free to use
            true_types = [g_true.vs[v_true]['type'] if v_true < g_true.vcount() 
                          else self.START_TYPE for g_true in G_true]
            Hg = self._get_graph_state(G, decode=True)
            type_scores = self.add_vertex(Hg)
            # vertex log likelihood
            vll = self.logsoftmax1(type_scores)[np.arange(len(G)), true_types].sum()  
            res = res + vll
            for i, g in enumerate(G):
                if true_types[i] != self.START_TYPE:
                    g.add_vertex(type=true_types[i])
            self._update_v(G, v_true)

            # calculate the likelihood of adding true edges
            true_edges = []
            for i, g_true in enumerate(G_true):
                true_edges.append(g_true.get_adjlist(igraph.IN)[v_true] if v_true < g_true.vcount()
                                  else [])
            edge_scores = []
            for vi in range(v_true-1, -1, -1):
                Hvi = self._get_vertex_state(G, vi)
                H = self._get_vertex_state(G, v_true)
                ei_score = self._get_edge_score(Hvi, H, H0)
                edge_scores.append(ei_score)
                for i, g in enumerate(G):
                    if vi in true_edges[i]:
                        g.add_edge(vi, v_true)
                self._update_v(G, v_true)
            edge_scores = torch.cat(edge_scores[::-1], 1)

            ground_truth = torch.zeros_like(edge_scores)
            idx1 = [i for i, x in enumerate(true_edges) for _ in range(len(x))]
            idx2 = [xx for x in true_edges for xx in x]
            ground_truth[idx1, idx2] = 1.0

            # edges log-likelihood
            ell = - F.binary_cross_entropy(edge_scores, ground_truth, reduction='sum') 
            res = res + ell

        res = -res  # convert likelihood to loss
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return res + beta*kld, res, kld

    def encode_decode(self, G):
        mu, logvar = self.encode(G)
        z = self.reparameterize(mu, logvar)
        return self.decode(z)

    def forward(self, G):
        mu, logvar = self.encode(G)
        loss, _, _ = self.loss(mu, logvar, G)
        return loss
    
    def generate_sample(self, n):
        sample = torch.randn(n, self.nz).to(self.get_device())
        G = self.decode(sample)
        return G

    def generate_optimized_sample(self, n):
        sample = torch.randn(n, self.nz).to(self.get_device())
        
        G = self.decode(sample)
        return G

    '''search methods'''
    def random_search(self, k, search_optimizer, lr = 1e-2):
        pred_graphs = []
        # start_points_latent = torch.randn([k, self.nz], device='cuda:0', requires_grad=True)
        start_points_latent = torch.tensor(torch.randn(k, self.nz), device='cuda:0', requires_grad=True)
        for i, sample_latent in enumerate(start_points_latent):            
            pred_acc_before = self.predictor(sample_latent)
            pred_comp_before = self.predictor_complexity(sample_latent)

            #pred_acc = self.predictor(sample_latent)
            if search_optimizer == 'Newton':
                optimized_sample_latent = Newton_method(sample_latent, self.predictor, lr)
            else:
                optimized_sample_latent = sgd_method(sample_latent, self.predictor, self.predictor_complexity, lr)

            sample_graph = self.decode(optimized_sample_latent.reshape(1, self.nz))
            
            pred_acc_after = self.predictor(optimized_sample_latent)
            pred_comp_after = self.predictor_complexity(optimized_sample_latent)
            print('{} graph {} : accuracy {} -> {}, complexity {} -> {}'.format(search_optimizer, i, pred_acc_before[0].data, pred_acc_after[0].data, pred_comp_before[0].data, pred_comp_after[0].data))
            
            graph_exist = False
            for g in pred_graphs:
                if is_same_DAG(g[0], sample_graph[0]):
                    graph_exist = True
                    break
            if not graph_exist:
                pred_graphs.append((sample_graph[0], pred_acc_after[0]))
        return pred_graphs

    def optimal_search(self, k, search_optimizer, data, lr = 1e-2):
        pred_graphs = []
        # start_points_latent = torch.randn([k, self.nz], device='cuda:0', requires_grad=True)
        start_points_latent = self.best_k_performance(k, data)
        print("search samples: {}".format(k))

        for i, sample_latent in enumerate(start_points_latent):            
            pred_acc_before = self.predictor(sample_latent)
            pred_comp_before = self.predictor_complexity(sample_latent)

            if search_optimizer == 'Newton':
                optimized_sample_latent = Newton_method(sample_latent, self.predictor, lr)
            else:
                optimized_sample_latent = sgd_method(sample_latent, self.predictor, self.predictor_complexity, lr)
            
            pred_acc_after = self.predictor(optimized_sample_latent)
            pred_comp_after = self.predictor_complexity(optimized_sample_latent)
            print('{} graph {} : accuracy {} -> {}, complexity {} -> {}'.format(search_optimizer, i, pred_acc_before[0].data, pred_acc_after[0].data, pred_comp_before[0].data, pred_comp_after[0].data))
            sample_graph = self.decode(optimized_sample_latent.reshape(1, self.nz))
            # sample_graph = decode_from_latent_space(optimized_sample_latent.reshape(1, self.nz), self)
            
            graph_exist = False
            for g in pred_graphs:
                if is_same_DAG(g[0], sample_graph[0]):
                    graph_exist = True
                    break
            if not graph_exist:
                pred_graphs.append((sample_graph[0], pred_acc_after[0]))

        return pred_graphs

    def best_k_performance(self, k, data):
        perform_dict = {}
        worst_acc = 0
        for i, d in enumerate(data):
            acc = d[1]
            if acc in perform_dict:
                perform_dict[acc] += [i]
            elif len(perform_dict) < k:
                perform_dict[acc] = [i]
            else:                
                worst_acc = np.min(list(perform_dict.keys()))
                if acc > worst_acc:
                    del perform_dict[worst_acc]
                    perform_dict[acc] = [i]

        sample_list = []
        for (_, index_set) in perform_dict.items():
            for index in index_set:
                sample_list.append(data[index][0])
        
        samples_lentent_list = torch.zeros([len(sample_list), self.nz], device='cuda:0', requires_grad=True)
        for i, sample in enumerate(sample_list):
            sample_lentent, _ = self.encode(sample)
                # sample_lentent.requires_grad = True
            samples_lentent_list[i] = sample_lentent

        return samples_lentent_list

'''
    NGAE with GCN encoder
    The message passing happens at all nodes simultaneously.
'''
class NGAE_GCN(NGAE):
    def __init__(self, max_n, nvt, START_TYPE, END_TYPE, hs=501, nz=56, bidirectional=False, levels=3):
        # bidirectional means passing messages ignoring edge directions
        super(NGAE_GCN, self).__init__(max_n, nvt, START_TYPE, END_TYPE, hs, nz, bidirectional)
        self.levels = levels
        self.gconv = nn.ModuleList()
        self.gconv.append(
                nn.Sequential(
                    nn.Linear(nvt, hs), 
                    nn.ReLU(), 
                    )
                )
        for lv in range(1, levels):
            self.gconv.append(
                    nn.Sequential(
                        nn.Linear(hs, hs), 
                        nn.ReLU(), 
                        )
                    )

    def _get_feature(self, g, v, lv=0):
        # get the node feature vector of v
        if lv == 0:  # initial level uses type features
            v_type = g.vs[v]['type']
            x = self._one_hot(v_type, self.nvt)
        else:
            x = g.vs[v]['H_forward']
        return x

    def _get_zero_x(self, n=1):
        # get zero predecessor states X, used for padding
        return torch.zeros(n, self.nvt).to(self.get_device())

    def _get_graph_state(self, G, decode=False, start=0, end_offset=0):
        # get the graph states
        # sum all node states between start and n-end_offset as the graph state
        Hg = []
        max_n_nodes = max(g.vcount() for g in G)
        for g in G:
            hg = torch.cat([g.vs[i]['H_forward'] for i in range(start, g.vcount() - end_offset)],
                           0).unsqueeze(0)  # 1 * n * hs
            if g.vcount() < max_n_nodes:
                hg = torch.cat([hg, 
                    torch.zeros(1, max_n_nodes - g.vcount(), hg.shape[2]).to(self.get_device())],
                    1)  # 1 * max_n * hs
            Hg.append(hg)
        # sum node states as the graph state
        Hg = torch.cat(Hg, 0).sum(1)  # batch * hs
        return Hg  # batch * hs

    def _GCN_propagate_to(self, G, v, lv=0):
        # propagate messages to vertex index v for all graphs in G
        # return the new messages (states) at v
        G = [g for g in G if g.vcount() > v]
        if len(G) == 0:
            return

        if self.bidir:  # ignore edge directions, accept all neighbors' messages
            H_nei = [[self._get_feature(g, v, lv)/(g.degree(v)+1)] + 
                     [self._get_feature(g, x, lv)/math.sqrt((g.degree(x)+1)*(g.degree(v)+1)) 
                     for x in g.neighbors(v)] for g in G]
        else:  # only accept messages from predecessors (generalizing GCN to directed cases)
            H_nei = [[self._get_feature(g, v, lv)/(g.indegree(v)+1)] + 
                     [self._get_feature(g, x, lv)/math.sqrt((g.outdegree(x)+1)*(g.indegree(v)+1)) 
                     for x in g.predecessors(v)] for g in G]
            
        max_n_nei = max([len(x) for x in H_nei])  # maximum number of neighbors
        H_nei = [torch.cat(h_nei + [self._get_zeros(max_n_nei - len(h_nei), h_nei[0].shape[1])], 0).unsqueeze(0) 
                 for h_nei in H_nei]  # pad all to same length
        H_nei = torch.cat(H_nei, 0)  # batch * max_n_nei * nvt
        Hv = self.gconv[lv](H_nei.sum(1))  # batch * hs
        for i, g in enumerate(G):
            g.vs[v]['H_forward'] = Hv[i:i+1]
        return Hv

    def encode(self, G):
        # encode graphs G into latent vectors
        # GCN propagation is now implemented in a non-parallel way for consistency, but
        # can definitely be parallel to speed it up. However, the major computation cost
        # comes from the generation, which is not parallellizable.
        if type(G) != list:
            G = [G]
        prop_order = range(self.max_n)
        for lv in range(self.levels):
            for v_ in prop_order:
                self._GCN_propagate_to(G, v_, lv)
        Hg = self._get_graph_state(G, start=1, end_offset=1)  # does not use the dummy input 
                                                              # and output nodes
        mu, logvar = self.fc1(Hg), self.fc2(Hg) 
        return mu, logvar

'''
    A fast NGAE variant.
    Use NGAE's encoder + S-VAE's decoder to accelerate decoding.
'''
class NGAE_fast(NGAE):
    def __init__(self, max_n, nvt, START_TYPE, END_TYPE, hs=501, nz=56, bidirectional=False):
        super(NGAE_fast, self).__init__(max_n, nvt, START_TYPE, END_TYPE, hs, nz, bidirectional)
        self.grud = nn.GRU(hs, hs, batch_first=True)  # decoder GRU
        self.add_vertex = nn.Sequential(
                nn.Linear(self.hs, self.hs), 
                nn.ReLU(), 
                nn.Linear(self.hs, self.nvt), 
                )
        self.add_edges = nn.Sequential(
                nn.Linear(self.hs, self.hs), 
                nn.ReLU(), 
                nn.Linear(self.hs, self.max_n - 1), 
                )

    def _decode(self, z):
        h0 = self.relu(self.fc3(z))
        h_in = h0.unsqueeze(1).expand(-1, self.max_n - 1, -1)
        h_out, _ = self.grud(h_in)
        type_scores = self.add_vertex(h_out)  # batch * max_n-1 * nvt
        edge_scores = self.sigmoid(self.add_edges(h_out))  # batch * max_n-1 * max_n-1
        return type_scores, edge_scores

    def loss(self, mu, logvar, G_true, beta=0.005):
        # g_true: [batch_size * max_n-1 * xs]
        z = self.reparameterize(mu, logvar)
        type_scores, edge_scores = self._decode(z)
        res = 0
        true_types = torch.LongTensor([[g_true.vs[v_true]['type'] if v_true < g_true.vcount() 
                                      else self.START_TYPE for v_true in range(1, self.max_n)] 
                                      for g_true in G_true]).to(self.get_device())
        res += F.cross_entropy(type_scores.transpose(1, 2), true_types, reduction='sum')
        true_edges = torch.FloatTensor([np.pad(np.array(g_true.get_adjacency().data).transpose()[1:, :-1],
                                        ((0, self.max_n-g_true.vcount()), (0, self.max_n-g_true.vcount())),
                                        mode='constant', constant_values=(0, 0))
                                       for g_true in G_true]).to(self.get_device())
        res += F.binary_cross_entropy(edge_scores, true_edges, reduction='sum')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return res + beta*kld, res, kld

    def decode(self, z):
        type_scores, edge_scores = self._decode(z)
        return self.construct_igraph(type_scores, edge_scores)

    def construct_igraph(self, type_scores, edge_scores, stochastic=True):
        # copy from S-VAE
        assert(type_scores.shape[:2] == edge_scores.shape[:2])
        if stochastic:
            type_probs = F.softmax(type_scores, 2).cpu().detach().numpy()
        G = []
        for gi in range(len(type_scores)):
            g = igraph.Graph(directed=True)
            g.add_vertex(type=self.START_TYPE)
            for vi in range(1, self.max_n):
                if vi == self.max_n - 1:
                    new_type = self.END_TYPE
                else:
                    if stochastic:
                        new_type = np.random.choice(range(self.nvt), p=type_probs[gi][vi-1])
                    else:
                        new_type = torch.argmax(type_scores[gi][vi-1], 0).item()
                g.add_vertex(type=new_type)
                if new_type == self.END_TYPE:  
                    end_vertices = set([v.index for v in g.vs.select(_outdegree_eq=0) 
                                       if v.index != g.vcount()-1])
                    for v in end_vertices:
                        g.add_edge(v, vi)
                    break
                else:
                    for ej in range(vi):
                        ej_score = edge_scores[gi][vi-1][ej].item()
                        if stochastic:
                            if np.random.random_sample() < ej_score:
                                g.add_edge(ej, vi)
                        else:
                            if ej_score > 0.5:
                                g.add_edge(ej, vi)
            G.append(g)
        return G

class NASNetworkCIFAR(nn.Module):
    def __init__(self, args, classes, layers, nodes, channels, keep_prob, drop_path_keep_prob, use_aux_head, steps, arch):
        super(NASNetworkCIFAR, self).__init__()
        self.args = args
        self.search_space = args.search_space
        self.classes = classes
        self.layers = layers
        self.nodes = nodes
        self.channels = channels
        self.keep_prob = keep_prob
        self.drop_path_keep_prob = drop_path_keep_prob
        self.use_aux_head = use_aux_head
        self.steps = steps
        if isinstance(arch, str):
            arch = list(map(int, arch.strip().split()))
        elif isinstance(arch, list) and len(arch) == 2:
            arch = arch[0] + arch[1]
        self.conv_arch = arch[:4 * self.nodes]
        self.reduc_arch = arch[4 * self.nodes:]

        self.pool_layers = [self.layers, 2 * self.layers + 1]
        self.layers = self.layers * 3
        
        if self.use_aux_head:
            self.aux_head_index = self.pool_layers[-1] #+ 1
        stem_multiplier = 3
        channels = stem_multiplier * self.channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        outs = [[32, 32, channels],[32, 32, channels]]
        channels = self.channels
        self.cells = nn.ModuleList()
        for i in range(self.layers+2):
            if i not in self.pool_layers:
                cell = Cell(self.search_space, self.conv_arch, outs, channels, False, i, self.layers+2, self.steps, self.drop_path_keep_prob)
            else:
                channels *= 2
                cell = Cell(self.search_space, self.reduc_arch, outs, channels, True, i, self.layers+2, self.steps, self.drop_path_keep_prob)
            self.cells.append(cell)
            outs = [outs[-1], cell.out_shape]
            
            if self.use_aux_head and i == self.aux_head_index:
                self.auxiliary_head = AuxHeadCIFAR(outs[-1][-1], classes)
        
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(1 - self.keep_prob)
        self.classifier = nn.Linear(outs[-1][-1], classes)
        
        self.init_parameters()
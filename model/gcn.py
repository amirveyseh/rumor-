"""
GCN model for relation extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from model.tree import Tree, head_to_tree, tree_to_adj
from utils import constant, torch_utils
import math

class GCNClassifier(nn.Module):
    """ A wrapper classifier for GCNRelationModel. """
    def __init__(self, opt, emb_matrix=None):
        super().__init__()
        self.gcn_model = GCNRelationModel(opt, emb_matrix=emb_matrix)
        in_dim = opt['hidden_dim']
        self.classifier = nn.Linear(in_dim, opt['num_class'])
        self.opt = opt

    def conv_l2(self):
        return self.gcn_model.gcn.conv_l2()

    def forward(self, inputs):
        outputs, pooling_output, g, sparse_graph, c1, c2 = self.gcn_model(inputs)
        logits = self.classifier(outputs)
        return logits, pooling_output, g, sparse_graph, c1, c2

class GCNRelationModel(nn.Module):
    def __init__(self, opt, emb_matrix=None):
        super().__init__()
        self.opt = opt
        self.emb_matrix = emb_matrix

        # create embedding layers
        self.emb = nn.Embedding(opt['vocab_size'], opt['emb_dim'], padding_idx=constant.PAD_ID)
        embeddings = self.emb
        self.init_embeddings()

        # gcn layer
        self.gcn = GCN(opt, embeddings, opt['hidden_dim'], opt['num_layers'])

        # output mlp layers
        in_dim = opt['emb_dim']*1
        # in_dim = opt['emb_dim']*1
        layers = [nn.Linear(in_dim, opt['hidden_dim']), nn.ReLU()]
        for _ in range(self.opt['mlp_layers']-1):
            layers += [nn.Linear(opt['hidden_dim'], opt['hidden_dim']), nn.ReLU()]
        self.out_mlp = nn.Sequential(*layers)

        self.C1 = nn.Sequential(nn.Linear(opt['hidden_dim'], 100), nn.Tanh(), nn.Linear(100, 100), nn.Tanh())
        self.C2 = nn.Sequential(nn.Linear(opt['emb_dim'], 100), nn.Tanh(), nn.Linear(100, 100), nn.Tanh())

        self.pred = nn.Sequential(nn.Linear(100, 3), nn.Tanh())

    def init_embeddings(self):
        if self.emb_matrix is None:
            self.emb.weight.data[1:,:].uniform_(-1.0, 1.0)
        else:
            self.emb_matrix = torch.from_numpy(self.emb_matrix)
            self.emb.weight.data.copy_(self.emb_matrix)
        # decide finetuning
        if self.opt['topn'] <= 0:
            print("Do not finetune word embedding layer.")
            self.emb.weight.requires_grad = False
        elif self.opt['topn'] < self.opt['vocab_size']:
            print("Finetune top {} word embeddings.".format(self.opt['topn']))
            self.emb.weight.register_hook(lambda x: \
                    torch_utils.keep_partial_grad(x, self.opt['topn']))
        else:
            print("Finetune all embeddings.")

    def forward(self, inputs):
        words, masks, adj, main, main_mask, context, context_mask, tweetInd, user, user_mask = inputs # unpack

        h, pool_mask, g, sparse_graph, main, user = self.gcn(adj.float(), inputs)
        
        # pooling
        pool_type = self.opt['pooling']
        h_out = pool(h, pool_mask, type=pool_type)
        # h_out = h
        outputs = torch.cat([h_out], dim=1)
        outputs = self.out_mlp(outputs)
        c1 = self.pred(self.C1(outputs))
        c2 = self.pred(self.C2(main))
        return outputs, h_out, g, sparse_graph, c1, c2

class GCN(nn.Module):
    """ A GCN/Contextualized GCN module operated on dependency graphs. """
    def __init__(self, opt, embeddings, mem_dim, num_layers):
        super(GCN, self).__init__()
        self.opt = opt
        self.layers = num_layers
        self.use_cuda = opt['cuda']
        self.mem_dim = mem_dim
        self.in_dim = opt['emb_dim']

        self.emb = embeddings

        self.K = nn.Linear(self.in_dim, self.in_dim)
        self.Q = nn.Linear(self.in_dim, self.in_dim)

        self.K1 = nn.Linear(self.in_dim, self.in_dim)
        self.Q1 = nn.Linear(self.in_dim, self.in_dim)

        # self.K2 = nn.Linear(self.in_dim, 1)
        # self.Q2 = nn.Linear(self.in_dim, 1)

        # rnn layer
        # if self.opt.get('rnn', False):
        #     input_size = self.in_dim
        #     # self.rnn = nn.LSTM(input_size, opt['rnn_hidden'], opt['rnn_layers'], batch_first=True, \
        #     self.rnn = nn.LSTM(input_size, opt['emb_dim']//2, opt['rnn_layers'], batch_first=True, \
        #             dropout=opt['rnn_dropout'], bidirectional=True)
        #     self.rnn_drop = nn.Dropout(opt['rnn_dropout']) # use on last layer output

        # self.G = nn.Linear(self.in_dim, self.in_dim)

    def conv_l2(self):
        conv_weights = []
        for w in self.W:
            conv_weights += [w.weight, w.bias]
        return sum([x.pow(2).sum() for x in conv_weights])

    def encode_with_rnn(self, rnn_inputs, masks, batch_size):
        seq_lens = list(masks.data.eq(constant.PAD_ID).long().sum(1).squeeze())
        # zeros = 0
        for i, _ in enumerate(seq_lens):
            if seq_lens[i] == 0:
                seq_lens[i] = torch.tensor(1).long().cuda()
                # zeros += 1
        perms1 = [t[0] for t in sorted(zip(range(len(seq_lens)), [s.item() for s in seq_lens]), key = lambda t: t[1], reverse=True)]
        # print(np.sum([1 if perms1[i] != i else 0 for i in range(len(seq_lens))]) / len(seq_lens), zeros / len(seq_lens))
        perms2 = [t[1] for t in sorted(zip(perms1, range(len(seq_lens))), key=lambda t: t[0])]
        rnn_inputs = rnn_inputs[perms1, :]
        seq_lens = [seq_lens[i] for i in perms1]
        # h0, c0 = rnn_zero_state(batch_size, self.opt['rnn_hidden'], self.opt['rnn_layers'])
        h0, c0 = rnn_zero_state(batch_size, self.opt['emb_dim']//2, self.opt['rnn_layers'])
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens, batch_first=True)
        rnn_outputs, (ht, ct) = self.rnn(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        rnn_outputs = rnn_outputs[perms2, :]
        return rnn_outputs

    def forward(self, adj, inputs):
        words, masks, _, main, main_mask, context, context_mask, tweetInd, user, user_mask = inputs # unpack
        word_embs = self.emb(words)
        main_embs = self.emb(main)
        context_embs = self.emb(context)
        user_embs = self.emb(user)
        embs = [word_embs]
        embs = torch.cat(embs, dim=2)

        gcn_inputs = embs

        mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)

        masks = masks.unsqueeze(3)
        main_mask = main_mask.unsqueeze(3)
        context_mask = context_mask.unsqueeze(3)
        user_mask = user_mask.unsqueeze(3)

        main_embs = main_embs.masked_fill(main_mask, -constant.INFINITY_NUMBER)
        main_embs = torch.max(main_embs, 2)[0]
        main_embs = main_embs.masked_fill(main_embs.eq(-constant.INFINITY_NUMBER), 0)

        user_embs = user_embs.masked_fill(user_mask, -constant.INFINITY_NUMBER)
        user_embs = torch.max(user_embs, 2)[0]
        user_embs = torch.max(user_embs, 1)[0]
        user_embs = user_embs.masked_fill(user_embs.eq(-constant.INFINITY_NUMBER), 0)

        # context_embs = context_embs.masked_fill(context_mask, -constant.INFINITY_NUMBER)
        # context_embs = torch.max(context_embs, 2)[0]

        gcn_inputs = gcn_inputs.masked_fill(masks, -constant.INFINITY_NUMBER)
        # gcn_inputs = gcn_inputs.masked_fill(masks, 0)
        gcn_inputs = torch.max(gcn_inputs, 2)[0]
        # gcn_inputs = gcn_inputs.masked_fill(gcn_inputs.eq(-constant.INFINITY_NUMBER), 0)
        # gcn_inputs = gcn_inputs.sum(2)

        # gcn_inputs = context_embs
        # gcn_inputs = torch.cat([context_embs, main_embs], dim=2)

        key = self.K(gcn_inputs)
        query = self.Q(gcn_inputs)

        # gate = self.G(main_embs).repeat(1, words.shape[1], 1)

        sf2 = nn.Softmax(2)
        dense_graph = sf2(key.bmm(query.transpose(1,2)) / math.sqrt(self.opt['emb_dim']))
        gcn_inputs = dense_graph.bmm(gcn_inputs)

        # main_tweet = []
        # tweetInd = tweetInd.data.cpu().numpy()
        # for i, t in enumerate(tweetInd):
        #     main_tweet.append(gcn_inputs[i,t,:].unsqueeze(0))
        # main_tweet = torch.cat(main_tweet, dim=0)

        # main_tweet = self.encode_with_rnn(main_embs.squeeze(), main_mask.squeeze(1), main_embs.shape[0])[:,-1,:]
        # main_gate = nn.ReLU()(self.K1(main_tweet)).unsqueeze(1).repeat(1,gcn_inputs.shape[1],1)
        # k = nn.Sigmoid()(self.K2(main_gate * gcn_inputs))
        # # sf1 = nn.Softmax(1)
        # # k = sf1(self.K2(gcn_inputs))
        # gcn_inputs = (k*gcn_inputs).sum(1).squeeze()


        # gcn_inputs = self.encode_with_rnn(gcn_inputs, mask, words.size()[0])

        # seq_lens = mask.data.eq(constant.PAD_ID).long().sum(1).squeeze() - 1
        #
        # gcn_inputs = torch.gather(gcn_inputs, 1, seq_lens.view(-1,1).unsqueeze(2).repeat(1,1,gcn_inputs.shape[-1])).squeeze()

        return gcn_inputs, mask, dense_graph, sf2(adj), main_embs.squeeze(), user_embs.squeeze()

def pool(h, mask, type='max'):
    if type == 'max':
        # h = h.masked_fill(mask, -constant.INFINITY_NUMBER)
        return torch.max(h, 1)[0]
    elif type == 'avg':
        h = h.masked_fill(mask, 0)
        return h.sum(1) / (mask.size(1) - mask.float().sum(1))
    else:
        h = h.masked_fill(mask, 0)
        return h.sum(1)
    
def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True, use_cuda=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    if use_cuda:
        return h0.cuda(), c0.cuda()
    else:
        return h0, c0

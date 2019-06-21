"""
A trainer class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from model.gcn import GCNClassifier
from utils import constant, torch_utils

class Trainer(object):
    def __init__(self, opt, emb_matrix=None):
        raise NotImplementedError

    def update(self, batch):
        raise NotImplementedError

    def predict(self, batch):
        raise NotImplementedError

    def update_lr(self, new_lr):
        torch_utils.change_lr(self.optimizer, new_lr)

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']

    def save(self, filename, epoch):
        params = {
                'model': self.model.state_dict(),
                'config': self.opt,
                }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")


def unpack_batch(batch, cuda):
    inputs = [Variable(b).cuda() for b in batch[:10]]
    labels = Variable(batch[10]).cuda()
    lens = batch[1].eq(0).long().sum(2).squeeze()
    return inputs, labels, lens

class GCNTrainer(Trainer):
    def __init__(self, opt, emb_matrix=None):
        self.opt = opt
        self.emb_matrix = emb_matrix
        self.model = GCNClassifier(opt, emb_matrix=emb_matrix)
        self.criterion = nn.CrossEntropyLoss()
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if opt['cuda']:
            self.model.cuda()
            self.criterion.cuda()
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.parameters, opt['lr'])

    def update(self, batch):
        inputs, labels, lens = unpack_batch(batch, self.opt['cuda'])

        # step forward
        self.model.train()
        self.optimizer.zero_grad()
        logits, pooling_output, g, sparse_graph, c1, c2 = self.model(inputs)
        loss = self.criterion(logits, labels)
        # l2 decay on all conv layers
        if self.opt.get('conv_l2', 0) > 0:
            loss += self.model.conv_l2() * self.opt['conv_l2']
        # l2 penalty on output representations
        # if self.opt.get('pooling_l2', 0) > 0:
        #     loss += self.opt['pooling_l2'] * (pooling_output ** 2).sum(1).mean()
        # loss += 0.0000001 * self.hloss(g.view(-1,g.shape[1]))
        # loss += 0.0000000001 * torch.norm(torch.abs(g.view(-1, g.shape[-1])-sparse_graph.view(-1, sparse_graph.shape[-1])))
        # c1l = c1.pow(2).sum(1).sqrt().unsqueeze(1)
        # c2l = c2.pow(2).sum(1).sqrt().unsqueeze(1)
        # loss_pred = -(torch.mm(c1, c2.transpose(0,1)) / torch.mm(c1l, c2l.transpose(0,1))).diag().abs().mean() + (c1l-c2l).abs().mean()
        c2 = torch.max(c2, 1)[1]
        loss_pred = self.criterion(c1, c2)
        loss += loss_pred
        loss_val = loss.item()
        # backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt['max_grad_norm'])
        self.optimizer.step()
        return loss_val, loss_pred.item()

    def predict(self, batch, unsort=True):
        inputs, labels, lens = unpack_batch(batch, self.opt['cuda'])
        # orig_idx = batch[11]
        # forward
        self.model.eval()
        logits, _, _, _, _, _ = self.model(inputs)
        loss = self.criterion(logits, labels)
        probs = F.softmax(logits, 1).data.cpu().numpy().tolist()
        predictions = np.argmax(logits.data.cpu().numpy(), axis=1).tolist()
        # if unsort:
        #     _, predictions, probs = [list(t) for t in zip(*sorted(zip(orig_idx,\
        #             predictions, probs)))]
        return predictions, probs, loss.item()


class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = x * torch.log(x.clamp(min=1e-8))
        b = -1.0 * b.sum(1)
        return b.mean()

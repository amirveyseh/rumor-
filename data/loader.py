"""
Data loader for TACRED json files.
"""

import json
import random
import torch
import numpy as np

from utils import constant, helper, vocab
import random

MAX_TWEET_LEN = 200

class DataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self, filename, batch_size, opt, vocab, evaluation=False):
        self.batch_size = batch_size
        self.opt = opt
        self.vocab = vocab
        self.eval = evaluation
        self.label2id = constant.LABEL_TO_ID

        with open(filename) as infile:
            data = json.load(infile)
        with open(opt['data_dir']+'/user.json') as file:
            user_data = json.load(file)
        with open(opt['data_dir']+'/author.json') as file:
            authors = json.load(file)
        self.raw_data = data
        self.user_data = user_data
        self.authors = authors
        data = self.preprocess(data, vocab, opt)

        # shuffle for training
        if not evaluation:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
        self.id2label = dict([(v,k) for k,v in self.label2id.items()])
        self.labels = [self.id2label[d[-1]] for d in data]
        self.num_examples = len(data)

        # chunk into batches
        data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        self.data = data
        print("{} batches created for {}".format(len(data), filename))

    def preprocess(self, data, vocab, opt):
        """ Preprocess the data and convert to ids. """
        processed = []
        for d in data:
            tokens = list(d['tokens'])
            # anonymize tokens
            tokens = map_to_ids(tokens, vocab.word2id)[:MAX_TWEET_LEN]
            adj = list(np.asarray(d['tree'])[:MAX_TWEET_LEN,:MAX_TWEET_LEN])
            label = self.label2id[d['label']]
            main = []
            context = []
            tweetInd = -1
            for i, t in enumerate(tokens):
                if d['tweets'][i]['tweetId'] == d['id']:
                    main.append(t)
                    tweetInd = i
                else:
                    context.append(t)
            tweetInd = min(tweetInd, MAX_TWEET_LEN-1)
            user = map_to_ids(list(filter(lambda u: u['userId'] == self.authors[d['id']], self.user_data))[0]['tokens'], vocab.word2id)
            processed += [(tokens, adj, main, context, tweetInd, user, label)]
        return processed

    def gold(self):
        """ Return gold labels as a list. """
        return self.labels

    def correct(self, predictions, path1, path2, r):
        with open(path1) as file:
            data = json.load(file)

        for i, d in enumerate(data):
            if random.random() < r:
                d['label'] = predictions[i]

        with open(path2, 'w') as file:
            json.dump(data, file)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch_size = len(batch)
        batch = list(zip(*batch))
        assert len(batch) == 7

        # # sort all fields by lens for easy RNN operations
        # lens = [len(x) for x in batch[0]]
        # batch, orig_idx = sort_all(batch, lens)

        # word dropout
        # if not self.eval:
        #     words = [[word_dropout(tweet, self.opt['word_dropout'])] for tweets in batch[0] for tweet in tweets]
        # else:
        #     words = batch[0]

        words = batch[0]

        # convert to tensors
        words = get_long_tensor(words, batch_size)
        main = get_long_tensor(batch[2], batch_size)
        context = get_long_tensor(batch[3], batch_size)
        user = get_long_tensor(batch[5], batch_size)
        main_mask = torch.eq(main, 0)
        context_mask = torch.eq(context, 0)
        user_mask = torch.eq(user, 0)
        masks = torch.eq(words, 0)
        adj = get_long_tensor(batch[1], batch_size)

        tweetInd = torch.LongTensor(batch[4])
        label = torch.LongTensor(batch[-1])

        return (words, masks, adj, main, main_mask, context, context_mask, tweetInd, user, user_mask, label)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

def map_to_ids(tokens, vocab):
    ids = [[vocab[t] if t in vocab else constant.UNK_ID for t in ts] for ts in tokens]
    return ids

def get_positions(start_idx, end_idx, length):
    """ Get subj/obj position sequence. """
    return list(range(-start_idx, 0)) + [0]*(end_idx - start_idx + 1) + \
            list(range(1, length-end_idx))

def get_long_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded LongTensor. """
    tweet_len = max(len(x) for x in tokens_list)
    token_len = max(len(x) for ts in tokens_list for x in ts)
    tokens = torch.LongTensor(batch_size, tweet_len, token_len).fill_(constant.PAD_ID)
    for i, s in enumerate(tokens_list):
        for j, t in enumerate(s):
                tokens[i, j, :len(t)] = torch.LongTensor(t)
    return tokens

def sort_all(batch, lens):
    """ Sort all fields by descending order of lens, and return the original indices. """
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]

def word_dropout(tokens, dropout):
    """ Randomly dropout tokens (IDs) and replace them with <UNK> tokens. """
    return [constant.UNK_ID if x != constant.UNK_ID and np.random.random() < dropout \
            else x for x in tokens]


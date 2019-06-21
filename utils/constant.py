"""
Define constants.
"""
EMB_INIT_RANGE = 1.0

# vocab
PAD_TOKEN = '<PAD>'
PAD_ID = 0
UNK_TOKEN = '<UNK>'
UNK_ID = 1

VOCAB_PREFIX = [PAD_TOKEN, UNK_TOKEN]

LABEL_TO_ID = {'false': 0, 'true': 1, 'unverified': 2, 'non-rumor': 3}

INFINITY_NUMBER = 1e12

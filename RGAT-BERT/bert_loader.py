# coding:utf-8

import json
import torch
import numpy as np
from transformers import BertTokenizer


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def pad_and_truncate(sequence, maxlen, dtype="int64", padding="post", truncating="post", value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == "pre":
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == "post":
        x[: len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x


class ABSADataLoader(object):
    def __init__(self, filename, batch_size, args, vocab, shuffle=True):
        self.batch_size = batch_size
        self.args = args
        self.vocab = vocab

        with open(filename) as infile:
            data = json.load(infile)
        self.raw_data = data

        # preprocess data
        data = self.preprocess(data, vocab, args)
        if shuffle:
            indices = np.arange(len(data))
            np.random.shuffle(indices)
            data = [data[idx] for idx in indices]
        # labels
        pol_vocab = vocab[-1]
        self.labels = [pol_vocab.itos[d[-1]] for d in data]

        # example num
        self.num_examples = len(data)

        # chunk into batches
        data = [data[i: i + batch_size] for i in range(0, len(data), batch_size)]
        self.data = data
        print("{} batches created for {}".format(len(data), filename))

    def text_to_bert_sequence(self, text, max_len, padding="post", truncating="post"):
        text = tokenizer.tokenize(text)
        text = ["[CLS]"] + text + ["[SEP]"]
        sequence = tokenizer.convert_tokens_to_ids(text)
        return pad_and_truncate(sequence, max_len, padding=padding, truncating=truncating)

    def text_aspect_bert_sequence(
        self, text, aspect, aspect_len, max_len, padding="post", truncating="post"
    ):
        text_raw_indices = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
        text = "[CLS] " + text + " [SEP] " + aspect + " [SEP]"
        sequence = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
        sequence = pad_and_truncate(sequence, max_len, padding=padding, truncating=truncating)
        bert_segments_ids = np.asarray([0] * (len(text_raw_indices) + 2) + [1] * (aspect_len + 1))
        bert_segments_ids = pad_and_truncate(
            bert_segments_ids, max_len, padding=padding, truncating=truncating
        )
        return sequence, bert_segments_ids

    def preprocess(self, data, vocab, args):
        # unpack vocab
        token_vocab, post_vocab, pos_vocab, dep_vocab, pol_vocab = vocab
        processed = []

        for d in data:
            for aspect in d["aspects"]:
                # word token
                tok = list(d["token"])

                if args.lower:
                    tok = [t.lower() for t in tok]
                text = " ".join(tok)
                # aspect
                asp = list(aspect["term"])
                asp_len = len(asp)
                # label
                label = aspect["polarity"]
                # pos_tag
                pos = list(d["pos"])
                # head
                head = list(d["head"])
                # deprel
                deprel = list(d["deprel"])
                # real length
                length = len(tok)
                # position
                post = (
                    [i - aspect["from"] for i in range(aspect["from"])]
                    + [0 for _ in range(aspect["from"], aspect["to"])]
                    + [i - aspect["to"] + 1 for i in range(aspect["to"], length)]
                )
                # aspect mask
                if len(asp) == 0:
                    mask = [1 for _ in range(length)]  # for rest16
                else:
                    mask = (
                        [0 for _ in range(aspect["from"])]
                        + [1 for _ in range(aspect["from"], aspect["to"])]
                        + [0 for _ in range(aspect["to"], length)]
                    )

                text_raw_bert_indices = self.text_to_bert_sequence(text, args.max_len)
                asp_text = " ".join(asp)
                bert_sequence, bert_segments_ids = self.text_aspect_bert_sequence(
                    text, asp_text, asp_len, args.max_len
                )
                # mapping token
                tok = [token_vocab.stoi.get(t, token_vocab.unk_index) for t in tok]
                # mapping aspect
                asp = [token_vocab.stoi.get(t, token_vocab.unk_index) for t in asp]
                # mapping label
                label = pol_vocab.stoi.get(label)
                # mapping pos
                pos = [pos_vocab.stoi.get(t, pos_vocab.unk_index) for t in pos]
                # mapping head to int
                head = [int(x) for x in head]
                assert any([x == 0 for x in head])
                # mapping deprel
                deprel = [dep_vocab.stoi.get(t, dep_vocab.unk_index) for t in deprel]
                # mapping post
                post = [post_vocab.stoi.get(t, post_vocab.unk_index) for t in post]

                assert (
                    len(tok) == length
                    and len(pos) == length
                    and len(head) == length
                    and len(deprel) == length
                    and len(post) == length
                    and len(mask) == length
                )

                processed += [
                    (
                        tok,
                        asp,
                        pos,
                        head,
                        deprel,
                        post,
                        mask,
                        length,
                        text_raw_bert_indices,
                        bert_sequence,
                        bert_segments_ids,
                        label,
                    )
                ]
        return processed

    def gold(self):
        return self.labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError

        batch = self.data[key]
        batch_size = len(batch)
        batch = list(zip(*batch))

        # sort all fields by lens for easy RNN operations
        lens = [len(x) for x in batch[0]]
        batch, orig_idx = sort_all(batch, lens)

        # convert to tensors
        # token
        tok = get_long_tensor(batch[0], batch_size)
        # aspect
        asp = get_long_tensor(batch[1], batch_size)
        # pos
        pos = get_long_tensor(batch[2], batch_size)
        # head
        head = get_long_tensor(batch[3], batch_size)
        # deprel
        deprel = get_long_tensor(batch[4], batch_size)
        # post
        post = get_long_tensor(batch[5], batch_size)
        # mask
        mask = get_float_tensor(batch[6], batch_size)
        # length
        length = torch.LongTensor(batch[7])

        text_raw_bert_indices = torch.LongTensor(batch[8])

        bert_sequence = torch.LongTensor(batch[9])

        bert_segments_ids = get_long_tensor(batch[10], batch_size)
        # label
        label = torch.LongTensor(batch[11])
        # bert input

        return (
            tok,
            asp,
            pos,
            head,
            deprel,
            post,
            mask,
            length,
            text_raw_bert_indices,
            bert_sequence,
            bert_segments_ids,
            label,
        )

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)


def get_long_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.LongTensor(batch_size, token_len).fill_(0)
    for i, s in enumerate(tokens_list):
        tokens[i, : len(s)] = torch.LongTensor(s)
    return tokens


def get_float_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded FloatTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.FloatTensor(batch_size, token_len).fill_(0)
    for i, s in enumerate(tokens_list):
        tokens[i, : len(s)] = torch.FloatTensor(s)
    return tokens


def sort_all(batch, lens):
    """ Sort all fields by descending order of lens, and return the original indices. """
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]


# coding:utf-8
import sys
sys.path.append('../')
import torch
import numpy as np
import torch.nn as nn
from common.tree import head_to_adj
from common.transformer_encoder import TransformerEncoder
from common.RGAT import RGATEncoder
from transformers import BertModel, BertConfig
from torch.autograd import Variable

bert_config = BertConfig.from_pretrained("bert-base-uncased")
bert_config.output_hidden_states = True
bert_config.num_labels = 3
bert = BertModel.from_pretrained("bert-base-uncased", config=bert_config)


class RGATABSA(nn.Module):
    def __init__(self, args, emb_matrix=None):
        super().__init__()
        in_dim = args.hidden_dim + args.bert_out_dim
        self.args = args
        self.enc = ABSAEncoder(args)
        self.classifier = nn.Linear(in_dim, args.num_class)
        self.dropout = nn.Dropout(0.1)

    def forward(self, inputs, show_attn=False):
        outputs, attn_layers = self.enc(inputs, show_attn=show_attn)
        outputs = self.dropout(outputs)
        logits = self.classifier(outputs)
        try:
            assert torch.isnan(logits).sum() == 0
        except:
            # print("inputs has nan", inputs)
            # print("outputs has nan", outputs)
            # print("logit has nan", logits)
            print("alter nan to 0", logits)
            # logits = torch.where(torch.isnan(logits), torch.full_like(logits, 0), logits)
            # logits = torch.nan_to_num(logits, nan=1e-8)
            print("alter nan to 0", logits)
        return logits, outputs, attn_layers


class ABSAEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.pos_emb = (
            nn.Embedding(args.pos_size, args.pos_dim, padding_idx=0) if args.pos_dim > 0 else None
        )  # pos tag emb
        self.post_emb = (
            nn.Embedding(args.post_size, args.post_dim, padding_idx=0)
            if args.post_dim > 0
            else None
        )  # position emb
        if self.args.model.lower() in ["std", "gat"]:
            embs = (self.pos_emb, self.post_emb)
            self.encoder = DoubleEncoder(args, embeddings=embs, use_dep=True)
        elif self.args.model.lower() == "rgat":
            self.dep_emb = (
                nn.Embedding(args.dep_size, args.dep_dim, padding_idx=0)
                if args.dep_dim > 0
                else None
            )  # position emb
            embs = (self.pos_emb, self.post_emb, self.dep_emb)
            self.encoder = DoubleEncoder(args, embeddings=embs, use_dep=True)

        if self.args.output_merge.lower() == "gate":
            self.gate_map = nn.Linear(args.bert_out_dim * 2, args.bert_out_dim)
        elif self.args.output_merge.lower() == "none":
            pass
        else:
            print('Invalid output_merge type !!!')
            exit()

    def forward(self, inputs, show_attn=False):
        (
            tok,
            asp,
            pos,
            head,
            deprel,
            post,
            mask,
            l,
            text_raw_bert_indices,
            bert_sequence,
            bert_segments_ids,
            dist,
        ) = inputs  # unpack inputs
        maxlen = max(l.data)
        """
        print('tok', tok, tok.size())
        print('asp', asp, asp.size())
        print('pos-tag', pos, pos.size())
        print('head', head, head.size())
        print('deprel', deprel, deprel.size())
        print('postition', post, post.size())
        print('mask', mask, mask.size())
        print('l', l, l.size())
        """

        adj_lst, label_lst = [], []
        for idx in range(len(l)):
            adj_i, label_i = head_to_adj(
                maxlen,
                head[idx],
                tok[idx],
                deprel[idx],
                l[idx],
                mask[idx],
                directed=self.args.direct,
                self_loop=self.args.loop,
            )
            adj_lst.append(adj_i.reshape(1, maxlen, maxlen))
            label_lst.append(label_i.reshape(1, maxlen, maxlen))

        adj = np.concatenate(adj_lst, axis=0)  # [B, maxlen, maxlen]
        adj = torch.from_numpy(adj).cuda()

        labels = np.concatenate(label_lst, axis=0)  # [B, maxlen, maxlen]
        label_all = torch.from_numpy(labels).cuda()
        if self.args.model.lower() == "std":
            h = self.encoder(adj=None, inputs=inputs, lengths=l)
        elif self.args.model.lower() == "gat":
            h = self.encoder(adj=adj, inputs=inputs, lengths=l)
        elif self.args.model.lower() == "rgat": # add attention
            h = self.encoder(
                adj=adj, relation_matrix=label_all, inputs=inputs, lengths=l, show_attn=show_attn
            )
        else:
            print(
                "Invalid model name {}, it should be (std, GAT, RGAT)".format(
                    self.args.model.lower()
                )
            )
            exit(0)

        graph_out, bert_pool_output, bert_out, attn_layers = h[0], h[1], h[2], h[3]
        asp_wn = mask.sum(dim=1).unsqueeze(-1)                          # aspect words num
        mask = mask.unsqueeze(-1).repeat(1, 1, self.args.bert_out_dim)  # mask for h
        graph_enc_outputs = (graph_out * mask).sum(dim=1) / asp_wn        # mask h
        bert_enc_outputs = (bert_out * mask).sum(dim=1) / asp_wn

        if self.args.output_merge.lower() == "none":
            merged_outputs = graph_enc_outputs
        elif self.args.output_merge.lower() == "gate":
            gate = torch.sigmoid(self.gate_map(torch.cat([graph_enc_outputs, bert_enc_outputs], 1)))
            merged_outputs = gate * graph_enc_outputs + (1 - gate) * bert_enc_outputs
        else:
            print('Invalid output_merge type !!!')
            exit()
        cat_outputs = torch.cat([merged_outputs, bert_pool_output], 1)
        return cat_outputs, attn_layers

def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    return h0.cuda(), c0.cuda()

class DoubleEncoder(nn.Module):
    def __init__(self, args, embeddings=None, use_dep=False):
        super(DoubleEncoder, self).__init__()
        self.args = args
        self.Sent_encoder = bert
        self.in_drop = nn.Dropout(args.input_dropout)
        self.dense = nn.Linear(args.hidden_dim, args.bert_out_dim)  # dimension reduction
        self.use_dist = args.use_dist
        
        if use_dep:
            self.pos_emb, self.post_emb, self.dep_emb = embeddings
            self.Graph_encoder = RGATEncoder(
                num_layers=args.num_layer,
                d_model=args.bert_out_dim,
                heads=4,
                d_ff=args.hidden_dim,
                dep_dim=self.args.dep_dim,
                att_drop=self.args.att_dropout,
                dropout=0.0,
                use_structure=True
            )
        else:
            self.pos_emb, self.post_emb = embeddings
            self.Graph_encoder = TransformerEncoder(
                num_layers=args.num_layer,
                d_model=args.bert_out_dim,
                heads=4,
                d_ff=args.hidden_dim,
                dropout=0.0
            )
        if args.reset_pooling:
            self.reset_params(bert.pooler.dense)

        self.use_lstm = args.use_lstm
        if args.use_lstm:
            self.in_dim = args.bert_out_dim + args.post_dim + args.pos_dim
            input_size = self.in_dim
            self.LSTM_encoder = nn.LSTM(
                input_size,
                args.rnn_hidden,
                args.rnn_layers,
                batch_first=True,
                dropout=args.rnn_dropout,
                bidirectional=True,
            )
            # if args.bidirect:
            self.in_dim = args.rnn_hidden * 2
            # dropout
            self.rnn_drop = nn.Dropout(args.rnn_dropout)

            self.Graph_encoder = RGATEncoder(
                num_layers=args.num_layer,
                d_model=args.rnn_hidden,
                heads=4,
                d_ff=args.hidden_dim,
                dep_dim=self.args.dep_dim,
                att_drop=self.args.att_dropout,
                dropout=0.0,
                use_structure=True
            )
            self.dense_LSTM = nn.Linear(args.rnn_hidden*2, args.rnn_hidden)


    def reset_params(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)

    def encode_with_rnn(self, rnn_inputs, seq_lens, batch_size):
        h0, c0 = rnn_zero_state(
            batch_size, self.args.rnn_hidden, self.args.rnn_layers, True
        )
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens, batch_first=True)
        rnn_outputs, (ht, ct) = self.LSTM_encoder(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        return rnn_outputs

    def forward(self, adj, inputs, lengths, relation_matrix=None, show_attn=False):
        (
            tok,
            asp,
            pos,
            head,
            deprel,
            post,
            a_mask,
            l,
            text_raw_bert_indices,
            bert_sequence,
            bert_segments_ids,
            dist,
        ) = inputs  # unpack inputs

        bert_sequence = bert_sequence[:, 0:bert_segments_ids.size(1)]
        # input()
        bert_out, bert_pool_output, bert_all_out = self.Sent_encoder(
            bert_sequence, token_type_ids=bert_segments_ids, return_dict=False
        )
        bert_out = self.in_drop(bert_out)
        bert_out = bert_out[:, 0:max(l), :]
        bert_out = self.dense(bert_out)

        if adj is not None:
            mask = adj.eq(0)
        else:
            mask = None
        # print('adj mask', mask, mask.size())
        if lengths is not None:
            key_padding_mask = sequence_mask(lengths)  # [B, seq_len]
        
        if relation_matrix is not None:
            dep_relation_embs = self.dep_emb(relation_matrix)
        else:
            dep_relation_embs = None

        inp = bert_out  # [bsz, seq_len, H]

        if self.use_lstm:
            ##################################
            # embedding
            embs = [bert_out]
            if self.args.pos_dim > 0:
                embs += [self.pos_emb(pos)]
            if self.args.post_dim > 0:
                embs += [self.post_emb(post)]
            embs = torch.cat(embs, dim=2)
            embs = self.in_drop(embs)
            ##################################

            # Sentence encoding
            sent_output = self.rnn_drop(
                self.encode_with_rnn(embs, l.cpu(), tok.size()[0])
            )  # [B, seq_len, H]
            ##################################
            sent_output = self.dense_LSTM(sent_output)
            inp = sent_output
            # print(inp.shape)

        # Graph_encoder is RGATEncoder
        if not self.use_dist:
            dist = None

        graph_out, attn_layers = self.Graph_encoder(
            inp, mask=mask, src_key_padding_mask=key_padding_mask, structure=dep_relation_embs, show_attn=show_attn, weight=dist
        )               # [bsz, seq_len, H]
        return graph_out, bert_pool_output, bert_out, attn_layers


def sequence_mask(lengths, max_len=None):
    """
    create a boolean mask from sequence length `[batch_size, 1, seq_len]`
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return torch.arange(0, max_len, device=lengths.device).type_as(lengths).unsqueeze(0).expand(
        batch_size, max_len
    ) >= (lengths.unsqueeze(1))


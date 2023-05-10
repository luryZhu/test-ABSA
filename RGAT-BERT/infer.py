# encoding=utf-8
import json
import sys
sys.path.append('../')
import torch
import random
import argparse
import numpy as np
from vocab import Vocab
from utils import helper

from sklearn import metrics
from bert_loader import ABSADataLoader
from bert_trainer import ABSATrainer

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="dataset/Restaurants")
parser.add_argument("--vocab_dir", type=str, default="dataset/Restaurants")
parser.add_argument("--hidden_dim", type=int, default=768, help="bert dim.")

parser.add_argument("--dep_dim", type=int, default=30, help="dep embedding dimension.")
parser.add_argument("--pos_dim", type=int, default=0, help="pos embedding dimension.")
parser.add_argument("--post_dim", type=int, default=0, help="position embedding dimension.")
parser.add_argument("--num_class", type=int, default=3, help="Num of sentiment class.")

parser.add_argument("--input_dropout", type=float, default=0.1, help="Input dropout rate.")
parser.add_argument("--layer_dropout", type=float, default=0, help="RGAT layer dropout rate.")
parser.add_argument(
    "--att_dropout", type=float, default=0, help="self-attention layer dropout rate."
)
parser.add_argument("--lower", default=True, help="Lowercase all words.")
parser.add_argument("--direct", default=False)
parser.add_argument("--loop", default=True)
parser.add_argument("--reset_pooling", default=False, action="store_true")
parser.add_argument("--lr", type=float, default=2e-5, help="learning rate.")
parser.add_argument("--bert_lr", type=float, default=2e-5, help="learning rate for bert.")
parser.add_argument("--l2", type=float, default=1e-5, help="weight decay rate.")
parser.add_argument(
    "--optim",
    choices=["sgd", "adagrad", "adam", "adamax"],
    default="adam",
    help="Optimizer: sgd, adagrad, adam or adamax.",
)
parser.add_argument("--num_layer", type=int, default=3, help="Number of graph layers.")
parser.add_argument("--num_epoch", type=int, default=20, help="Number of total training epochs.")
parser.add_argument("--batch_size", type=int, default=16, help="Training batch size.")
parser.add_argument("--log_step", type=int, default=16, help="Print log every k steps.")
parser.add_argument(
    "--save_dir", type=str, default="./saved_models/res14", help="Root dir for saving models."
)
parser.add_argument("--model", type=str, default="SGAT", help="model to use, (std, GAT, SGAT)")
parser.add_argument("--seed", type=int, default=29)
parser.add_argument("--bert_out_dim", type=int, default=100)
parser.add_argument(
    "--output_merge",
    type=str,
    default="gatenorm2",
    help="merge method to use, (none, addnorm, add, attn, gate, gatenorm2)",
)
parser.add_argument("--max_len", type=int, default=80)

args = parser.parse_args()

# set random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.cuda.manual_seed(args.seed)
helper.print_arguments(args)

# load vocab
print("Loading vocab...")
token_vocab = Vocab.load_vocab(args.vocab_dir + "/vocab_tok.vocab")  # token
post_vocab = Vocab.load_vocab(args.vocab_dir + "/vocab_post.vocab")  # position
pos_vocab = Vocab.load_vocab(args.vocab_dir + "/vocab_pos.vocab")  # POS
dep_vocab = Vocab.load_vocab(args.vocab_dir + "/vocab_dep.vocab")  # deprel
pol_vocab = Vocab.load_vocab(args.vocab_dir + "/vocab_pol.vocab")  # polarity
vocab = (token_vocab, post_vocab, pos_vocab, dep_vocab, pol_vocab)
print(
    "token_vocab: {}, post_vocab: {}, pos_vocab: {}, dep_vocab: {}, pol_vocab: {}".format(
        len(token_vocab), len(post_vocab), len(pos_vocab), len(dep_vocab), len(pol_vocab)
    )
)
args.tok_size = len(token_vocab)
args.post_size = len(post_vocab)
args.pos_size = len(pos_vocab)
args.dep_size = len(dep_vocab)

# load data
print("Loading data from {} with batch size {}...".format(args.data_dir, args.batch_size))
test_batch = ABSADataLoader(
    args.data_dir + "/test.json", args.batch_size, args, vocab, shuffle=False
)

# check saved_models director
model_save_dir = args.save_dir
helper.ensure_dir(model_save_dir, verbose=True)


def evaluate(model, data_loader, show_attn=False):
    def get_case(batch, j, id):
        tokens, aspects, deps = data_loader.id2tags(batch[0][j], batch[1][j], batch[4][j])
        mask = batch[6][j]
        for i in range(len(mask)):
            if mask[i] == 1:
                from_idx = i
                break

        for i in range(len(mask) - 1, -1, -1):
            if mask[i] == 1:
                to_idx = i+1
                break
        return {
            "id": id,
            "tokens": tokens,
            "aspects": aspects,
            "from_to": [from_idx, to_idx],
            "deps": deps,
            "label": label[j],
            "prediction": pred[j],
            "attention": attn_layers[j].tolist(),  # 记录最后一层注意力权重
        }
    predictions, labels = [], []
    val_loss, val_acc, val_step = 0.0, 0.0, 0
    bad_case, good_case = [],[]  # 记录bad case的列表
    for i, batch in enumerate(data_loader):
        loss, acc, pred, label, _, _, attn_layers = model.predict(batch, show_attn=show_attn)
        val_loss += loss
        val_acc += acc
        predictions += pred
        labels += label
        val_step += 1
        # print(val_loss,val_acc,predictions,labels)
        if show_attn:
            # bad case
            for j in range(len(label)):
                id = i * args.batch_size + j
                if label[j] != pred[j]:
                    # print("bad case!")
                    bad_case.append(get_case(batch, j, id))
                else:
                    # add good case
                    if j%16==0:
                        good_case.append(get_case(batch, j, id))


    # f1 score
    f1_score = metrics.f1_score(labels, predictions, average="macro")
    print("total bad cases:{}, collect good cases:{}".format(len(bad_case), len(good_case)))
    return val_loss / val_step, val_acc / val_step, f1_score, bad_case, good_case

best_path = model_save_dir + "/best_model.pt"
print("Valid/Test Set: {}".format(len(test_batch)))

test_acc_history, test_loss_history, test_f1_score_history = [0.0], [0.0], [0.0]

print("Loading best checkpoint from ", best_path)
trainer = torch.load(best_path)
test_loss, test_acc, test_f1, bad_case, good_case = evaluate(trainer, test_batch, show_attn=True)
print("Evaluation Results: test_loss:{}, test_acc:{}, test_f1:{}".format(test_loss, test_acc, test_f1))
with open(model_save_dir + '/good_case.json', 'w') as file:
    json.dump(good_case, file)
with open(model_save_dir + '/bad_case.json', 'w') as file:
    json.dump(bad_case, file)
print("done write cases to", model_save_dir)
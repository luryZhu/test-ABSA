#!/bin/bash
# build vocab for different datasets
setting=new_data
dataset=Restaurants
python prepare_vocab.py --data_dir dataset/$setting/$dataset/Stanza --vocab_dir dataset/$setting/$dataset/Stanza
python prepare_vocab.py --data_dir dataset/$setting/$dataset/Biaffine --vocab_dir dataset/$setting/$dataset/Biaffine
python prepare_vocab.py --data_dir dataset/$setting/$dataset/LAL --vocab_dir dataset/$setting/$dataset/LAL
python prepare_vocab.py --data_dir dataset/$setting/$dataset/Merge --vocab_dir dataset/$setting/$dataset/Merge
#python prepare_vocab.py --data_dir dataset/$setting/Restaurants --vocab_dir dataset/$setting/Restaurants
#python prepare_vocab.py --data_dir dataset/$setting/Laptops --vocab_dir dataset/$setting/Laptops
#python prepare_vocab.py --data_dir dataset/$setting/Tweets --vocab_dir dataset/$setting/Tweets
#python prepare_vocab.py --data_dir dataset/$setting/MAMS --vocab_dir dataset/$setting/MAMS




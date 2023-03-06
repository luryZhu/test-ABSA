#!/usr/bin/env bash

source_dir=../dataset
save_dir=/kaggle/working/

#exp_setting=train
#exp_parser=Stanza/Biaffine/LAL/Merge
exp_dataset=new_data/Restaurants/$exp_parser

############# Restaurants acc:86.68 f1:80.92 #################
# saved_models/Restaurants/Test
exp_path=$save_dir
if [ ! -d "$exp_path" ]; then
  mkdir -p "$exp_path"
fi

# params
# lr=5e-6
# num_layer=2
# num_epoch=10

CUDA_VISIBLE_DEVICES=0 python3 -u bert_train.py \
	--lr 5e-6 \
	--bert_lr 2e-5 \
	--input_dropout 0.1 \
	--att_dropout 0.0 \
	--num_layer 2 \
	--bert_out_dim 100 \
	--dep_dim 100 \
	--max_len 90 \
	--data_dir $source_dir/$exp_dataset \
	--vocab_dir $source_dir/$exp_dataset \
	--save_dir $exp_path \
	--model "RGAT" \
	--seed 33 \
	--output_merge "gate" \
	--reset_pool \
	--num_epoch 10 2>&1 | tee $exp_path/infer.log

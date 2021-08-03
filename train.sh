#!/bin/sh
module load anaconda/2020.11
module load cuda/10.2
source activate py36_pt15
python trainer.py \
    --input_file ./wiki_final.txt \
    --vocab_file ./data_dev_1.1/vocabulary.txt \
    --line_batch_size 64 \
    --sample_batch_size 128 \
	--emb_dim 100 \
    --learning_rate 1e-3 \
    --num_train_epochs 50 \
    --output_dir ./embedding/ \
    --img_data_file ./data_dev_1.1/char_img_mean.pkl \
    --char2ix_file ./data_dev_1.1/char2ix.pkl \
    --seed 12345 \
    --exp_name 惠南2

# line_batch_size * sample_batch_size
# line_batch_size 一次读几个句子

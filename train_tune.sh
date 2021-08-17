python trainer.py \
    --input_file /home/stu/Documents/dataset/zhwiki_2021714/wiki_final.txt \
    --vocab_file /home/stu/LRR/trainW2C/VCWE/data_dev_1.1/vocabulary.txt \
    --line_batch_size 16 \
    --sample_batch_size 128 \
    --num_train_epochs 50 \
    --img_data_file /home/stu/LRR/trainW2C/VCWE/data_dev_1.1/char_img_mean.pkl \
    --char2ix_file /home/stu/LRR/trainW2C/VCWE/data_dev_1.1/char2ix.pkl \
    --exp_name $1 \
    # --output_dir ./embedding/ \
    # --emb_dim 128 \
    # --learning_rate 2e-3 \
    # --seed 12345 \

# line_batch_size * sample_batch_size
# line_batch_size 一次读几个句子

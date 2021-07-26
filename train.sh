python trainer.py \
    --input_file /home/stu/Documents/dataset/zhwiki_2021714/wiki_final.txt \
    --vocab_file ./data_dev_1.1/vocabulary.txt \
    --line_batch_size 16 \
    --sample_batch_size 128 \
    --learning_rate 1e-3 \
    --num_train_epochs 50 \
    --output_dir ./embedding/ \
    --img_data_file ./data_dev_1.1/char_img_mean.pkl \
    --char2ix_file ./data_dev_1.1/char2ix.pkl \
    --seed 12345 

# line_batch_size * sample_batch_size
# line_batch_size 一次读几个句子

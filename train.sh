python trainer.py \
    --input_file ./data/zh_wiki_sample_lines_500.txt \
    --vocab_file ./data/vocabulary.txt \
    --line_batch_size 1 \ 
    --sample_batch_size 128 \
    --learning_rate 1e-3 \
    --num_train_epochs 50 \
    --output_dir ./embedding/ \
    --seed 12345

# line_batch_size * sample_batch_size
# line_batch_size 一次读几个句子
#!/bin/sh

torchrun --nproc_per_node 1 example_text_completion.py --ckpt_dir /home/wjeong/.llama/checkpoints/Llama3.2-1B/ --tokenizer_path /home/wjeong/.llama/checkpoints/Llama3.2-1B/tokenizer.model --max_seq_len 64 --max_batch_size 4

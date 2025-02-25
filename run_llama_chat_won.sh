#!/bin/sh

#PYTHON_SCRIPT=example_chat_completion.py
PYTHON_SCRIPT=wjeong_chat_completion.py

#CHECKPOINTS_DIR=$HOME/.llama/checkpoints/Llama3.2-1B-Instruct
CHECKPOINTS_DIR=$HOME/.llama/checkpoints/Llama3.2-3B-Instruct

#DIALOGS_PATH=wjeong_dialogs_simple.json
#DIALOGS_PATH=dialogs/dialogs_korean.json
#DIALOGS_PATH=dialogs/dialogs_youtube.json
#DIALOGS_PATH=dialogs/dialogs_youtube_short.json
DIALOGS_PATH=dialogs/dialogs_won.json

torchrun --nproc_per_node 1 $PYTHON_SCRIPT --ckpt_dir $CHECKPOINTS_DIR --tokenizer_path $CHECKPOINTS_DIR/tokenizer.model --max_seq_len 512 --max_batch_size 6 --dialogs_path $DIALOGS_PATH

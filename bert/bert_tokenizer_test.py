################################################################################
# BERT Tokenizer Test
################################################################################

import os, sys, traceback

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from transformers import AutoTokenizer
from transformers import XLNetTokenizer
from kobert_tokenizer import KoBERTTokenizer

################################################################################
# Main
################################################################################

#model_id = "distilbert-base-uncased"
model_id = "skt/kobert-base-v1"

# tokenizer_config.json for "skt/kobert-base-v1"
# {
#   "do_lower_case": false, 
#   "remove_space": true, 
#   "keep_accents": false, 
#   "bos_token": "[CLS]", 
#   "eos_token": "[SEP]", 
#   "unk_token": "[UNK]", 
#   "sep_token": "[SEP]", 
#   "pad_token": "[PAD]", 
#   "cls_token": "[CLS]", 
#   "mask_token": {
#     "content": "[MASK]", 
#     "single_word": false, 
#     "lstrip": true, 
#     "rstrip": false, 
#     "normalized": true, 
#     "__type": "AddedToken"}, 
#   "additional_special_tokens": null, 
#   "sp_model_kwargs": {}, 
#   "tokenizer_class": "XLNetTokenizer"
# }

queries = [
    "한국어 모델을 공유합니다.",
    "원불교 대종사 삼학팔조"
]

def tokenize_queries(tokenizer, queries):
    for query in queries:
        print("=" * 80)
        print(query)
        token_ids = tokenizer.encode(query)
        print(token_ids)
        token_strs = [tokenizer.decode(token_id) for token_id in token_ids]
        print(token_strs)
        print("-" * 80)

def main():
    if True:
        print(">" * 80)
        print("> Loading by transformers.AutoTokenizer")
        print(">" * 80)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenize_queries(tokenizer, queries)

    if True:
        print(">" * 80)
        print("> Loading by transformers.XLNetTokenizer")
        print(">" * 80)
        tokenizer = XLNetTokenizer.from_pretrained(model_id)
        tokenize_queries(tokenizer, queries)

    if True:
        print(">" * 80)
        print("> Loading by kobert_tokenizer.KoBERTTokenizer")
        print(">" * 80)
        tokenizer = KoBERTTokenizer.from_pretrained(model_id)
        tokenize_queries(tokenizer, queries)

if __name__ == "__main__":
    try:
        main()
    except:
        traceback.print_exc(file=sys.stdout)

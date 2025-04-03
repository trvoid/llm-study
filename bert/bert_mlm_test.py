################################################################################
# BERT MLM Test
################################################################################

import os, sys, traceback
import json

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from transformers import AutoTokenizer, TFBertForMaskedLM, FillMaskPipeline

################################################################################
# Main
################################################################################

model_id = "bert-large-uncased"
text = "Soccer is a really fun [MASK]."

def main():
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print(tokenizer)

    model = TFBertForMaskedLM.from_pretrained(model_id)
    print(model)

    # 텍스트를 BERT 모델 입력 형태로 변환 (토큰화, ID 변환, 패딩 등)
    encoded_input = tokenizer(text, return_tensors='tf')
    print("=" * 80)
    print(encoded_input)
    token_strs = tokenizer.convert_ids_to_tokens(encoded_input["input_ids"][0])
    print(token_strs)
    print("-" * 80)

    pip = FillMaskPipeline(model=model, tokenizer=tokenizer)
    results = pip(text)

    json_str = json.dumps(results, indent=4, ensure_ascii=False)
    print(json_str)

if __name__ == "__main__":
    try:
        main()
    except:
        traceback.print_exc(file=sys.stdout)

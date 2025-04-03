################################################################################
# BERT Encode Test
################################################################################

import os, sys, traceback

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from transformers import BertTokenizer, BertModel

################################################################################
# Main
################################################################################

model_id = "bert-base-uncased"  # 다양한 사전 학습 모델 사용 가능 (e.g., bert-large-uncased, bert-base-multilingual-cased)
text = "Here is some text to encode"

def main():
    tokenizer = BertTokenizer.from_pretrained(model_id)
    print(tokenizer)

    model = BertModel.from_pretrained(model_id)
    print(model)

    # 텍스트를 BERT 모델 입력 형태로 변환 (토큰화, ID 변환, 패딩 등)
    encoded_input = tokenizer(text, return_tensors='pt')  # PyTorch 텐서로 반환
    print("=" * 80)
    print(encoded_input)
    token_strs = tokenizer.convert_ids_to_tokens(encoded_input["input_ids"][0])
    print(token_strs)
    print("-" * 80)

    # BERT 모델에 입력하여 출력 얻기
    output = model(**encoded_input)
    print("=" * 80)
    #print(output)
    print("-" * 80)

    # 출력 확인 (last_hidden_state, pooler_output 등)
    print(output.keys())
    print(output.last_hidden_state.shape)  # 마지막 은닉 상태 (sequence length, hidden size)
    print(output.pooler_output.shape)     # 풀링된 출력 (hidden size)

if __name__ == "__main__":
    try:
        main()
    except:
        traceback.print_exc(file=sys.stdout)

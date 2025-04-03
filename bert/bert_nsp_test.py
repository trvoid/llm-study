################################################################################
# BERT NSP Test
################################################################################

import os, sys, traceback
import json

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from transformers import AutoTokenizer, TFBertForNextSentencePrediction

################################################################################
# Main
################################################################################

model_id = "bert-base-uncased"
prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
next_sentence = "pizza is eaten with the use of a knife and fork. In casual settings, however, it is cut into wedges to be eaten while held in the hand."

def main():
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print(tokenizer)

    model = TFBertForNextSentencePrediction.from_pretrained(model_id)
    print(model)

    # 텍스트를 BERT 모델 입력 형태로 변환 (토큰화, ID 변환, 패딩 등)
    encoded_input = tokenizer(prompt, next_sentence, return_tensors='tf')
    print("=" * 80)
    print(encoded_input)
    token_strs = tokenizer.convert_ids_to_tokens(encoded_input["input_ids"][0])
    print(token_strs)
    print("-" * 80)

    logits = model(encoded_input['input_ids'], token_type_ids=encoded_input['token_type_ids'])[0]
    softmax = tf.keras.layers.Softmax()
    probs = softmax(logits)
    print(probs)

    print('최종 예측 레이블 :', tf.math.argmax(probs, axis=-1).numpy())

if __name__ == "__main__":
    try:
        main()
    except:
        traceback.print_exc(file=sys.stdout)

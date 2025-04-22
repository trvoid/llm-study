# DistilBERT + KorQuAD + MLM 테스트

# 1. 데이터셋 적재

from datasets import load_dataset

dataset_name = "KorQuAD/squad_kor_v1"
dataset = load_dataset(dataset_name, trust_remote_code=True)
print(dataset)

# 2. 토크나이저 적재

from transformers import AutoTokenizer

model_checkpoint = "distilbert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def tokenize_function(examples, max_length=512):
    result = tokenizer(examples["question"], 
                       examples["context"],
                       max_length=max_length, 
                       truncation="only_second"
                      )
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result

# 빠른 멀티스레딩을 작동시키기 위해서, batched=True를 지정합니다.
tokenized_datasets = dataset.map(
    tokenize_function, batched=True, remove_columns=dataset["train"].column_names
)
print(tokenized_datasets)

# 3. 데이터 전처리

#chunk_size = 128
chunk_size = 150

def group_texts(examples):
    # 모든 텍스트들을 결합한다.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # 결합된 텍스트들에 대한 길이를 구한다.
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # `chunk_size`보다 작은 경우 마지막 청크를 삭제
    total_length = (total_length // chunk_size) * chunk_size
    # max_len 길이를 가지는 chunk 단위로 슬라이스
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # 새로운 레이블 컬럼을 생성
    result["labels"] = result["input_ids"].copy()
    return result

lm_datasets = tokenized_datasets.map(group_texts, batched=True)
print(lm_datasets)

from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

import collections
import numpy as np

from transformers import default_data_collator

wwm_probability = 0.2

def whole_word_masking_data_collator(features):
    for feature in features:
        word_ids = feature.pop("word_ids")

        # 단어와 해당 토큰 인덱스 간의 map 생성
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)

        # 무작위로 단어 마스킹
        mask = np.random.binomial(1, wwm_probability, (len(mapping),))
        input_ids = feature["input_ids"]
        labels = feature["labels"]
        new_labels = [-100] * len(labels)
        for word_id in np.where(mask)[0]:
            word_id = word_id.item()
            for idx in mapping[word_id]:
                new_labels[idx] = labels[idx]
                input_ids[idx] = tokenizer.mask_token_id

    return default_data_collator(features)

#train_size = 10_000
#test_size = int(0.1 * train_size)
train_size = None
test_size = 0.1

downsampled_dataset = lm_datasets["train"].train_test_split(
    train_size=train_size, test_size=test_size, seed=42
)
print(downsampled_dataset)

# 4. 모델 적재

import torch

# GPU 사용 가능 여부 확인 및 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"
print(f"device: {device}")

from transformers import DistilBertForMaskedLM

model = DistilBertForMaskedLM.from_pretrained(model_checkpoint).to(device)

# 모델 저장 테스트
if False:
    finetuned_model_path = "./fine-tuned-distilbert-korquad-mlm"
    tokenizer.save_pretrained(finetuned_model_path)
    model.save_pretrained(finetuned_model_path)

# 5. 미세조정 훈련

from transformers import TrainingArguments, Trainer

#epochs = 4.0
epochs = 0.05
batch_size = 64
# Show the training loss with every epoch
logging_steps = len(downsampled_dataset["train"]) // batch_size
model_name = model_checkpoint.split("/")[-1]

training_args = TrainingArguments(
    output_dir=f"{model_name}-finetuned-korquad",
    overwrite_output_dir=True,
    eval_strategy="epoch",
    num_train_epochs=epochs,
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    push_to_hub=False,
    fp16=True,
    logging_steps=logging_steps,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=downsampled_dataset["train"],
    eval_dataset=downsampled_dataset["test"],
    data_collator=data_collator,
)

trainer.train()

# 6. 모델 저장 및 사용

finetuned_model_path = "./fine-tuned-distilbert-korquad-mlm"
tokenizer.save_pretrained(finetuned_model_path)
model.save_pretrained(finetuned_model_path, safe_serialization=True)

tokenzier = AutoTokenizer.from_pretrained(finetuned_model_path)
model = DistilBertForMaskedLM.from_pretrained(finetuned_model_path).to(device)

def find_topk_for_masked(tokenizer, model, text, topk=5):
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

    token_logits = model(**inputs).logits
    #print(token_logits.shape)
    
    # [MASK]의 위치를 찾고, 해당 logits을 추출합니다.
    #print(torch.where(inputs["input_ids"] == tokenizer.mask_token_id))
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
    #print(mask_token_index)
    mask_token_logits = token_logits[0, mask_token_index, :]
    #print(mask_token_logits)
    
    # 가장 큰 logits값을 가지는 [MASK] 후보를 선택합니다.
    top_5_tokens = torch.topk(mask_token_logits, topk, dim=1).indices[0].tolist()
    
    return top_5_tokens

test_texts = [
    "미세먼지가 심하면 차량 2부제와 [MASK] 비상저감조치를 시행", 
    "미[MASK]먼지가 심하면 차량 2부제와 같은 비상저감조치를 시행",
    "미세먼지가 심[MASK] 차량 2부제와 같은 비상저감조치를 시행",
    "[MASK]가 심하면 차량 2부제와 같은 비상저감조치를 시행",
    "미세먼지가 심하면 차량 2부제와 같은 [MASK]를 시행"
]
for text in test_texts:
    print(f"'input text: {text}'")
    topk_tokens = find_topk_for_masked(tokenizer, model, text, topk=5)
    for token in topk_tokens:
        print(f"'>>> {text.replace(tokenizer.mask_token, tokenizer.decode([token]))}'")

################################################################################
#
################################################################################

from huggingface_hub import login
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    TrainingArguments, 
    Trainer
)
from peft import LoraConfig, TaskType, get_peft_model
from datasets import load_dataset
#from trl import SFTTrainer

################################################################################
# 설정
################################################################################

# Base model
#model_id = "meta-llama/Llama-3.2-3B"
#save_dir = "./fine_tuned_base"

# Instruct model
model_id = "meta-llama/Llama-3.2-3B-Instruct"
save_dir = "./fine_tuned_instruct"

################################################################################
# 훈련
################################################################################

# 1. Hugging Face 로그인

login(token="your_token")

# 2. 모델과 토크나이저 적재

# Quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,  # Adjust based on your GPU
    bnb_4bit_use_double_quant=True,
)

# Load the model
model = AutoModelForCausalLM.from_pretrained(
    model_id,  # Or the path to your Llama 3.2 model
    quantization_config=bnb_config,
    device_map="auto",
)

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False, 
    r=8, 
    lora_alpha=32, 
    lora_dropout=0.1
)
model = get_peft_model(model, peft_config)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# 3. 데이터 적재

# Load your text file
dataset = load_dataset("text", data_files="data/won-gyojeon.txt", split="train")
new_dataset = dataset.train_test_split(0.05) # Let's keep 5% of the data for testing

# Tokenize the data
def tokenize_func(example):
    tokens = tokenizer(example['text'], padding="max_length", truncation=True, max_length=128)
    # Set padding token labels to -100 to ignore them in loss calculation
    tokens['labels'] = [
        -100 if token == tokenizer.pad_token_id else token for token in tokens['input_ids']
    ]
    return tokens

tokenized_dataset = new_dataset.map(tokenize_func)
tokenized_dataset = tokenized_dataset.remove_columns(['text'])

# 4. 모델 훈련 및 저장

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="steps", # To evaluate during training
    eval_steps=40,
    num_train_epochs=2,  # Adjust as needed
    per_device_train_batch_size=4,  # Adjust based on your GPU memory
    gradient_accumulation_steps=4,  # Adjust based on your GPU memory
    learning_rate=2e-4,  # Adjust as needed
    save_steps=100,
    logging_steps=10,
)

# Create the trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    args=training_args,
)

# Train the model
trainer.train()

# Save the model
trainer.save_model(save_dir)

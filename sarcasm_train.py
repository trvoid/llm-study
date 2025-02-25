import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments,
    BitsAndBytesConfig # wjeong
)
# wjeong
from peft import (
    LoraConfig, 
    TaskType, 
    get_peft_model
)
from datasets import load_dataset

# wjeong
# Quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,  # Adjust based on your GPU
    bnb_4bit_use_double_quant=True,
)

# Load the base model and tokenizer
model_id = "meta-llama/Llama-3.2-3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.float32, 
    quantization_config=bnb_config, # wjeong
    device_map="auto") # Must be float32 for MacBooks!

# wjeong
# PEFT LoRA
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False, 
    r=8, 
    lora_alpha=32, 
    lora_dropout=0.1
)
model = get_peft_model(model, peft_config)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# Load the training dataset
dataset = load_dataset("csv", data_files="data/sarcasm.csv", split="train")
print(dataset)

# Define a function to apply the chat template
def apply_chat_template(example):
    messages = [
        {"role": "user", "content": example['question']},
        {"role": "assistant", "content": example['answer']}
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return {"prompt": prompt}

# Apply the chat template function to the dataset
new_dataset = dataset.map(apply_chat_template)
print('#####')
print(new_dataset)
new_dataset = new_dataset.train_test_split(0.05) # Let's keep 5% of the data for testing
print('>>>>>')
print(new_dataset)
exit()

# Tokenize the data
def tokenize_function(example):
    tokens = tokenizer(example['prompt'], padding="max_length", truncation=True, max_length=128)
    # Set padding token labels to -100 to ignore them in loss calculation
    tokens['labels'] = [
        -100 if token == tokenizer.pad_token_id else token for token in tokens['input_ids']
    ]
    return tokens

# Apply tokenize_function to each row
tokenized_dataset = new_dataset.map(tokenize_function)
tokenized_dataset = tokenized_dataset.remove_columns(['question', 'answer', 'prompt'])

# Define training arguments
model.train()
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="steps", # To evaluate during training
    eval_steps=40,
    logging_steps=40,
    save_steps=150,
    per_device_train_batch_size=2, # Adjust based on your hardware
    per_device_eval_batch_size=2,
    num_train_epochs=2, # How many times to loop through the dataset
    fp16=False, # Must be False for MacBooks
    report_to="none", # Here we can use something like tensorboard to see the training metrics
    log_level="info",
    learning_rate=1e-5, # Would avoid larger values here
    max_grad_norm=2 # Clipping the gradients is always a good idea
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer)

# Train the model
trainer.train()

# Save the model and tokenizer
trainer.save_model("./fine-tuned-model")
tokenizer.save_pretrained("./fine-tuned-model")

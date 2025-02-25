from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from peft import LoraConfig, PeftModel
import torch

model_id = "meta-llama/Llama-3.2-3B-Instruct"
#new_model = "./output_train" # change if needed
new_model = "./output_train2" # change if needed
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4'
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    low_cpu_mem_usage=True,
    quantization_config=quantization_config,
    torch_dtype=torch.float16,
    device_map='auto'
)

model = PeftModel.from_pretrained(model, new_model)
tokenizer = AutoTokenizer.from_pretrained(model_id)

pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)

chat = [{"role": "user", "content": "Who wrote the book innovator's dilemma?"}]
prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
print('***** prompt')
print(prompt)
result = pipe(prompt)
print('***** response')
print(result[0]['generated_text'])

chat = [{"role": "user", "content": "Summarize in five lines the book 'The Brain: The story of you' written by David Eagleman."}]
prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
print('***** prompt')
print(prompt)
result = pipe(prompt)
print('***** response')
print(result[0]['generated_text'])

chat = [
    {"role": "system", "content": "너는 한국에서 발생한 원불교라는 종교에 대해 전문적인 지식을 가지고 있어."},
    {"role": "user", "content": "우리가 천지에서 입은 은혜를 한 줄로 설명해 줘."}
]
prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
print('***** prompt')
print(prompt)
result = pipe(prompt)
print('***** response')
print(result[0]['generated_text'])

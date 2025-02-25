from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from peft import LoraConfig, PeftModel
import torch

model_id = "meta-llama/Llama-3.2-3B"
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

prompt = "Who wrote the book innovator's dilemma?"
result = pipe(f"<|begin_of_text|>{prompt}")
print(result[0]['generated_text'])

prompt = "우리가 천지에서 입은 은혜를"
result = pipe(f"<|begin_of_text|>{prompt}")
print(result[0]['generated_text'])

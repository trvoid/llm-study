from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "./fine_tuned_base" #path/to/your/model/or/name/on/hub
#device = "cpu" # or "cuda" if you have a GPU
device = "cuda"

model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)

#inputs = tokenizer.encode("This movie was really", return_tensors="pt").to(device)
#inputs = tokenizer.encode("물질이 개벽되니", return_tensors="pt").to(device)
inputs = tokenizer("공익심 없는 사람을", return_tensors="pt", padding=True, max_length=128).to(device)
print(inputs)
attention_mask = inputs["attention_mask"]
print(attention_mask)

outputs = model.generate(inputs["input_ids"],
    attention_mask=attention_mask,
    pad_token_id=tokenizer.eos_token_id,
    max_new_tokens=128)

print(tokenizer.decode(outputs[0]))

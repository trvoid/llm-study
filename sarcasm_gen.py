from transformers import pipeline

model_id = "fine-tuned-model"
pipe = pipeline(
    "text-generation",
    model=model_id,
    device_map="auto",
)
messages = [
    {"role": "user", "content": "Should I move to Scandinavia?"},
]
outputs = pipe(
    messages,
    max_new_tokens=128
)
print(outputs[0]["generated_text"][-1])

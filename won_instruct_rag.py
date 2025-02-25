from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the model and tokenizer
model_id = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
device = model.device # Get the device the model is loaded on

# Define conversation input
conversation = [
    {"role": "user", "content": "What has Man always dreamed of?"}
]

# Define documents for retrieval-based generation
documents = [
    {
        "title": "The Moon: Our Age-Old Foe",
        "contents": "Man has always dreamed of destroying the moon. In this essay, I shall..."
    },
    {
        "title": "The Sun: Our Age-Old Friend",
        "contents": "Although often underappreciated, the sun provides several notable benefits..."
    }
]

# Tokenize conversation and documents using a RAG template, returning PyTorch tensors.
input_ids = tokenizer.apply_chat_template(
    conversation=conversation,
    documents=documents,
    chat_template="rag",
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt").to(device)

# Generate a response
gen_tokens = model.generate(
    input_ids,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.3,
    )

# Decode and print the generated text along with generation prompt
gen_text = tokenizer.decode(gen_tokens[0])
print(gen_text)

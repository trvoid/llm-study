from langchain_community.llms import Ollama

with open('/home/wjeong/DevData/won/won-gyojeon.txt', mode='r', encoding='utf-8') as f:
    data = f.read()

llm = Ollama(model="llama3.3")

prompt = """아래의 원불교 교전에서 일상수행의 요법을 찾아 주시오.

원불교 교전:
"""

# Concatenate the data list using new-line, and append to the prompt text.
prompt_with_data = prompt + "\r\n" + data

# Execute the prompt using streaming method
for chunks in llm.stream(prompt_with_data):
    print(chunks, end="")

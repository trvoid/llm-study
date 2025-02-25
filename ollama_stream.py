from langchain_community.llms import Ollama

llm = Ollama(model='llama3.2')
prompt = '원불교 소개'

for chunks in llm.stream(prompt):
    print(chunks, end='')

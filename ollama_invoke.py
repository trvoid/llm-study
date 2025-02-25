from langchain_community.llms import Ollama

llm = Ollama(model='llama3.2')

#prompt = 'Tell me a joke about llama'
prompt = '원불교 소개'

result = llm.invoke(prompt)
print(result)

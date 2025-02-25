################################################################################
# RAG
################################################################################

##### 1. 데이터 전처리 #####

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1) 텍스트 파일 적재

data_filepath = 'data/won-gyojeon.txt'

loader = TextLoader(data_filepath)
data = loader.load()
print(data[0].metadata)
#print(data[0].page_content)

print('##### Document Loaded !!! #####')

# 2) 텍스트 데이터 분할

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000,
    chunk_overlap=200,
    encoding_name='cl100k_base'
)
documents = text_splitter.split_documents(data)
#print(len(documents))
#print(documents[0])

print('##### Document Splitted !!! #####')

##### 2. 벡터 DB 저장 #####

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.embeddings import HuggingFaceEmbeddings

# 1) 임베딩 모델 적재

#embedding_model_name = 'jhgan/ko-sbert-nli'
embedding_model_name = 'nlpai-lab/KoE5'

embeddings_model = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings':True},
)

print('##### Embedding Model Loaded !!! #####')

# 2) 벡터 DB 생성

vectorstore = FAISS.from_documents(
    documents,
    embedding = embeddings_model,
    distance_strategy = DistanceStrategy.COSINE  
)

print('##### Vector Store Populated !!! #####')

##### 3. 추론 모델 적재 #####

from huggingface_hub import login
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
)
from langchain_huggingface import HuggingFacePipeline

# 1) Hugging Face 로그인

login(token="your_token")

# 2) 모델 적재

model_id = "meta-llama/Llama-3.1-8B-Instruct"
#model_id = "meta-llama/Llama-3.2-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)

print('##### Tokenizer Loaded !!! #####')

# Quantization config
bnb_config_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,  # Adjust based on your GPU
    bnb_4bit_use_double_quant=True,
)
bnb_config_8bit = BitsAndBytesConfig(
    load_in_8bit=True,
)

# Load the model
model = AutoModelForCausalLM.from_pretrained(
    model_id,  # Or the path to your Llama 3.2 model
    quantization_config=bnb_config_8bit,
    device_map="auto",
)

print('##### Model Loaded !!! #####')

# 3) 파이프라인 구성

from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

pipe = pipeline(
    "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200
)

hf = HuggingFacePipeline(pipeline=pipe)

print('##### Pipeline Created !!! #####')

##### 4. 프롬프트 생성 #####

#from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1) 프롬프트 템플릿 구성

system_message = '''당신은 질문-답변(Question-Answering)을 수행하는 친절한 AI 어시스턴트입니다. 당신의 임무는 주어진 문맥(context) 에서 주어진 질문(question) 에 답하는 것입니다.
검색된 다음 문맥(context) 을 사용하여 질문(question) 에 답하세요. 만약, 주어진 문맥(context) 에서 답을 찾을 수 없다면, 답을 모른다면 `주어진 정보에서 질문에 대한 정보를 찾을 수 없습니다` 라고 답하세요.
한글로 답변해 주세요. 단, 기술적인 용어나 이름은 번역하지 않고 그대로 사용해 주세요. Don't narrate the answer, just answer the question. Let's think step-by-step.'''

human_message = '''#Question: 
{question} 

#Context: 
{context} 

#Answer:
'''

#prompt_template = ChatPromptTemplate.from_template(human_message_template)
prompt_template = ChatPromptTemplate([
    ('system', system_message),
    ('human', human_message)
])

# 2) 체인 구성

chain = prompt_template | hf | StrOutputParser()

# 3) 벡터 DB 검색 및 문맥 생성

#question = '여러 가지 일들이 생기니 마음이 편치 않아. 마음을 진정시키기에 좋은 방법을 알려 줘.'
question = '정 일성이 여쭙고 대종사가 대답한 내용을 알려 줘.'

retriever = vectorstore.as_retriever(
    search_type='mmr',
    search_kwargs={'k': 5, 'lambda_mult': 0.15}
)

docs = retriever.get_relevant_documents(question)
print(len(docs))
#print(docs[-1].page_content)

# 4) 문서 합치기

def format_docs(docs):
    return '\n\n'.join([d.page_content for d in docs])

context = format_docs(docs)

prompt_value = prompt_template.invoke({'context': context, 'question':question})
print('>>>>> prompt_value.to_string() >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
print(prompt_value.to_string())
print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

##### 5. 추론 실행 #####

response = chain.invoke({'context': context, 'question':question})
print('***** chain.invoke() *******************************')
print(response)
print('****************************************************')

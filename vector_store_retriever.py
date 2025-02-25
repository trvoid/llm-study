# Load data -> Text split

from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

#loader = PyMuPDFLoader('data/kakao_bank_2023.pdf')
loader = TextLoader('data/won-gyojeon.txt')
data = loader.load()
print(data[0].metadata)
#print('>> producer: ' + data[0].metadata['producer'])
#print('>> format  : ' + data[0].metadata['format'])

#print(data[0].page_content)

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000,
    chunk_overlap=200,
    encoding_name='cl100k_base'
)
documents = text_splitter.split_documents(data)
#print(len(documents))
#print(documents[0])

print('##### Document Splitted !!! #####')

# 벡터스토어에 문서 임베딩을 저장
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings_model = HuggingFaceEmbeddings(
    model_name='jhgan/ko-sbert-nli',
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings':True},
)

print('##### Embedding Model Loaded !!! #####')

vectorstore = FAISS.from_documents(
    documents,
    embedding = embeddings_model,
    distance_strategy = DistanceStrategy.COSINE  
)

print('##### Vector Store Populated !!! #####')

# 검색 쿼리
query = '염불 좌선'

# 가장 유사도가 높은 문장을 하나만 추출
retriever1 = vectorstore.as_retriever(search_kwargs={'k': 1})

# MMR - 다양성 고려 (lambda_mult = 0.5)
retriever2 = vectorstore.as_retriever(
    search_type='mmr',
    search_kwargs={'k': 5, 'fetch_k': 50}
)

# Similarity score threshold (기준 스코어 이상인 문서를 대상으로 추출)
retriever3 = vectorstore.as_retriever(
    search_type='similarity_score_threshold',
    search_kwargs={'score_threshold': 0.2}
)

docs = retriever3.get_relevant_documents(query)
print(len(docs))
print(docs[-1].page_content)

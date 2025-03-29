################################################################################
# SentenceTransformer Test
################################################################################

import os, sys, traceback
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search

################################################################################
# Functions
################################################################################

def encode_and_calculate(model):
    # The sentences to encode
    sentences = [
        "The weather is lovely today.",
        "It's so sunny outside!",
        "He drove to the stadium.",
    ]

    # 1. Calculate embeddings by calling model.encode()
    embeddings = model.encode(sentences)
    print(embeddings.shape)
    # [3, 384]

    # 2. Calculate the embedding similarities
    similarities = model.similarity(embeddings, embeddings)
    print(similarities)
    # tensor([[1.0000, 0.6660, 0.1046],
    #         [0.6660, 1.0000, 0.1411],
    #         [0.1046, 0.1411, 1.0000]])

def search_top_k(model):
    queries = [
        "How can Medicare help me?"
    ]

    texts = [
        "How do I get a replacement Medicare card?",
        "What is the monthly premium for Medicare Part B?",
        "How do I terminate my Medicare Part B (medical insurance)?",
        "How do I sign up for Medicare?",
        "Can I sign up for Medicare Part B if I am working and have health insurance through an employer?",
        "How do I sign up for Medicare Part B if I already have Part A?",
        "What are Medicare late enrollment penalties?",
        "What is Medicare and who can get it?",
        "How can I get help with my Medicare Part A and Part B premiums?",
        "What are the different parts of Medicare?",
        "Will my Medicare premiums be higher because of my higher income?",
        "What is TRICARE ?",
        "Should I sign up for Medicare Part B if I have Veterans' Benefits?"
    ]

    query_embeddings = model.encode(queries)
    corpus_embeddings = model.encode(texts)
    
    hits = semantic_search(query_embeddings, corpus_embeddings, top_k=5)

    print([texts[hits[0][i]['corpus_id']] for i in range(len(hits[0]))])


################################################################################
# Main
################################################################################

model_id = "all-MiniLM-L6-v2"


def main():
    # 1. Load a pretrained Sentence Transformer model
    model = SentenceTransformer(model_id)

    encode_and_calculate(model)

    search_top_k(model)

if __name__ == "__main__":
    try:
        main()
    except:
        traceback.print_exc(file=sys.stdout)

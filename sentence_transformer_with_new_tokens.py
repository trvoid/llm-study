################################################################################
# SentenceTransformer training with new tokens
################################################################################

import os, sys, traceback
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader

################################################################################
# Main
################################################################################

fine_tuned_model_name = "fine_tuned_model"

def main():
    model_id = "sentence-transformers/all-MiniLM-L6-v2"

    # 1. 토크나이저에 새 토큰 추가
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print(f"토크나이저 크기: {len(tokenizer)}")
    new_tokens = ["새로운토큰1", "새로운토큰2", "new_token3"]
    num_added_tokens = tokenizer.add_tokens(new_tokens)
    print(f"{num_added_tokens}개의 토큰이 추가되었습니다.")

    # 2. 모델 적재 및 임베딩 레이어 크기 조정
    model = SentenceTransformer(model_id)
    model.tokenizer = tokenizer
    model._first_module().auto_model.resize_token_embeddings(len(tokenizer))
    print(model)

    # 3. 훈련 데이터 준비
    train_examples = [
        InputExample(texts=["anchor 문장", "positive 문장"], label=1.0),
        InputExample(texts=["anchor 문장", "새로운토큰1 포함 문장"], label=1.0),
        InputExample(texts=["다른 anchor 문장", "다른 positive 문장"], label=1.0),
    ]
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # (선택) 평가
    eval_examples = [
        InputExample(texts=["평가 anchor", "평가 positive"], label=1.0),
        InputExample(texts=["평가 anchor 문장", "평가 positive 문장"], label=1.0)
    ]
    evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(eval_examples)

    # 4. 훈련
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=10,
        warmup_steps=100,
        evaluator = evaluator,
        evaluation_steps=500,
        output_path='./output'
    )

    # 5. 모델 저장
    model.save(fine_tuned_model_name)
    print(f"SAVED MODEL: {fine_tuned_model_name}")

if __name__ == "__main__":
    try:
        main()
    except:
        traceback.print_exc(file=sys.stdout)

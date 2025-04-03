from datasets import load_dataset
from transformers import AutoTokenizer

squad = load_dataset("squad", split="train[:5000]")

squad = squad.train_test_split(test_size=0.2)

print(squad["train"][0])
print(squad["train"].column_names)

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

def preprocess_function(examples):
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print(examples)
    print('---------------------------')
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )
    #print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    #print(inputs['input_ids'][0][0:600])
    #print(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][0:600]))
    #print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    #print(inputs['offset_mapping'])
    #print('***************************************')

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

squad1 = squad['train'].select(range(2))
print(squad1)
print('~~~~~~~~~~~~~~~~~~~~~~~~')
tokenized_squad = squad1.map(preprocess_function, 
                            batched=True, 
                            batch_size=2,
                            remove_columns=squad["train"].column_names)

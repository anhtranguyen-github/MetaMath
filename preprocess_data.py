import json

if __name__ == "__main__":
    with open("./data/train/math_train.json") as f:
        data_dict = json.load(f)

    train_data = data_dict["data"]
    processed_data = []
    for sample in train_data:
        instruction = sample['question'] + ". Các đáp án: "
        for choice in sample['choices']:
            instruction += f"{choice}; "
        output = sample['explanation'] +'. ' if 'explanation' in sample.keys() else ''
        output += "Đáp án là: " + sample["answer"]
        processed_data.append({'instruction' : instruction, 'output': output})
    with open("./data/train/processed_math_train.json", "w", encoding='utf8') as f:
        json.dump(processed_data, f, ensure_ascii=False)

    with open("./data/test/math_test.json") as f:
        test_dict = json.load(f)

    test_data = test_dict["data"]
    processed_data_test = []
    for sample in test_data:
        instruction = sample['question'] + ". Các đáp án: "
        for choice in sample['choices']:
            instruction += f"{choice}; "
        processed_data_test.append({'instruction' : instruction})
    with open("./data/test/processed_math_test.json", "w", encoding='utf8') as f:
        json.dump(processed_data_test, f, ensure_ascii=False)
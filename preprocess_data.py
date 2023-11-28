import json

if __name__ == "__main__":
    with open('./data/train/math_train.json', 'r', encoding='utf-8') as f:
        data_dict = json.load(f)


    train_data = data_dict["data"]
    processed_data = []
    for sample in train_data:
        instruction = sample['question'] + ". Choose from the following: \n"
        for choice in sample['choices']:
            instruction += f"{choice}; \n"
        output = sample['explanation'] +'. ' if 'explanation' in sample.keys() else ''
        output += "\nThe answer is: " + sample["answer"]
        processed_data.append({'instruction': instruction, 'output': output})
    with open("./data/train/processed_math_train.json", "w", encoding='utf8') as f:
        json.dump(processed_data, f, ensure_ascii=False)

    with open('./data/test/math_test.json', 'r', encoding='utf-8') as f:
        test_dict = json.load(f)

    test_data = test_dict["data"]
    processed_data_test = []
    for sample in test_data:
        instruction = sample['question'] + ". Các lựa chọn: \n"
        for choice in sample['choices']:
            instruction += f"{choice}; \n"
        processed_data_test.append({'id': sample['id'], 'question': sample['question'],  'choices': sample['choices'], 'instruction' : instruction})
    with open("./data/test/processed_math_test.json", "w", encoding='utf8') as f:
        json.dump(processed_data_test, f, ensure_ascii=False)
import argparse
import json
import pdb
import jsonlines
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteriaList,
    StoppingCriteria
)
from peft import PeftModel    

import util
from vllm import LLM, SamplingParams
import sys
from tqdm import tqdm
MAX_INT = sys.maxsize
INVALID_ANS = "[invalid]"

##############################################################################################
# bitsandbytes parameters. Used if run_model_on_gpu = True. CPU doesn't support quantization
##############################################################################################

# Activate 4-bit precision base model loading
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "bfloat16"  # Efficient. Newer GPUs support bfloat16 

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

invalid_outputs = []
def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None

class StopOnTokens(StoppingCriteria):
    
    stop_token_ids = [0]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:

        for stop_id in self.stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

def process_results(doc, completion):
    print(completion)

    return completion
    # split_ans = completion.split('The answer is: ')
    # if len(split_ans) > 1:
    #     ans = split_ans[-1]
    #     extract_ans_temp = ans.split('.\n')[0]
    #     extract_ans_temp = extract_ans_temp.strip()
    #     if len(extract_ans_temp)>0 and extract_ans_temp[-1] == '.':
    #         extract_ans = extract_ans_temp[0:-1]
    #     else:
    #         extract_ans = extract_ans_temp
    #     extract_ans = extract_ans.strip()
    #     if util.is_equiv(extract_ans, answer):
    #         return True
    #     else:
    #         return False
    # else:
    #     temp = {'question': doc, 'output': completion, 'answer': answer}
    #     invalid_outputs.append(temp)
    #     return False
def batch_data(data_list, batch_size=1):
    n = len(data_list) // batch_size
    batch_data = []
    for i in range(n-1):
        start = i * batch_size
        end = (i+1)*batch_size
        batch_data.append(data_list[start:end])

    last_start = (n-1) * batch_size
    last_end = MAX_INT
    batch_data.append(data_list[last_start:last_end])
    return batch_data

def test_hendrycks_math(model, checkpoint, data_path, start=0, end=MAX_INT, batch_size=1, tensor_parallel_size=1):
    hendrycks_math_ins = []
    hendrycks_math_answers = []
    problem_prompt = (
        "Dưới đây là hướng dẫn mô tả một bài toán tiểu học. "
        "Viết lời giải để hoàn thành bài toán.\n\n"
        "### Bài toán:\n{instruction}\n\n### Hãy nghĩ theo từng bước. Câu trả lời:"
    )
    print('promt =====', problem_prompt)
    with open(data_path, "r+", encoding="utf8") as f:
        test_data = json.load(f)
        for item in test_data:
            temp_instr = problem_prompt.format(instruction=item["instruction"])
            hendrycks_math_ins.append(temp_instr)
            # solution = item['output']
            # temp_ans = remove_boxed(util.last_boxed_only_string(solution))
            # hendrycks_math_answers.append(temp_ans)

    print('total length ===', len(hendrycks_math_ins))
    hendrycks_math_ins = hendrycks_math_ins[start:end]
    #hendrycks_math_answers = hendrycks_math_answers[start:end]
    print('lenght ====', len(hendrycks_math_ins))
    batch_hendrycks_math_ins = batch_data(hendrycks_math_ins, batch_size=batch_size)

    stop_tokens = ["Question:", "Question", "USER:", "USER", "ASSISTANT:", "ASSISTANT", "Instruction:", "Instruction", "Response:", "Response"]
    stop = StopOnTokens()
    sampling_params = dict(
        temperature=0, 
        top_p=1, 
        do_sample=False,
        max_new_tokens=2048, 
        stopping_criteria=StoppingCriteriaList([stop]))

    print('sampling =====', sampling_params)
    
    
    device_map = {"": 0}
    # Load tokenizer and model with QLoRA configuration
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )
    llm = AutoModelForCausalLM.from_pretrained(
        model,
        quantization_config=bnb_config,
        device_map=device_map,
    )
    llm = PeftModel.from_pretrained(llm, checkpoint)
    llm = llm.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(model)

    # llm = LLM(model=model,tensor_parallel_size=tensor_parallel_size)


    res_completions = []
    for idx, prompt in tqdm(enumerate(batch_hendrycks_math_ins), total=len(batch_hendrycks_math_ins)):
        if isinstance(prompt, list):
            pass
        else:
            prompt = [prompt]
        input_ids = tokenizer.batch_encode_plus(
            prompt, 
            padding=True,
            truncation=True,
            return_tensors="pt").input_ids
        input_ids = input_ids.to(llm.device)

        completions = llm.generate(input_ids=input_ids, **sampling_params)
        #prompt_temp = output.prompt
        generated_text = tokenizer.batch_decode(completions, skip_special_tokens=True)
        print("Generated Text: ", generated_text)
        res_completions += generated_text

    results = []
    for idx, (prompt, completion) in enumerate(zip(hendrycks_math_ins, res_completions, hendrycks_math_answers)):
        res = process_results(prompt, completion)
        results.append(res)

    acc = sum(results) / len(results)
    print('len invalid outputs ====', len(invalid_outputs), ', valid_outputs===', invalid_outputs)
    print('start===', start, ', end====',end)
    print('length====', len(results), ', acc====', acc)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='')  # model path
    parser.add_argument("--checkpoint", type=str, default='')  # model path
    parser.add_argument("--data_file", type=str, default='')  # data path
    parser.add_argument("--start", type=int, default=0) #start index
    parser.add_argument("--end", type=int, default=MAX_INT)  # end index
    parser.add_argument("--batch_size", type=int, default=400)  # batch_size
    parser.add_argument("--tensor_parallel_size", type=int, default=8)  # tensor_parallel_size
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    test_hendrycks_math(model=args.model, checkpoint=args.checkpoint, data_path=args.data_file, start=args.start, end=args.end, batch_size=args.batch_size, tensor_parallel_size=args.tensor_parallel_size)

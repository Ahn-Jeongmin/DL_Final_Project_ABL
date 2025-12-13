import os
import json 
from qwen_inf import Infeqwen

def log_write(text, filename="log_case_1_3.txt"):
    with open(filename, "a", encoding="utf-8") as f:
        f.write(text + "\n")
def textprompt(qa_text):
    sys_prompt = (
        "You are an AI that answers based on the image and text.\n"
        "Analyze the image carefully and answer accurately."
    )

    return (
        f"{sys_prompt}\n\n"
        f"Question: {qa_text}\n"
    )
    
def main():
    filename = "anaylsis/casetest_social/social_bias_casetest.jsonl" 
    qwen='sft_grpo'
    qwen = Infeqwen(inf_type=qwen)
    data_list =[]
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue 
            record = json.loads(line)
            data_list.append(record)
    
    for data in data_list:
        img_path = "anaylsis/casetest_social/social_bias/" + data['image_filename']
        qa_dict = json.loads(data["qa"])
        
        input_text = textprompt(qa_dict['question'])
        model_output = qwen.inference_qwen(input_text,img_path)
        log_write("=============")
        log_write(f"INPUT TEXT:\n{input_text}")
        log_write(f"IMAGE PATH: {img_path}")
        log_write(f"MODEL OUTPUT:\n{model_output}")
        log_write(f"ANSWER FORMAT:\n{qa_dict['answers']}")

    return

if __name__ == "__main__":
    main()
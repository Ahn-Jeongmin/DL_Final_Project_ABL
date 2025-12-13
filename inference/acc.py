import re
import json
from pprint import pprint
import base64
import random
from tqdm import tqdm
from openai import OpenAI
from qwen_inf import Infeqwen

#open-router client config
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=""
)

def acc_ambig_dryrun(rec):
    print("==== DRY RUN: Qwen Ambig Evaluation ====")

    for data in rec[:5]:   # 앞 5개만 보기 쉽게
        image_path = "test_dataset/test_ambig_raw_images/" + data["image_filename"]
        input_text, target_answer = textprompt(data)

        print("\n-----------------------------")
        print("IMAGE PATH:", image_path)
        print("QUESTION + SHUFFLED CHOICES:")
        print(input_text)
        print("TARGET ANSWER TEXT:", target_answer)

        # 선택지 파싱
        option_texts = {}
        for line in input_text.splitlines():
            line = line.strip()
            if len(line) >= 3 and line[0] in "1234" and line[1] == ")":
                num = int(line[0])
                text = line[3:].strip()
                option_texts[num] = text

        print("PARSED OPTION TEXTS:", option_texts)

        # inference 대신 dummy output
        dummy_output = "The correct choice is 2."
        print("DUMMY MODEL OUTPUT:", dummy_output)

        match = re.search(r"[1-4]", dummy_output)
        if match:
            pred_choice_num = int(match.group())
        else:
            pred_choice_num = None

        print("PARSED PRED CHOICE NUM:", pred_choice_num)

        if pred_choice_num is not None:
            pred_answer_text = option_texts.get(pred_choice_num)
        else:
            pred_answer_text = None

        print("PRED ANSWER TEXT:", pred_answer_text)
        print("-----------------------------")

#get jsonfile data
def load_and_process_jsonl(filename):

    data_records = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue 
                
            record = json.loads(line)
            
            entity_description_str = record['entity_description'].strip()
            if entity_description_str.startswith("```json"):
                entity_description_str = entity_description_str.replace("```json\n", "").strip()
            if entity_description_str.endswith("```"):
                entity_description_str = entity_description_str.replace("\n```", "").strip()

            entity_desc = json.loads(entity_description_str)

            qa_pair = json.loads(record['qa'])
                
            extracted_data = {
                "image_filename": record['image_filename'],
                "sentences": record['sentences'],
                "count_label": record['count_label'],
                "bounding_boxes": record['boxes'],
                "question": qa_pair['question'],
                "answers": qa_pair['answers']
            }
            data_records.append(extracted_data)
            
    return data_records


def image2url(image_path):
    with open(image_path, "rb") as f:
        image_bytes = f.read()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    data_url = f"data:image/jpeg;base64,{image_b64}"  
    return data_url


def textprompt(qa_text):
    qa_question = qa_text["question"]
    answers = qa_text["answers"]
    target_answer = answers["4"]
    choices = [
        ("1", answers["1"]),
        ("2", answers["2"]),
        ("3", answers["3"]),
        ("4", answers["4"]),
    ]
    random.shuffle(choices)

    shuffled_texts = [choice_text for _, choice_text in choices]

    sys_prompt = "You are an AI language model that provides answers based on the given image and text input.\nAnalyze the image and text carefully to generate accurate and relevant responses. \nYour answer should be only a number (1, 2, 3, or 4) corresponding to the most appropriate answer choice."

    input_txt = (
        f"{sys_prompt}\n\n"
        f"Question: {qa_question}\n"
        f"1) {shuffled_texts[0]}\n"
        f"2) {shuffled_texts[1]}\n"
        f"3) {shuffled_texts[2]}\n"
        f"4) {shuffled_texts[3]}\n\n"
        "Please provide the most appropriate answer based on the image and text information."
    )

    return input_txt, target_answer
    

def textprompt_unambig(qa_text):
    qa_question = qa_text["question"]
    answers = qa_text["answers"]
    target_answer = answers["1"]
    choices = [
        ("1", answers["1"]),
        ("2", answers["2"]),
        ("3", answers["3"]),
        ("4", answers["4"]),
    ]
    random.shuffle(choices)

    shuffled_texts = [choice_text for _, choice_text in choices]

    sys_prompt = "You are an AI language model that provides answers based on the given image and text input.\nAnalyze the image and text carefully to generate accurate and relevant responses. \nYour answer should be only a number (1, 2, 3, or 4) corresponding to the most appropriate answer choice."

    input_txt = (
        f"{sys_prompt}\n\n"
        f"Question: {qa_question}\n"
        f"1) {shuffled_texts[0]}\n"
        f"2) {shuffled_texts[1]}\n"
        f"3) {shuffled_texts[2]}\n"
        f"4) {shuffled_texts[3]}\n\n"
        "Please provide the most appropriate answer based on the image and text information."
    )

    return input_txt, target_answer


def chat(model_name, text_prompt, data_url):
    completion = client.chat.completions.create(
        model= model_name,    
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
    )   
    return completion.choices[0].message.content



def acc_ambig(rec, model, qwen=None):
    correct = 0
    total = len(rec)
    if qwen:
        for data in tqdm(rec, desc=f"Evaluating {model}", unit="sample"):
            image_path = "test_dataset/test_ambig_raw_images/" + data["image_filename"]
            input_text, target_answer = textprompt(data)

            print("=============")
            print(input_text)
            print(target_answer)
            content = qwen.inference_qwen(input_text,image_path)

            match = re.search(r"[1-4]", content)
            pred_choice_num = int(match.group())
            option_texts = {}
            for line in input_text.splitlines():
                line = line.strip()
                if len(line) >= 3 and line[0] in "1234" and line[1] == ")":
                    num = int(line[0])
                    text = line[3:].strip()
                    option_texts[num] = text

            pred_answer_text = option_texts.get(pred_choice_num)
            if pred_answer_text is not None and pred_answer_text == target_answer:
                correct += 1
    else:
        for data in tqdm(rec, desc=f"Evaluating {model}", unit="sample"):
            image_path = "/home/ahnjm/dlbench/dataset/test_ambig_raw_images/" + data["image_filename"]
            input_image = image2url(image_path)
            input_text, target_answer = textprompt(data)

            content = chat(model, input_text, input_image)

            match = re.search(r"[1-4]", content)
            pred_choice_num = int(match.group())
            option_texts = {}
            for line in input_text.splitlines():
                line = line.strip()
                if len(line) >= 3 and line[0] in "1234" and line[1] == ")":
                    num = int(line[0])
                    text = line[3:].strip()
                    option_texts[num] = text

            pred_answer_text = option_texts.get(pred_choice_num)
            if pred_answer_text is not None and pred_answer_text == target_answer:
                correct += 1
            

    return correct / total if total > 0 else 0.0

def acc_unambig(rec, model, qwen=None):
    correct = 0
    total = len(rec)

    if qwen:
        for data in tqdm(rec, desc=f"Evaluating {model}", unit="sample"):
            image_path = "test_dataset/test_unambig_raw_images/" + data["image_filename"]
            input_text, target_answer = textprompt_unambig(data)
            print("=============")
            print(input_text)
            print(target_answer)
            content = qwen.inference_qwen(input_text,image_path)

            match = re.search(r"[1-4]", content)
            pred_choice_num = int(match.group())
            option_texts = {}
            for line in input_text.splitlines():
                line = line.strip()
                if len(line) >= 3 and line[0] in "1234" and line[1] == ")":
                    num = int(line[0])
                    text = line[3:].strip()
                    option_texts[num] = text

            pred_answer_text = option_texts.get(pred_choice_num)
            if pred_answer_text is not None and pred_answer_text == target_answer:
                correct += 1        
    else:
        for data in tqdm(rec, desc=f"Evaluating {model}", unit="sample"):
            image_path = "/home/ahnjm/dlbench/dataset/test_unambig_raw_images/" + data["image_filename"]
            input_image = image2url(image_path)
            input_text, target_answer = textprompt_unambig(data)

            content = chat(model, input_text, input_image)

            match = re.search(r"[1-4]", content)
            pred_choice_num = int(match.group())
            option_texts = {}
            for line in input_text.splitlines():
                line = line.strip()
                if len(line) >= 3 and line[0] in "1234" and line[1] == ")":
                    num = int(line[0])
                    text = line[3:].strip()
                    option_texts[num] = text

            pred_answer_text = option_texts.get(pred_choice_num)
            if pred_answer_text is not None and pred_answer_text == target_answer:
                correct += 1

    return correct / total if total > 0 else 0.0


def main():
    mode = 'sft_grpo'
    
    ambig = "test_dataset/test_ambig_qa_description.jsonl"
    unambig = "test_dataset/test_unambig_qa_description.jsonl"
    rec_ambig = load_and_process_jsonl(ambig)[:50] #json list records
    rec_unambig = load_and_process_jsonl(unambig) #json list records
    # mode = 'dryrun'

    # ambig = "test_dataset/test_ambig_qa_description.jsonl"
    # rec_ambig = load_and_process_jsonl(ambig)

    # if mode == 'dryrun':
    #     acc_ambig_dryrun(rec_ambig)
    #     return

    if mode == 'others':
        model = "anthropic/claude-sonnet-4.5"  
        print(model+" Evaluation Results:")
        print(acc_ambig(rec_ambig, model))
        print(acc_unambig(rec_unambig, model))
    else:
        model = "Qwen/Qwen3-VL-2B-Instruct"
        qwen = Infeqwen(inf_type=mode)
        print(mode+" Evaluation Results:")
        print(acc_ambig(rec_ambig, model, qwen))
        print(acc_unambig(rec_unambig, model, qwen))
        

if __name__ == "__main__":
    main()



# meta-llama/llama-3.2-11b-vision-instruct
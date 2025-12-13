import re
import json
import time
from pprint import pprint
import base64
import random
from tqdm import tqdm
from openai import OpenAI

from qwen_inf import Infeqwen
# ------------------------
# 🔹 GPT API Client
# ------------------------
client = OpenAI(
    api_key="api-key"
)

def log_write(text, filename="debug_log_sft_grpo.txt"):
    with open(filename, "a", encoding="utf-8") as f:
        f.write(text + "\n")
# ------------------------
# 🔹 JSONL Loader
# ------------------------
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


# ------------------------
# 🔹 Image → Base64 URL 변환
# ------------------------
def image2url(image_path):
    with open(image_path, "rb") as f:
        image_bytes = f.read()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    data_url = f"data:image/jpeg;base64,{image_b64}"
    return data_url


# ------------------------
# 🔹 Prompt 생성
# ------------------------
def textprompt(qa_text):
    sys_prompt = (
        "You are an AI that answers based on the image and text.\n"
        "Analyze the image carefully and answer accurately."
    )

    return (
        f"{sys_prompt}\n\n"
        f"Question: {qa_text['question']}\n"
    )


# ------------------------
# 🔹 GPT 모델 호출 함수
# ------------------------
def chat(model_name, text_prompt, data_url):
    response = client.responses.create(
        model=model_name,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": text_prompt},
                    {"type": "input_image", "image_url": data_url},
                ]
            }
        ]
    )

    output_text = response.output[1].content[0].text
    return output_text

# ------------------------
# 🔹 LLM Judge 함수
# ------------------------
def llmjudge(rec, model, judge_model, qwen=None):
    judge_prompt = (
        "You are an expert LLM judge evaluating whether a model produced "
        "an appropriate follow-up question to resolve ambiguity.\n\n"
        "### Scoring Format (only this format):\n"
        "<scoring>\n"
        "1. [1-5]\n"
        "2. [1-5]\n"
        "3. [1-5]\n"
        "4. [1-5]\n"
        "</scoring>\n"
        "<Validity_of_the_answer>\n"
        "yes or no\n"
        "</Validity_of_the_answer>\n\n"
        "Do not provide explanations.\n"
        "---\n\n"
        "<original_query>\n"
        "{ORIGINAL_TEXT_QUERY}\n"
        "</original_query>\n\n"
        "<model_followup_question>\n"
        "{MODEL_OUTPUT}\n"
        "</model_followup_question>\n"
    )
    
    
    total_scores = [0] * 4
    validity_counts = {'yes': 0, 'no': 0}
    total_evaluations = 0
    if qwen:
       qwen = Infeqwen(inf_type=qwen)
       for data in tqdm(rec, desc=f"Evaluating {qwen}", unit="sample"):
            # 1) 이미지 불러오기
            image_path = "test_dataset/test_ambig_raw_images/" + data["image_filename"]
            input_image = image2url(image_path) 
            # 2) 모델 답변 생성
            input_text = textprompt(data)
            model_output = qwen.inference_qwen(input_text,image_path)
            log_write("=============")
            log_write(f"INPUT TEXT:\n{input_text}")
            log_write(f"IMAGE PATH: {image_path}")
            log_write(f"MODEL OUTPUT:\n{model_output}")
            time.sleep(1)

            # 3) 재판 프롬프트 구성 + GPT Judge 호출
            judge_input = judge_prompt.format(
                ORIGINAL_TEXT_QUERY=input_text,
                MODEL_OUTPUT=model_output
            )

            judge_output = chat(judge_model, judge_input, input_image)
            log_write(f"SCORE OUTPUT:\n{judge_output}")
            time.sleep(1)

            # 4) 점수 파싱
            scores_match = re.findall(r'[1-4]\.\s*(\d+)', judge_output)
            validity_match = re.search(
                r'<Validity_of_the_answer>\s*(yes|no)\s*</Validity_of_the_answer>',
                judge_output,
                re.IGNORECASE
            )

            if len(scores_match) == 4:
                scores = list(map(int, scores_match))
                for i in range(4):
                    total_scores[i] += scores[i]
                total_evaluations += 1

            if validity_match:
                validity_counts[validity_match.group(1).lower()] += 1
    else:
        for data in tqdm(rec, desc=f"Evaluating {model}", unit="sample"):

            # 1) 이미지 불러오기
            image_path = "test_dataset/test_ambig_raw_images/" + data["image_filename"]
            input_image = image2url(image_path)

            # 2) 모델 답변 생성
            input_text = textprompt(data)
            model_output = chat(model, input_text, input_image)
            time.sleep(1)

            # 3) 재판 프롬프트 구성 + GPT Judge 호출
            judge_input = judge_prompt.format(
                ORIGINAL_TEXT_QUERY=input_text,
                MODEL_OUTPUT=model_output
            )

            judge_output = chat(judge_model, judge_input, input_image)
            time.sleep(1)

            # 4) 점수 파싱
            scores_match = re.findall(r'[1-4]\.\s*(\d+)', judge_output)
            validity_match = re.search(
                r'<Validity_of_the_answer>\s*(yes|no)\s*</Validity_of_the_answer>',
                judge_output,
                re.IGNORECASE
            )

            if len(scores_match) == 4:
                scores = list(map(int, scores_match))
                for i in range(4):
                    total_scores[i] += scores[i]
                total_evaluations += 1

            if validity_match:
                validity_counts[validity_match.group(1).lower()] += 1

    # ------------------------
    # 🔹 결과 출력
    # ------------------------
    print("\n" + "="*50)
    print(f"✨ LLM Judge 최종 평가 (총 {total_evaluations}건)")
    print("="*50)

    for i in range(4):
        print(f"항목 {i+1} 평균: {total_scores[i] / total_evaluations:.4f}")

    overall = sum(total_scores) / (total_evaluations * 4)
    print(f"\n전체 평균 점수: {overall:.4f}")

    print("\n--- Validity ---")
    print(validity_counts)

    return overall

def main():
    model = "Qwen/Qwen3-VL-2B-Instruct"
    mode = 'sft_grpo'
    judge_model = "gpt-5-mini"

    ambig = "test_dataset/test_ambig_qa_description.jsonl"
    rec_ambig = load_and_process_jsonl(ambig)

    llmjudge(rec_ambig, model, judge_model, mode)

if __name__ == "__main__":
    main()



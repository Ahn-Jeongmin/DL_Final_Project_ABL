import os
import json
import base64
import io
import re
from openai import OpenAI
from PIL import Image
import random

client = OpenAI(
    api_key=""
)


def pil_image_to_base64(pil_image):
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def strategic_judge(question,answer, data_type):
    judge_prompt = f'''### Instruction:
You are an LLM-as-judge evaluating whether the model's answer is appropriate
given the ambiguity type of the question and answer.

Ignore the original question. Judge only the model's answer relative to the data_type.

Data Type:
- "unambig": The question has a single, clear referent.
  The answer should provide a direct factual description without mentioning ambiguity.

- "ambig": The question is ambiguous. A good answer should either
  (a) request clarification naturally, or
  (b) enumerate plausible interpretations, or
  (c) explicitly resolve the ambiguity from context.
  A poor answer directly picks one referent without acknowledging ambiguity.

Scoring (1–4):
1 — Incorrect Handling**
  - For unambig: introduces ambiguity unnecessarily or refuses to answer.
  - For ambig: answers definitively without acknowledging ambiguity.

2 — Partially Appropriate**
  - For unambig: mostly correct but slightly awkward, or adds mild unnecessary speculation.
  - For ambig: recognizes ambiguity but does so weakly (e.g., vague “not sure” without asking or explaining).

3 — Good Handling**
  - For unambig: clear factual answer, no ambiguity statements.
  - For ambig: politely asks for clarification OR lists possibilities, but with minor omissions.

4 — Ideal Handling**
  - For unambig: fully natural, concise, correct factual answer.
  - For ambig: explicitly acknowledges ambiguity AND either:
      (a) naturally asks for clarification, OR
      (b) clearly enumerates plausible interpretations in natural sentences.
    Answer must be well-written and unambiguous in meta-level intent.

### Input:
- data_type: ambig  
- question: What color is that dog?
- answer: There are two dogs. Could you clarify which one you mean?  

### Output:
<score>4</score>

### Input:
- data_type: {data_type}
- question {question}
- answer: {answer}

### Output:'''

    try:
        response = client.responses.create(
            model="gpt-5-mini",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": judge_prompt},
                    ]
                }
            ]
        )
        
        output_text = response.output[1].content[0].text
        # print('ambiguous_test\n')
        # print(output_text)
        # print(data_type)
        # print(question)
        # print(answer)
        match = re.search(r"<score>\s*(\d+)\s*</score>", output_text)
        if not match:
            return 1
        return int(match.group(1))
        
    except Exception as e:
        print(f"[ERROR] GPT judgment failed: {e}")
        return 1


def factual_judge(question, answer):
    judge_prompt = f'''### Instruction:
You are a judge evaluating whether a model's answer is factually correct given the image.
- Ignore style, grammar, or completeness.
- PASS if the answer correctly refers to something that is actually visible in the image, even if it does not mention everything or omits other details.
- FAIL only if the answer mentions something not present in the image, contradicts the image, or hallucinates details.

### Output:
Return exactly one label:
- PASS
- FAIL

### Visual Question and Answer:
- Question: {question}
- Answer: {answer}

### Response:
'''

    try:
        response = client.responses.create(
            model="gpt-5-mini",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": judge_prompt},
                    ]
                }
            ]
        )

        output_text = response.output[1].content[0].text
        # print('factual test\n')
        # print(question)
        # print(answer)
        # print(output_text)
        return output_text
    except Exception as e:
        print(f"[ERROR] GPT judgment failed: {e}")
        return "FAIL"


def llm_judge_reward_function(completions, kwargs):
    images = kwargs.get('images', [])
    questions = kwargs.get('questions', [])
    data_types = kwargs.get('level', [])
    # print(data_types)
    rewards = []
    for i, completion in enumerate(completions):
        try:
            if isinstance(completion, list) and len(completion) > 0:
                if isinstance(completion[0], dict) and 'content' in completion[0]:
                    answer = completion[0]['content']
                else:
                    answer = str(completion[0])
            else:
                answer = str(completion)

            question = questions[i] if i < len(questions) else "What do you see in this image?"
            gt_type = data_types[i] if i < len(data_types) else "unambig"
            
            strategic_judgment = strategic_judge(question,answer, gt_type)
            # print(strategic_judgment)
            factual_judgment = factual_judge(question, answer)
            factual_judgment = factual_judgment.upper().strip()
            if "PASS" in factual_judgment:
                penalty = 0.0
            else:
                penalty = 0.3
                
            if strategic_judgment == 1:
                reward = 0.0
                print(f"[LLM JUDGE] Type mismatch! Reward: {reward}")
            elif strategic_judgment == 2:
                reward = 1.0 - penalty - 0.4
                print(f"[LLM JUDGE] Type mismatch! Reward: {reward}")
            elif strategic_judgment == 3:
                reward = 1.0 - penalty - 0.2
                print(f"[LLM JUDGE] Type mismatch! Reward: {reward}")
            else:
                reward = 1.0 - penalty
                print(f"[LLM JUDGE] Type mismatch! Reward: {reward}")

            rewards.append(reward)

        except Exception as e:
            print(f"[ERROR] Processing completion {i}: {e}")
            rewards.append(0.0)

    return rewards

import re
import string
import math
import json
from typing import Dict, List, Union
import numpy as np
import torch
from torch import distributed as dist
from torch import nn


ARTICLES_REGEX = re.compile(r"\b(a|an|the)\b", re.UNICODE)


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return ARTICLES_REGEX.sub(" ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def reform_for_pred(uid: str, pred: str):
    return {'id': uid, 'prediction_text': pred, 'no_answer_probability': 0.}


def reform_for_gold(uid: str, golds: list[str]):
    return {'id': uid, 'answers': {'answer_start': [], 'text': golds}}


def get_batch_PPL(logits: np.ndarray, labels: np.ndarray, ignore_index: int=-100) -> np.ndarray:
    # logits: (batch_size, seq_len, vocab_size)
    # labels: (batch_size, seq_len)
    loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
    loss: torch.FloatTensor = loss_fct(logits.reshape(-1, logits.shape[-1]), labels.view(-1))
    loss = loss.view(logits.shape[:2])
    denom: torch.FloatTensor = torch.sum((labels!=ignore_index).to(labels), dim=1)
    ppl: np.ndarray = torch.exp(torch.sum(loss, dim=1)/denom).cpu().numpy()
    return ppl


def parsing_json(mixed_text: str) -> dict[str, str]|None:
    # # json 형식이 섞여 있는 텍스트
    # mixed_text = ''' Output:
    # {
    #     "A": "There are 7 episodes in Big Little Lies season 2."
    # }</s>'''

    # 정규식을 사용하여 JSON 형식의 텍스트 찾기
    match = re.search(r'\{.*?\}', mixed_text, re.DOTALL)
    if match:
        json_text = match.group()
        # json.loads()를 사용하여 JSON 텍스트를 파싱하고 dict 변수에 저장
        try:
            parsed_dict: dict[str, str] = json.loads(json_text)
        except json.decoder.JSONDecodeError:
            parsed_dict = None
    else:
        # raise ValueError('No JSON object found in the text')
        parsed_dict = None

    # 결과 출력
    # print(parsed_dict)
    return parsed_dict

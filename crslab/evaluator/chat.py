# @Time   : 2023/6/14
# @Author : Xinyu Tang
# @Email  : txy20010310@163.com

import argparse
import copy
import json
import os
import re
import random
import time
import typing
import warnings
import tiktoken

import numpy as np
import openai
import nltk

from copy import copy
from tqdm import tqdm
from loguru import logger
from thefuzz import fuzz
from tenacity import Retrying, retry_if_not_exception_type, _utils
from tenacity.stop import stop_base
from tenacity.wait import wait_base

from crslab.config import DATASET_PATH, SAVE_PATH
from crslab.evaluator.base import BaseEvaluator
from crslab.evaluator.utils import get_entity, get_instruction

def get_exist_dialog_set(save_dir):
    exist_id_set = set()
    for file in os.listdir(save_dir):
        file_id = os.path.splitext(file)[0]
        exist_id_set.add(file_id)
    return exist_id_set

def my_before_sleep(retry_state):
    logger.debug(
        f'Retrying: attempt {retry_state.attempt_number} ended with: {retry_state.outcome}, spend {retry_state.seconds_since_start} in total')


class my_wait_exponential(wait_base):
    def __init__(
        self,
        multiplier: typing.Union[int, float] = 1,
        max: _utils.time_unit_type = _utils.MAX_WAIT,  # noqa
        exp_base: typing.Union[int, float] = 2,
        min: _utils.time_unit_type = 0,  # noqa
    ) -> None:
        self.multiplier = multiplier
        self.min = _utils.to_seconds(min)
        self.max = _utils.to_seconds(max)
        self.exp_base = exp_base

    def __call__(self, retry_state: "RetryCallState") -> float:
        if retry_state.outcome == openai.error.Timeout:
            return 0

        try:
            exp = self.exp_base ** (retry_state.attempt_number - 1)
            result = self.multiplier * exp
        except OverflowError:
            return self.max
        return max(max(0, self.min), min(result, self.max))


class my_stop_after_attempt(stop_base):
    """Stop when the previous attempt >= max_attempt."""

    def __init__(self, max_attempt_number: int) -> None:
        self.max_attempt_number = max_attempt_number

    def __call__(self, retry_state: "RetryCallState") -> bool:
        if retry_state.outcome == openai.error.Timeout:
            retry_state.attempt_number -= 1
        return retry_state.attempt_number >= self.max_attempt_number

def annotate_completion(prompt, logit_bias=None):
    if logit_bias is None:
        logit_bias = {}

    request_timeout = 20
    for attempt in Retrying(
            reraise=True,
            retry=retry_if_not_exception_type((openai.error.InvalidRequestError, openai.error.AuthenticationError)),
            wait=my_wait_exponential(min=1, max=60), stop=(my_stop_after_attempt(8))
    ):
        with attempt:
            response = openai.Completion.create(
                model='text-davinci-003', prompt=prompt, temperature=0, max_tokens=128, stop='Recommender',
                logit_bias=logit_bias,
                request_timeout=request_timeout,
            )['choices'][0]['text']
        request_timeout = min(300, request_timeout * 2)

    return response

class Chat():
    
    def __init__(self, turn_num, crs_model, dataset) -> None:
        self.turn_num = turn_num
        self.crs_model = crs_model
        self.dataset_path = os.path.join(DATASET_PATH, dataset)
        self.item_embedding_path = os.path.join(SAVE_PATH, dataset, 'embed')
        item_emb_list = []
        id2item_id = []
        
        with open(f"{self.dataset_path}/entity2id.json", 'r', encoding="utf-8") as f:
            self.entity2id = json.load(f)
        self.id2entity = {}
        for entity, idx in self.entity2id.items():
            self.id2entity[idx] = entity
        with open(f"{self.dataset_path}/id2info.json", 'r', encoding="utf-8") as f:
            self.id2info = json.load(f)

        self.id2entity = {}
        for k, v in self.entity2id.items():
            self.id2entity[int(v)] = k
        
        self.id2entityid = {}
        for id, info in self.id2info.items():
            if info['name'] in self.entity2id:
                self.id2entityid[id] = self.entity2id[info['name']]
        
        for i, file in tqdm(enumerate(os.listdir(self.item_embedding_path))):
            item_id = os.path.splitext(file)[0]
            if item_id in self.id2entityid:
                id2item_id.append(item_id)

                with open(f'{self.item_embedding_path}/{file}', encoding='utf-8') as f:
                    embed = json.load(f)
                    item_emb_list.append(embed)

        self.id2item_id_arr = np.asarray(id2item_id)
        self.item_emb_arr = np.asarray(item_emb_list)
    

    def chat(self, batch, turn_num):
        
        recommender_instruction, seeker_instruction_template = get_instruction()
        
        contexts_batch = batch['context']
        items_batch = batch['item']
        entity_batch = batch['entity']
        
        for context, items, entities in zip(contexts_batch, items_batch, entity_batch):
        
            goal_item_list = [self.id2entity[item] for item in items]
            goal_item_str = ', '.join(goal_item_list)
            seeker_prompt = seeker_instruction_template.format(goal_item_str, goal_item_str, goal_item_str, goal_item_str)
            
            context_list = []
            if len(context) % 2 == 0:
                context = [""] + context
            
            for i, text in enumerate(context):
                if len(text) == 0:
                    continue
                if i % 2 == 0:
                    seeker_prompt += f'Seeker: {text}\n'
                    context_list.append({
                        'role': 'user',
                        'content': text,
                    })
                else:
                    seeker_prompt += f'Recommender: {text}\n'
                    context_list.append({
                        'role': 'assistant',
                        'content': text
                    })
            
            rec_success = False
            recommendation_template = "I would recommend the following items: {}:"
            
            for i in range(0, turn_num):
                # rec only
                _, item_rank_arr = self.crs_model.recommend(batch)
                
                pred_items = item_rank_arr[0]
                
                for rec_label in items:
                    if rec_label in pred_items:
                        rec_success = True
                        break
                
                _, recommender_text = self.crs_model.converse(batch)
                
                if rec_success == True or i == turn_num - 1:
                    rec_items_str = ''
                    for j, rec_item in enumerate(pred_items):
                        rec_items_str += f"{j+1}: {self.id2entity[rec_item]}\n"
                    recommendation_template = recommendation_template.format(rec_items_str)
                    recommender_text = recommendation_template + recommender_text
                    
                recommender_resp_entity = get_entity(recommender_text, self.entity2id)
                
                context.append(recommender_text)
                entities += recommender_resp_entity
                entities = list(set(entities))
                
                batch['context'] = [copy(context)]
                batch['entity'] = [copy(entities)]
                
                context_list.append({
                    'role': 'assistant',
                    'content': recommender_text,
                    'entity': recommender_resp_entity,
                    'pred_items': pred_items,
                    'rec_items': items,
                    'rec_success': rec_success
                })
                
                seeker_prompt += f'Recommender: {recommender_text}\nSeeker:'
                
                # seeker
                year_pattern = re.compile(r'\(\d+\)')
                goal_item_no_year_list = [year_pattern.sub('', rec_item).strip() for rec_item in goal_item_list]
                seeker_text = annotate_completion(seeker_prompt).strip()
                
                seeker_response_no_movie_list = []
                for sent in nltk.sent_tokenize(seeker_text):
                    use_sent = True
                    for rec_item_str in goal_item_list + goal_item_no_year_list:
                        if fuzz.partial_ratio(rec_item_str.lower(), sent.lower()) > 90:
                            use_sent = False
                            break
                    if use_sent is True:
                        seeker_response_no_movie_list.append(sent)
                seeker_response = ' '.join(seeker_response_no_movie_list)
                if not rec_success:
                    seeker_response = 'Sorry, ' + seeker_response
                seeker_prompt += f' {seeker_response}\n'
                
                # public
                seeker_resp_entity = get_entity(seeker_text, self.entity2id)
                
                context_list.append({
                    'role': 'user',
                    'content': seeker_text,
                    'entity': seeker_resp_entity,
                })
                
                context.append(seeker_text)
                entities += seeker_resp_entity
                entities = list(set(entities))
                
                batch['context'] = [copy(context)]
                batch['entity'] = [copy(entities)]
                
                if rec_success:
                    break
                
            # score persuativeness
            encoding = tiktoken.encoding_for_model("text-davinci-003")
            logit_bias = {encoding.encode(str(score))[0]: 10 for score in range(3)}
            
            persuasiveness_template = '''Does the explanation make you want to accept the recommendation? Please give your score.
    If mention one of [{}], give 2.
    Else if you think recommended items are worse than [{}], give 0.
    Else if you think recommended items are comparable to [{}] according to the explanation, give 1.
    Else if you think recommended items are better than [{}] according to the explanation, give 2.
    Only answer the score number.'''

            persuasiveness_template = persuasiveness_template.format(goal_item_str, goal_item_str, goal_item_str, goal_item_str)
            prompt_str_for_persuasiveness = seeker_prompt + persuasiveness_template
            prompt_str_for_persuasiveness += "\nSeeker:"
            persuasiveness_score = annotate_completion(prompt_str_for_persuasiveness, logit_bias).strip()
            
            context_list.append({
                'persuasiveness_score': persuasiveness_score
            })
        
        return context_list
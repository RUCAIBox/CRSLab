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

class Ask():
    
    def __init__(self, turn_num, crs_model, dataset, ask_instruction_dict) -> None:
        self.turn_num = turn_num
        self.crs_model = crs_model
        self.ask_instruction_dict = ask_instruction_dict
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
        
        self.entityid2id = {}
        for id, entityid in self.id2entityid.items():
            self.entityid2id[entityid] = id
        
        for i, file in tqdm(enumerate(os.listdir(self.item_embedding_path))):
            item_id = os.path.splitext(file)[0]
            if item_id in self.id2entityid:
                id2item_id.append(item_id)

                with open(f'{self.item_embedding_path}/{file}', encoding='utf-8') as f:
                    embed = json.load(f)
                    item_emb_list.append(embed)

        self.id2item_id_arr = np.asarray(id2item_id)
        self.item_emb_arr = np.asarray(item_emb_list)
        
    def ask(self, batch, turn_num):
        
        ask_instruction = self.ask_instruction_dict['ask_instruction']
        option2attr = self.ask_instruction_dict['option2attr']
        option2template = self.ask_instruction_dict['option2template']
        rec_instruction = self.ask_instruction_dict['rec_instruction']
        recommendation_template = "I would recommend the following items:\n{}"
        
        contexts_batch = batch['context']
        items_batch = batch['item']
        entity_batch = batch['entity']
        
        for context, items, entities in zip(contexts_batch, items_batch, entity_batch):
            
            context_list = [] 
            
            if len(context) % 2 == 0:
                context = [""] + context
            
            for i, text in enumerate(context):
                if len(text) == 0:
                    continue
                if i % 2 == 0:
                    role_str = 'user'
                else:
                    role_str = 'assistant'
                context_list.append({
                    'role': role_str,
                    'content': text
                })
                
            rec_success = False
            option2index = {
                'A': 0,
                'B': 1,
                'C': 2,
                'D': 3,
                'E': 4
            }
            
            options = list(option2attr.keys())
            state = [0 for _ in range(len(options))]

            for i in range(0, turn_num):
                # seeker
                
                context_list.append({
                    'role': 'user',
                    'content': ask_instruction
                })
                
                context.append(ask_instruction)
                batch['context'] = [copy(context)]

                # recommender
                # choose option
                # options (list of str): available options, generate one of them
                gen_inputs, recommender_text = self.crs_model.converse(batch)
                recommender_choose = self.crs_model.choose(gen_inputs, options, state, batch)
                selected_option = recommender_choose 

                if selected_option == options[-1]:  # choose to rec
                    # recommender
                    _, item_rank_arr = self.crs_model.recommend(batch)
                    pred_items = item_rank_arr[0]

                    rec_items_str = ''
                    for j, rec_item in enumerate(pred_items):
                        rec_items_str += f"{j + 1}: {self.id2entity[rec_item]}\n"
                    recommender_text = recommendation_template.format(rec_items_str)

                    # judge whether success
                    for rec_label in items:
                        if rec_label in pred_items:
                            rec_success = True
                            break
                        
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

                    # seeker
                    if rec_success is True:
                        seeker_text = "That's perfect, thank you!"
                    else:
                        seeker_text = "I don't like them."

                    context_list.append({
                        'role': 'user',
                        'content': seeker_text
                    })
                    
                    context.append(seeker_text)
                    batch['context'] = [copy(context)]

                else:  # choose to ask
                    recommender_text = option2template[selected_option]
                    context_list.append({
                        'role': 'assistant',
                        'content': recommender_text,
                    })
                    context.append(recommender_text)
                    batch['context'] = [copy(context)]
                
                    # seeker
                    ask_attr = option2attr[selected_option]
                    
                    # update state
                    state[option2index[selected_option]] = -1e5
                    
                    ans_attr_list = []
                    id2info_items = [self.entityid2id[item] for item in items]
                    for id2info_item in id2info_items:
                        if str(id2info_item) in self.id2info and ask_attr in self.id2info[str(id2info_item)]:
                            ans_attr_list.extend(self.id2info[str(id2info_item)][ask_attr])
                    if len(ans_attr_list) > 0:
                        seeker_text = ', '.join(list(set(ans_attr_list)))
                    else:
                        seeker_text = 'Sorry, no information about this, please choose another option.'

                    context_list.append({
                        'role': 'user',
                        'content': seeker_text,
                        'entity': ans_attr_list,
                    })
                    
                    seeker_resp_entities = get_entity(seeker_text, self.entity2id)
                    
                    context.append(seeker_text)
                    entities += seeker_resp_entities
                    entities = list(set(entities))
                    
                    batch['context'] = [copy(context)]
                    batch['entity'] = [copy(entities)]
                
                if rec_success is True:
                    break
        
        return context_list
            
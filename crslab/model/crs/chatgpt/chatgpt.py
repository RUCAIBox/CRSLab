# @Time   : 2023/6/14
# @Author : Xinyu Tang
# @Email  : txy20010310@163.com

import json
import os
import numpy as np
import openai
import typing
import tiktoken

from tqdm import tqdm
from loguru import logger
from copy import copy
from tenacity import Retrying, retry_if_not_exception_type, _utils
from tenacity.stop import stop_base
from tenacity.wait import wait_base
from sklearn.metrics.pairwise import cosine_similarity

from crslab.config import DATASET_PATH, SAVE_PATH
from crslab.model.base import BaseModel

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

def annotate(item_text_list):
        request_timeout = 6
        for attempt in Retrying(
            reraise=True, retry=retry_if_not_exception_type((openai.error.InvalidRequestError, openai.error.AuthenticationError)),
            wait=my_wait_exponential(min=1, max=60), stop=(my_stop_after_attempt(8)), before_sleep=my_before_sleep
        ):
            with attempt:
                response = openai.Embedding.create(
                    model='text-embedding-ada-002', input=item_text_list, request_timeout=request_timeout
                )
            request_timeout = min(30, request_timeout * 2)

        return response

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

def annotate_chat(messages, logit_bias=None):
    if logit_bias is None:
        logit_bias = {}

    request_timeout = 20
    for attempt in Retrying(
        reraise=True, retry=retry_if_not_exception_type((openai.error.InvalidRequestError, openai.error.AuthenticationError)),
        wait=my_wait_exponential(min=1, max=60), stop=(my_stop_after_attempt(8)), before_sleep=my_before_sleep
    ):
        with attempt:
            response = openai.ChatCompletion.create(
                model='gpt-3.5-turbo', messages=messages, temperature=0, logit_bias=logit_bias,
                request_timeout=request_timeout,
            )['choices'][0]['message']['content']
        request_timeout = min(300, request_timeout * 2)

    return response

class ChatGPTModel(BaseModel):
    
    def __init__(self, opt, device, vocab=None, side_data=None):
        self.dataset = opt['dataset']
        self.dataset_path = os.path.join(DATASET_PATH, self.dataset)
        self.item_embedding_path = os.path.join(SAVE_PATH, self.dataset, 'embed')
        
        with open(f"{self.dataset_path}/entity2id.json", 'r', encoding="utf-8") as f:
            self.entity2id = json.load(f)
        self.id2entity = {}
        for entity, idx in self.entity2id.items():
            self.id2entity[idx] = entity
        with open(f"{self.dataset_path}/id2info.json", 'r', encoding="utf-8") as f:
            self.id2info = json.load(f)
        
        self.id2entityid = {}
        for id, info in self.id2info.items():
            if info['name'] in self.entity2id:
                self.id2entityid[id] = self.entity2id[info['name']]
        
        self.get_item_embedding()
        super(ChatGPTModel, self).__init__(opt, device)
        
    def build_model(self, *args, **kwargs):
        return super().build_model(*args, **kwargs)
    
    def get_item_embedding(self):
        
        item_emb_list = []
        id2item_id = []
        
        if os.path.exists(self.item_embedding_path):
            for i, file in tqdm(enumerate(os.listdir(self.item_embedding_path))):
                item_id = os.path.splitext(file)[0]
                if item_id in self.id2entityid:
                    id2item_id.append(item_id)

                    with open(f'{self.item_embedding_path}/{file}', encoding='utf-8') as f:
                        embed = json.load(f)
                        item_emb_list.append(embed)

            self.id2item_id_arr = np.asarray(id2item_id)
            self.item_emb_arr = np.asarray(item_emb_list)
    
    def get_instruction(self):
    
        recommender_instruction = '''You are a recommender chatting with the user to provide recommendation. You must follow the instructions below during chat.
If you do not have enough information about user preference, you should ask the user for his preference.
If you have enough information about user preference, you can give recommendation. The recommendation list must contain 10 items that are consistent with user preference. The recommendation list can contain items that the dialog mentioned before. The format of the recommendation list is: no. title. Don't mention anything other than the title of items in your recommendation list.
'''
        
        seeker_instruction_template = '''You are a seeker chatting with a recommender for recommendation. Your target items: {}. You must follow the instructions below during chat.
    If the recommender recommend {}, you should accept.
    If the recommender recommend other items, you should refuse them and provide the information about {}. You should never directly tell the target item title.
    If the recommender asks for your preference, you should provide the information about {}. You should never directly tell the target item title\n.
    '''

        return recommender_instruction, seeker_instruction_template
    
    def recommend(self, batch, mode='test'):
        
        context_batch = batch['context']
        item_rank_arr_batch = []
        
        for context in context_batch:
        
            if len(context) % 2 == 0:
                context = [""] + context
                
            conv_str = ""
            
            for i, text in enumerate(context[-2:]):
                if len(text) == 0:
                    continue
                if i % 2 == 0:
                    conv_str += f'Seeker: {text}\n'
                else:
                    conv_str += f'Recommender: {text}\n'
            conv_embed = annotate(conv_str)['data'][0]['embedding']
            conv_embed = np.asarray(conv_embed).reshape(1, -1)
            
            sim_mat = cosine_similarity(conv_embed, self.item_emb_arr)
            rank_arr = np.argsort(sim_mat, axis=-1).tolist()
            rank_arr = np.flip(rank_arr, axis=-1)[:, :50]
            item_rank_arr = self.id2item_id_arr[rank_arr].tolist()
            # modify item_rank_arr, item_id -> entity_id
            item_rank_arr = [self.id2entityid[item_id] for item_id in item_rank_arr[0]]
            item_rank_arr_batch.append(item_rank_arr)
            
            loss = None
        
        return loss, item_rank_arr_batch
                
    def converse(self, batch, mode='test'):
        recommender_instruction, seeker_instruction_template = self.get_instruction()
        
        context_batch = batch['context']
        item_batch = batch['item']

        for context, items in zip(context_batch, item_batch):
            
            context_list = [{
                'role': 'system',
                'content': recommender_instruction
            }]
            
            if len(context) % 2 == 0:
                context = [""] + context
            
            for i, text in enumerate(context):
                if len(text) == 0:
                    continue
                if i % 2 == 0:
                    context_list.append({
                        'role': 'user',
                        'content': text,
                    })
                else:
                    context_list.append({
                        'role': 'assistant',
                        'content': text
                    })
            
            gen_str = annotate_chat(context_list)
            gen_inputs = None
        
        return gen_inputs, gen_str
    
    def choose(self, gen_inputs, options, state, batch, mode='test'):
        
        context_batch = batch['context']
        
        for context in context_batch:
        
            updated_options = []
            for i, st in enumerate(state):
                if st >= 0:
                    updated_options.append(options[i])
                    
            logger.info(updated_options)
                    
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            logit_bias = {encoding.encode(option)[0]: 20 for option in updated_options}
            
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
            
            logger.info(context_list)
            
            response_op = annotate_chat(context_list, logit_bias=logit_bias)
            return response_op[0]
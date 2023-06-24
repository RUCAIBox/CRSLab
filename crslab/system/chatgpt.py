# @Time   : 2023/6/14
# @Author : Xinyu Tang
# @Email  : txy20010310@163.com

import os
import json
import random

import openai
from tqdm import tqdm
from loguru import logger
from tenacity import _utils, Retrying, retry_if_not_exception_type
from tenacity.stop import stop_base
from tenacity.wait import wait_base

from crslab.config import DATASET_PATH, SAVE_PATH
from crslab.system.base import BaseSystem
from crslab.data.dataset import BaseDataset
from crslab.data import dataset_register_table
from crslab.model import Model_register_table
from crslab.evaluator.chat import my_wait_exponential, my_stop_after_attempt, Chat
from crslab.evaluator.ask import Ask
from crslab.evaluator.rec import RecEvaluator
from crslab.system.utils.functions import get_exist_item_set

def get_exist_dialog_set(save_dir):
    exist_id_set = set()
    for file in os.listdir(save_dir):
        file_id = os.path.splitext(file)[0]
        exist_id_set.add(file_id)
    return exist_id_set

class ChatGPTSystem(BaseSystem):
    
    def __init__(self, opt, train_dataloader, valid_dataloader, test_dataloader, vocab, side_data, restore_system=False, interact=False, debug=False, tensorboard=False):
        super(ChatGPTSystem, self).__init__(opt, train_dataloader, valid_dataloader, test_dataloader, vocab, side_data, restore_system, interact, debug, tensorboard)
        
        openai.api_key = opt['api_key']
        self.dataset = opt['dataset']
        self.dpath = os.path.join(DATASET_PATH, opt['dataset'])
        self.embed_save_dir = os.path.join(SAVE_PATH, self.dataset, 'embed')
        os.makedirs(self.embed_save_dir, exist_ok=True)
        self.chat_save_dir = os.path.join(SAVE_PATH, self.dataset, 'chat')
        os.makedirs(self.chat_save_dir, exist_ok=True)
        self.ask_save_dir = os.path.join(SAVE_PATH, self.dataset, 'ask')
        os.makedirs(self.ask_save_dir, exist_ok=True)
        with open(os.path.join(self.dpath, 'id2info.json'), 'r', encoding='utf-8') as f:
            self.id2info = json.load(f)
        self.test_dataloader = test_dataloader
        self.rec_optim_opt = opt['rec']
        self.conv_optim_opt = opt['conv']
        self.cache_item_opt = opt['cache_item']
        self.rec_batch_size = self.rec_optim_opt['batch_size']
        self.conv_batch_size = self.conv_optim_opt['batch_size']
        self.cache_item_batch_size= self.cache_item_opt['batch_size']
        self.api_key = opt['api_key']
        self.turn_num = opt['turn_num']
        self.dataset_class = dataset_register_table[self.dataset](opt, opt['tokenize'])
        crs_model_name = opt['model_name']
        self.crs_model = Model_register_table[crs_model_name](opt, opt['device'])
    
    def my_before_sleep(self, retry_state):
        logger.debug(f'Retrying: attempt {retry_state.attempt_number} ended with: {retry_state.outcome}, spend {retry_state.seconds_since_start} in total')
    
    def annotate(self, item_text_list):
        request_timeout = 6
        for attempt in Retrying(
            reraise=True, retry=retry_if_not_exception_type((openai.error.InvalidRequestError, openai.error.AuthenticationError)),
            wait=my_wait_exponential(min=1, max=60), stop=(my_stop_after_attempt(8)), before_sleep=self.my_before_sleep
        ):
            with attempt:
                response = openai.Embedding.create(
                    model='text-embedding-ada-002', input=item_text_list, request_timeout=request_timeout
                )
            request_timeout = min(30, request_timeout * 2)

        return response
        
    def cache_item(self):
        attr_list = self.dataset_class.get_attr_list()
        id2text = {}
        for item_id, info_dict in self.id2info.items():
            attr_str_list = [f'Title: {info_dict["name"]}']
            for attr in attr_list:
                if attr not in info_dict:
                    continue
                if isinstance(info_dict[attr], list):
                    value_str = ', '.join(info_dict[attr])
                else:
                    value_str = info_dict[attr]
                attr_str_list.append(f'{attr.capitalize()}: {value_str}')
            item_text = '; '.join(attr_str_list)
            id2text[item_id] = item_text
        
        item_ids = set(self.id2info.keys()) - get_exist_item_set(self.embed_save_dir)
        while len(item_ids) > 0:
            logger.info(len(item_ids))
            batch_item_ids = random.sample(tuple(item_ids), min(self.cache_item_batch_size, len(item_ids)))
            batch_texts = [id2text[item_id] for item_id in batch_item_ids]

            batch_embeds = self.annotate(batch_texts)['data']
            for embed in batch_embeds:
                item_id = batch_item_ids[embed['index']]
                with open(f'{self.embed_save_dir}/{item_id}.json', 'w', encoding='utf-8') as f:
                    json.dump(embed['embedding'], f, ensure_ascii=False)

            item_ids -= get_exist_item_set(self.embed_save_dir)
        
    
    def iEvaLM_chat(self):
        logger.info('[Test]')
        iEvaLM_CHAT = Chat(self.turn_num, self.crs_model, self.dataset)
        dataid2data = {}
        for i, batch in enumerate(self.test_dataloader.get_conv_data(batch_size=self.conv_batch_size, shuffle=False)):
            dialog_ids = batch['dialog_id']
            for dialog_id in dialog_ids:
                dataid2data[dialog_id] = batch
        
        dialog_id_set = set(dataid2data.keys()) - get_exist_dialog_set(self.chat_save_dir)
        while len(dialog_id_set) > 0:
            logger.info(len(dialog_id_set))
            dialog_id = random.choice(tuple(dialog_id_set))
            batch_data = dataid2data[dialog_id]
            returned_data = iEvaLM_CHAT.chat(batch_data, self.turn_num)
            
            with open(f'{self.chat_save_dir}/{dialog_id}.json', 'w', encoding='utf-8') as f:
                json.dump(returned_data, f, ensure_ascii=False, indent=2)

            dialog_id_set -= get_exist_dialog_set(self.chat_save_dir)
    
    def iEvaLM_ask(self):
        logger.info('[Test]')
        ask_instruction_dict = self.dataset_class.get_ask_instruction()
        iEvaLM_ASK = Ask(self.turn_num, self.crs_model, self.dataset, ask_instruction_dict)
        dataid2data = {}
        for i, batch in enumerate(self.test_dataloader.get_conv_data(batch_size=self.conv_batch_size, shuffle=False)):
            dialog_ids = batch['dialog_id']
            for dialog_id in dialog_ids:
                dataid2data[dialog_id] = batch
        
        dialog_id_set = set(dataid2data.keys()) - get_exist_dialog_set(self.ask_save_dir)
        while len(dialog_id_set) > 0:
            logger.info(len(dialog_id_set))
            dialog_id = random.choice(tuple(dialog_id_set))
            batch_data = dataid2data[dialog_id]
            returned_data = iEvaLM_ASK.ask(batch_data, self.turn_num)
            
            with open(f'{self.ask_save_dir}/{dialog_id}.json', 'w', encoding='utf-8') as f:
                json.dump(returned_data, f, ensure_ascii=False, indent=2)

            dialog_id_set -= get_exist_dialog_set(self.ask_save_dir)
    
    def step(self, batch, stage, mode):
        return super().step(batch, stage, mode)
    
    def evaluate_iEvaLM(self, mode):
        metric = RecEvaluator(k_list=[1, 10, 25, 50])
        persuatiness_list = []
        if mode == 'chat':
            save_path = self.chat_save_dir
        elif mode == 'ask':
            save_path = self.ask_save_dir
        if os.path.exists(save_path) and len(os.listdir(save_path)) > 0:
            path_list = os.listdir(save_path)
            print(save_path, len(path_list))
            
            for path in tqdm(path_list):
                with open(f"{save_path}/{path}", 'r', encoding="utf-8") as f:
                    context_list = json.load(f)
                    if mode == 'chat':
                        persuasiveness_score = context_list[-1]['persuasiveness_score']
                        persuatiness_list.append(float(persuasiveness_score))
                    # TODO： modify chatgpt evaluator
                    for context in context_list[::-1]:
                        if 'rec_items' in context:
                            rec_labels = context['rec_items']
                            rec_items = context['pred_items']
                            for rec_label in rec_labels:
                                metric.rec_evaluate(rec_items, rec_label)
                            break
        
        metric.report(mode='test')
        if mode == 'chat':
            avg_persuatiness_score = sum(persuatiness_list) / len(persuatiness_list)
            logger.info(avg_persuatiness_score)
    
    def fit(self, mode):
        self.cache_item()
        if mode == 'chat':
            self.iEvaLM_chat()
        elif mode == 'ask':
            self.iEvaLM_ask()
        else:
            raise ValueError(f'Invalid mode: {mode}')
        self.evaluate_iEvaLM(mode)

    def interact(self):
        pass
    
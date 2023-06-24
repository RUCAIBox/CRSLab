# @Time   : 2020/12/9
# @Author : Yuanhang Zhou
# @Email  : sdzyh002@gmail.com

# UPDATE:
# @Time   : 2021/1/3
# @Author : Xiaolei Wang
# @Email  : wxl1999@foxmail.com

import os

import torch
from loguru import logger
from math import floor

from crslab.config import PRETRAIN_PATH
from crslab.data import get_dataloader, dataset_language_map
from crslab.evaluator.metrics.base import AverageMetric
from crslab.evaluator.metrics.gen import PPLMetric
from crslab.system.base import BaseSystem
from crslab.system.utils.functions import ind2txt


class TGReDialSystem(BaseSystem):
    """This is the system for TGReDial model"""

    def __init__(self, opt, train_dataloader, valid_dataloader, test_dataloader, vocab, side_data, restore_system=False,
                 interact=False, debug=False, tensorboard=False):
        """

        Args:
            opt (dict): Indicating the hyper parameters.
            train_dataloader (BaseDataLoader): Indicating the train dataloader of corresponding dataset.
            valid_dataloader (BaseDataLoader): Indicating the valid dataloader of corresponding dataset.
            test_dataloader (BaseDataLoader): Indicating the test dataloader of corresponding dataset.
            vocab (dict): Indicating the vocabulary.
            side_data (dict): Indicating the side data.
            restore_system (bool, optional): Indicating if we store system after training. Defaults to False.
            interact (bool, optional): Indicating if we interact with system. Defaults to False.
            debug (bool, optional): Indicating if we train in debug mode. Defaults to False.
            tensorboard (bool, optional) Indicating if we monitor the training performance in tensorboard. Defaults to False. 

        """
        super(TGReDialSystem, self).__init__(opt, train_dataloader, valid_dataloader,
                                             test_dataloader, vocab, side_data, restore_system, interact, debug,
                                             tensorboard)

        if hasattr(self, 'conv_model'):
            self.ind2tok = vocab['conv']['ind2tok']
            self.end_token_idx = vocab['conv']['end']
        if hasattr(self, 'rec_model'):
            self.item_ids = side_data['rec']['item_entity_ids']
            self.id2entity = vocab['rec']['id2entity']

        if hasattr(self, 'rec_model'):
            self.rec_optim_opt = self.opt['rec']
            self.rec_epoch = self.rec_optim_opt['epoch']
            self.rec_batch_size = self.rec_optim_opt['batch_size']

        if hasattr(self, 'conv_model'):
            self.conv_optim_opt = self.opt['conv']
            self.conv_epoch = self.conv_optim_opt['epoch']
            self.conv_batch_size = self.conv_optim_opt['batch_size']
            if self.conv_optim_opt.get('lr_scheduler', None) and 'Transformers' in self.conv_optim_opt['lr_scheduler'][
                'name']:
                batch_num = 0
                for _ in self.train_dataloader['conv'].get_conv_data(batch_size=self.conv_batch_size, shuffle=False):
                    batch_num += 1
                conv_training_steps = self.conv_epoch * floor(batch_num / self.conv_optim_opt.get('update_freq', 1))
                self.conv_optim_opt['lr_scheduler']['training_steps'] = conv_training_steps

        if hasattr(self, 'policy_model'):
            self.policy_optim_opt = self.opt['policy']
            self.policy_epoch = self.policy_optim_opt['epoch']
            self.policy_batch_size = self.policy_optim_opt['batch_size']

        self.language = dataset_language_map[self.opt['dataset']]

    def rec_evaluate(self, rec_predict, item_label):
        rec_predict = rec_predict.cpu()
        rec_predict = rec_predict[:, self.item_ids]
        _, rec_ranks = torch.topk(rec_predict, 50, dim=-1)
        rec_ranks = rec_ranks.tolist()
        item_label = item_label.tolist()
        for rec_rank, item in zip(rec_ranks, item_label):
            item = self.item_ids.index(item)
            self.evaluator.rec_evaluate(rec_rank, item)

    def policy_evaluate(self, rec_predict, movie_label):
        rec_predict = rec_predict.cpu()
        _, rec_ranks = torch.topk(rec_predict, 50, dim=-1)
        rec_ranks = rec_ranks.tolist()
        movie_label = movie_label.tolist()
        for rec_rank, movie in zip(rec_ranks, movie_label):
            self.evaluator.rec_evaluate(rec_rank, movie)

    def conv_evaluate(self, prediction, response):
        """
        Args:
            prediction: torch.LongTensor, shape=(bs, response_truncate-1)
            response: torch.LongTensor, shape=(bs, response_truncate)

            the first token in response is <|endoftext|>,  it is not in prediction
        """
        prediction = prediction.tolist()
        response = response.tolist()
        for p, r in zip(prediction, response):
            p_str = ind2txt(p, self.ind2tok, self.end_token_idx)
            r_str = ind2txt(r[1:], self.ind2tok, self.end_token_idx)
            self.evaluator.gen_evaluate(p_str, [r_str])

    def step(self, batch, stage, mode):
        """
        stage: ['policy', 'rec', 'conv']
        mode: ['train', 'val', 'test]
        """
        batch = [ele.to(self.device) for ele in batch]
        if stage == 'policy':
            if mode == 'train':
                self.policy_model.train()
            else:
                self.policy_model.eval()

            policy_loss, policy_predict = self.policy_model.forward(batch, mode)
            if mode == "train" and policy_loss is not None:
                policy_loss = policy_loss.sum()
                self.backward(policy_loss)
            else:
                self.policy_evaluate(policy_predict, batch[-1])
            if isinstance(policy_loss, torch.Tensor):
                policy_loss = policy_loss.item()
                self.evaluator.optim_metrics.add("policy_loss",
                                                 AverageMetric(policy_loss))
        elif stage == 'rec':
            if mode == 'train':
                self.rec_model.train()
            else:
                self.rec_model.eval()
            rec_loss, rec_predict = self.rec_model.forward(batch, mode)
            rec_loss = rec_loss.sum()
            if mode == "train":
                self.backward(rec_loss)
            else:
                self.rec_evaluate(rec_predict, batch[-1])
            rec_loss = rec_loss.item()
            self.evaluator.optim_metrics.add("rec_loss",
                                             AverageMetric(rec_loss))
        elif stage == "conv":
            if mode != "test":
                # train + valid: need to compute ppl
                gen_loss, pred = self.conv_model.forward(batch, mode)
                gen_loss = gen_loss.sum()
                if mode == 'train':
                    self.backward(gen_loss)
                else:
                    self.conv_evaluate(pred, batch[-1])
                gen_loss = gen_loss.item()
                self.evaluator.optim_metrics.add("gen_loss",
                                                 AverageMetric(gen_loss))
                self.evaluator.gen_metrics.add("ppl", PPLMetric(gen_loss))
            else:
                # generate response in conv_model.step
                pred = self.conv_model.forward(batch, mode)
                self.conv_evaluate(pred, batch[-1])
        else:
            raise

    def train_recommender(self):
        if hasattr(self.rec_model, 'bert'):
            if os.environ["CUDA_VISIBLE_DEVICES"] == '-1':
                bert_param = list(self.rec_model.bert.named_parameters())
            else:
                bert_param = list(self.rec_model.module.bert.named_parameters())
            bert_param_name = ['bert.' + n for n, p in bert_param]
        else:
            bert_param = []
            bert_param_name = []
        other_param = [
            name_param for name_param in self.rec_model.named_parameters()
            if name_param[0] not in bert_param_name
        ]
        params = [{'params': [p for n, p in bert_param], 'lr': self.rec_optim_opt['lr_bert']},
                  {'params': [p for n, p in other_param]}]
        self.init_optim(self.rec_optim_opt, params)

        for epoch in range(self.rec_epoch):
            self.evaluator.reset_metrics()
            logger.info(f'[Recommendation epoch {str(epoch)}]')
            for batch in self.train_dataloader['rec'].get_rec_data(self.rec_batch_size,
                                                                   shuffle=True):
                self.step(batch, stage='rec', mode='train')
            self.evaluator.report(epoch=epoch, mode='train')
            # val
            with torch.no_grad():
                self.evaluator.reset_metrics()
                for batch in self.valid_dataloader['rec'].get_rec_data(
                        self.rec_batch_size, shuffle=False):
                    self.step(batch, stage='rec', mode='val')
                self.evaluator.report(epoch=epoch, mode='val')
                # early stop
                metric = self.evaluator.rec_metrics['hit@1'] + self.evaluator.rec_metrics['hit@50']
                if self.early_stop(metric):
                    break
        # test
        with torch.no_grad():
            self.evaluator.reset_metrics()
            for batch in self.test_dataloader['rec'].get_rec_data(self.rec_batch_size,
                                                                  shuffle=False):
                self.step(batch, stage='rec', mode='test')
            self.evaluator.report(mode='test')

    def train_conversation(self):
        self.init_optim(self.conv_optim_opt, self.conv_model.parameters())

        for epoch in range(self.conv_epoch):
            self.evaluator.reset_metrics()
            logger.info(f'[Conversation epoch {str(epoch)}]')
            for batch in self.train_dataloader['conv'].get_conv_data(
                    batch_size=self.conv_batch_size, shuffle=True):
                self.step(batch, stage='conv', mode='train')
            self.evaluator.report(epoch=epoch, mode='train')
            # val
            with torch.no_grad():
                self.evaluator.reset_metrics()
                for batch in self.valid_dataloader['conv'].get_conv_data(
                        batch_size=self.conv_batch_size, shuffle=False):
                    self.step(batch, stage='conv', mode='val')
                self.evaluator.report(epoch=epoch, mode='val')
                # early stop
                metric = self.evaluator.gen_metrics['ppl']
                if self.early_stop(metric):
                    break
        # test
        with torch.no_grad():
            self.evaluator.reset_metrics()
            for batch in self.test_dataloader['conv'].get_conv_data(
                    batch_size=self.conv_batch_size, shuffle=False):
                self.step(batch, stage='conv', mode='test')
            self.evaluator.report(mode='test')

    def train_policy(self):
        policy_params = list(self.policy_model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        params = [{
            'params': [
                p for n, p in policy_params
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay':
                self.policy_optim_opt['weight_decay']
        }, {
            'params': [
                p for n, p in policy_params
                if any(nd in n for nd in no_decay)
            ],
        }]
        self.init_optim(self.policy_optim_opt, params)

        for epoch in range(self.policy_epoch):
            self.evaluator.reset_metrics()
            logger.info(f'[Policy epoch {str(epoch)}]')
            # change the shuffle to True
            for batch in self.train_dataloader['policy'].get_policy_data(
                    self.policy_batch_size, shuffle=True):
                self.step(batch, stage='policy', mode='train')
            self.evaluator.report(epoch=epoch, mode='train')
            # val
            with torch.no_grad():
                self.evaluator.reset_metrics()
                for batch in self.valid_dataloader['policy'].get_policy_data(
                        self.policy_batch_size, shuffle=False):
                    self.step(batch, stage='policy', mode='val')
                self.evaluator.report(epoch=epoch, mode='val')
                # early stop
                metric = self.evaluator.rec_metrics['hit@1'] + self.evaluator.rec_metrics['hit@50']
                if self.early_stop(metric):
                    break
        # test
        with torch.no_grad():
            self.evaluator.reset_metrics()
            for batch in self.test_dataloader['policy'].get_policy_data(
                    self.policy_batch_size, shuffle=False):
                self.step(batch, stage='policy', mode='test')
            self.evaluator.report(mode='test')

    def fit(self):
        if hasattr(self, 'rec_model'):
            self.train_recommender()
        if hasattr(self, 'policy_model'):
            self.train_policy()
        if hasattr(self, 'conv_model'):
            self.train_conversation()

    def interact(self):
        self.init_interact()
        input_text = self.get_input(self.language)
        while not self.finished:
            # rec
            if hasattr(self, 'rec_model'):
                rec_input = self.process_input(input_text, 'rec')
                scores = self.rec_model.forward(rec_input, 'infer')

                scores = scores.cpu()[0]
                scores = scores[self.item_ids]
                _, rank = torch.topk(scores, 10, dim=-1)
                item_ids = []
                for r in rank.tolist():
                    item_ids.append(self.item_ids[r])
                first_item_id = item_ids[:1]
                self.update_context('rec', entity_ids=first_item_id, item_ids=first_item_id)

                print(f"[Recommend]:")
                for item_id in item_ids:
                    if item_id in self.id2entity:
                        print(self.id2entity[item_id])
            # conv
            if hasattr(self, 'conv_model'):
                conv_input = self.process_input(input_text, 'conv')
                preds = self.conv_model.forward(conv_input, 'infer').tolist()[0]
                p_str = ind2txt(preds, self.ind2tok, self.end_token_idx)

                token_ids, entity_ids, movie_ids, word_ids = self.convert_to_id(p_str, 'conv')
                self.update_context('conv', token_ids, entity_ids, movie_ids, word_ids)

                print(f"[Response]:\n{p_str}")
            # input
            input_text = self.get_input(self.language)

    def process_input(self, input_text, stage):
        token_ids, entity_ids, movie_ids, word_ids = self.convert_to_id(input_text, stage)
        self.update_context(stage, token_ids, entity_ids, movie_ids, word_ids)

        data = {'role': 'Seeker', 'context_tokens': self.context[stage]['context_tokens'],
                'context_entities': self.context[stage]['context_entities'],
                'context_words': self.context[stage]['context_words'],
                'context_items': self.context[stage]['context_items'],
                'user_profile': self.context[stage]['user_profile'],
                'interaction_history': self.context[stage]['interaction_history']}
        dataloader = get_dataloader(self.opt, data, self.vocab[stage])
        if stage == 'rec':
            data = dataloader.rec_interact(data)
        elif stage == 'conv':
            data = dataloader.conv_interact(data)

        data = [ele.to(self.device) if isinstance(ele, torch.Tensor) else ele for ele in data]
        return data

    def convert_to_id(self, text, stage):
        if self.language == 'zh':
            tokens = self.tokenize(text, 'pkuseg')
        elif self.language == 'en':
            tokens = self.tokenize(text, 'nltk')
        else:
            raise

        entities = self.link(tokens, self.side_data[stage]['entity_kg']['entity'])
        words = self.link(tokens, self.side_data[stage]['word_kg']['entity'])

        if self.opt['tokenize'][stage] in ('gpt2', 'bert'):
            language = dataset_language_map[self.opt['dataset']]
            path = os.path.join(PRETRAIN_PATH, self.opt['tokenize'][stage], language)
            tokens = self.tokenize(text, 'bert', path)

        token_ids = [self.vocab[stage]['tok2ind'].get(token, self.vocab[stage]['unk']) for token in tokens]
        entity_ids = [self.vocab[stage]['entity2id'][entity] for entity in entities if
                      entity in self.vocab[stage]['entity2id']]
        movie_ids = [entity_id for entity_id in entity_ids if entity_id in self.item_ids]
        word_ids = [self.vocab[stage]['word2id'][word] for word in words if word in self.vocab[stage]['word2id']]

        return token_ids, entity_ids, movie_ids, word_ids

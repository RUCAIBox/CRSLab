# -*- encoding: utf-8 -*-
# @Time    :   2020/12/4
# @Author  :   Xiaolei Wang
# @email   :   wxl1999@foxmail.com

# UPDATE
# @Time    :   2021/1/3
# @Author  :   Xiaolei Wang
# @email   :   wxl1999@foxmail.com

import torch
from loguru import logger

from crslab.evaluator.metrics.base import AverageMetric
from crslab.evaluator.metrics.gen import PPLMetric
from crslab.system.base import BaseSystem
from crslab.system.utils.functions import ind2txt
from crslab.utils import ModelType


class KBRDSystem(BaseSystem):
    """This is the system for KBRD model"""
    def __init__(self, opt, agent, restore=False, save=False, interaction=False, tensorboard=False):
        """

        Args:
            opt (dict): Indicating the hyper parameters.
            agent (SupervisedAgent or Interactive Agent): Indicating the system agent.
            restore (bool, optional): Indicating if we store system after training. Defaults to False.
            interaction (bool, optional): Indicating if we interact with system. Defaults to False.
            tensorboard (bool, optional) Indicating if we monitor the training performance in tensorboard. Defaults to False. 

        """
        super(KBRDSystem, self).__init__(opt, agent, restore, save, interaction, tensorboard)

        self.ind2tok = self.agent.other_data['vocab']['ind2tok']
        self.end_token_idx = self.agent.other_data['vocab']['end']
        self.item_ids = self.agent.other_data['item_entity_ids']

        self.rec_optim_opt = opt['rec']
        self.conv_optim_opt = opt['conv']
        self.rec_epoch = self.rec_optim_opt['epoch']
        self.conv_epoch = self.conv_optim_opt['epoch']
        self.rec_batch_size = self.rec_optim_opt['batch_size']
        self.conv_batch_size = self.conv_optim_opt['batch_size']

    def _set_model_type(self) -> ModelType:
        return ModelType.GENERATION

    def rec_evaluate(self, rec_predict, item_label):
        rec_predict = rec_predict.cpu()
        rec_predict = rec_predict[:, self.item_ids]
        _, rec_ranks = torch.topk(rec_predict, 50, dim=-1)
        rec_ranks = rec_ranks.tolist()
        item_label = item_label.tolist()
        for rec_rank, label in zip(rec_ranks, item_label):
            label = self.item_ids.index(label)
            self.evaluator.rec_evaluate(rec_rank, label)

    def conv_evaluate(self, prediction, response):
        prediction = prediction.tolist()
        response = response.tolist()
        for p, r in zip(prediction, response):
            p_str = ind2txt(p, self.ind2tok, self.end_token_idx)
            r_str = ind2txt(r, self.ind2tok, self.end_token_idx)
            self.evaluator.gen_evaluate(p_str, [r_str])

    def step(self, batch, stage, mode):
        assert stage in ('rec', 'conv')
        assert mode in ('train', 'valid', 'test')

        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device)

        if stage == 'rec':
            rec_loss, rec_scores = self.model.forward(batch, mode, stage)
            rec_loss = rec_loss.sum()
            if mode == 'train':
                self.backward(rec_loss)
            else:
                self.rec_evaluate(rec_scores, batch['item'])
            rec_loss = rec_loss.item()
            self.evaluator.optim_metrics.add("rec_loss", AverageMetric(rec_loss))
        else:
            if mode != 'test':
                gen_loss, preds = self.model.forward(batch, mode, stage)
                if mode == 'train':
                    self.backward(gen_loss)
                else:
                    self.conv_evaluate(preds, batch['response'])
                gen_loss = gen_loss.item()
                self.evaluator.optim_metrics.add('gen_loss', AverageMetric(gen_loss))
                self.evaluator.gen_metrics.add("ppl", PPLMetric(gen_loss))
            else:
                preds = self.model.forward(batch, mode, stage)
                self.conv_evaluate(preds, batch['response'])

    def train_recommender(self):
        self.init_optim(self.rec_optim_opt, self.model.parameters())

        for epoch in range(self.rec_epoch):
            self.evaluator.reset_metrics()
            logger.info(f'[Recommendation epoch {str(epoch)}]')
            logger.info('[Train]')
            for batch in self.agent.get_rec_data('train', self.rec_batch_size):
                self.step(batch, stage='rec', mode='train')
            self.evaluator.report(epoch=epoch, mode='train')
            # val
            logger.info('[Valid]')
            with torch.no_grad():
                self.evaluator.reset_metrics()
                for batch in self.agent.get_rec_data('valid', self.rec_batch_size, shuffle=False):
                    self.step(batch, stage='rec', mode='valid')
                self.evaluator.report(epoch=epoch, mode='valid')
                # early stop
                metric = self.evaluator.optim_metrics['rec_loss']
                if self.early_stop(metric):
                    break
        # test
        logger.info('[Test]')
        with torch.no_grad():
            self.evaluator.reset_metrics()
            for batch in self.agent.get_rec_data('test', self.rec_batch_size, shuffle=False):
                self.step(batch, stage='rec', mode='test')
            self.evaluator.report(mode='test')

    def train_conversation(self):
        self.init_optim(self.conv_optim_opt, self.model.parameters())

        for epoch in range(self.conv_epoch):
            self.evaluator.reset_metrics()
            logger.info(f'[Conversation epoch {str(epoch)}]')
            logger.info('[Train]')
            for batch in self.agent.get_conv_data('train', self.conv_batch_size):
                self.step(batch, stage='conv', mode='train')
            self.evaluator.report(epoch=epoch, mode='train')
            # val
            logger.info('[Valid]')
            with torch.no_grad():
                self.evaluator.reset_metrics()
                for batch in self.agent.get_conv_data('valid', self.conv_batch_size, shuffle=False):
                    self.step(batch, stage='conv', mode='valid')
                self.evaluator.report(epoch=epoch, mode='valid')
                # early stop
                metric = self.evaluator.optim_metrics['gen_loss']
                if self.early_stop(metric):
                    break
        # test
        logger.info('[Test]')
        with torch.no_grad():
            self.evaluator.reset_metrics()
            for batch in self.agent.get_conv_data('test', self.conv_batch_size, shuffle=False):
                self.step(batch, stage='conv', mode='test')
            self.evaluator.report(mode='test')

    def fit(self):
        self.train_recommender()
        self.train_conversation()

    def interact(self):
        raise NotImplementedError('Interact function in KBRD System not implement yet.')

# @Time   : 2020/12/9
# @Author : Yuanhang Zhou
# @Email  : sdzyh002@gmail.com

# UPDATE:
# @Time   : 2020/12/17
# @Author : Xiaolei Wang
# @Email  : wxl1999@foxmail.com
from math import floor

import torch
from loguru import logger

from crslab.evaluator.metrics.base_metrics import AverageMetric
from crslab.evaluator.metrics.gen_metrics import PPLMetric
from crslab.system.base_system import BaseSystem
from crslab.system.utils import ind2txt


class TGReDialSystem(BaseSystem):
    def __init__(self,
                 opt,
                 train_dataloader,
                 valid_dataloader,
                 test_dataloader,
                 vocab,
                 side_data,
                 restore=False,
                 debug=False):
        super(TGReDialSystem, self).__init__(opt, train_dataloader, valid_dataloader,
                                             test_dataloader, vocab, side_data, restore,
                                             debug)

        self.dataset = self.opt['dataset']

        self.ind2tok = vocab['conv']['ind2tok']
        self.end_token_idx = vocab['conv']['end']
        self.movie_ids = side_data['rec']['item_entity_ids']

        self.rec_optim_opt = self.opt['rec']
        self.rec_epoch = self.rec_optim_opt['epoch']
        self.rec_batch_size = self.rec_optim_opt['batch_size']

        self.conv_optim_opt = self.opt['conv']
        self.conv_epoch = self.conv_optim_opt['epoch']
        self.conv_batch_size = self.conv_optim_opt['batch_size']

        self.policy_optim_opt = self.opt.get('policy', None)
        if self.policy_optim_opt:
            self.policy_epoch = self.policy_optim_opt['epoch']
            self.policy_batch_size = self.policy_optim_opt['batch_size']

        if self.conv_optim_opt.get('lr_scheduler', None) and 'Transformers' in self.conv_optim_opt['lr_scheduler'][
            'name']:
            batch_num = 0
            for _ in self.train_dataloader['conv'].get_conv_data(batch_size=self.conv_batch_size, shuffle=False):
                batch_num += 1
            conv_training_steps = self.conv_epoch * floor(batch_num / self.conv_optim_opt.get('update_freq', 1))
            self.conv_optim_opt['lr_scheduler']['training_steps'] = conv_training_steps

    def rec_evaluate(self, rec_predict, item_label):
        rec_predict = rec_predict.cpu().detach()
        if self.dataset == 'ReDial':
            rec_predict = rec_predict[:, self.movie_ids]
        _, rec_ranks = torch.topk(rec_predict, 50, dim=-1)
        item_label = item_label.cpu().detach()
        for rec_rank, item in zip(rec_ranks, item_label):
            if self.dataset == 'ReDial':
                item = self.movie_ids.index(item.item())
            self.evaluator.rec_evaluate(rec_rank, item)

    def conv_evaluate(self, prediction, response):
        """
        Args:
            prediction: torch.LongTensor, shape=(bs, response_truncate-1)
            response: torch.LongTensor, shape=(bs, response_truncate)

            the first token in response is <|endoftext|>,  it is not in prediction
        """
        prediction = [[ind.item() for ind in inds] for inds in prediction.cpu().detach()]
        response = [[ind.item() for ind in inds] for inds in response.cpu().detach()]
        for p, r in zip(prediction, response):
            p_str = ind2txt(p, self.ind2tok, self.end_token_idx)
            r_str = ind2txt(r[1:], self.ind2tok, self.end_token_idx)
            self.evaluator.gen_evaluate(p_str, [r_str])

    def policy_evaluate(self, rec_predict, movie_label):
        rec_predict = rec_predict.cpu().detach()
        _, rec_ranks = torch.topk(rec_predict, 50, dim=-1)
        movie_label = movie_label.cpu().detach()
        for rec_rank, movie in zip(rec_ranks, movie_label):
            self.evaluator.rec_evaluate(rec_rank, movie)

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

            policy_loss, policy_predict = self.policy_model(batch, mode)
            loss = policy_loss
            if mode == "train":
                self.backward(loss)
            else:
                self.rec_evaluate(policy_predict, batch[-1])
            policy_loss = policy_loss.item()
            self.evaluator.optim_metrics.add("policy_loss",
                                             AverageMetric(policy_loss))
        elif stage == 'rec':
            if mode == 'train':
                self.rec_model.train()
            else:
                self.rec_model.eval()

            rec_loss, rec_predict = self.rec_model(batch, mode)
            loss = rec_loss
            if mode == "train":
                self.backward(loss)
            else:
                self.rec_evaluate(rec_predict, batch[-1])
            rec_loss = rec_loss.item()
            self.evaluator.optim_metrics.add("rec_loss",
                                             AverageMetric(rec_loss))
        elif stage == "conv":
            if mode != "test":
                # train + valid: need to compute ppl
                gen_loss, pred = self.conv_model(batch, mode)
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
                pred = self.conv_model(batch, mode)
                self.conv_evaluate(pred, batch[-1])
        else:
            raise

    def train_recommender(self):
        bert_param = list(self.rec_model.bert.named_parameters())
        bert_param_name = ['bert.' + n for n, p in bert_param]
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
            self.evaluator.report()
            # val
            with torch.no_grad():
                self.evaluator.reset_metrics()
                for batch in self.valid_dataloader['rec'].get_rec_data(
                        self.rec_batch_size, shuffle=False):
                    self.step(batch, stage='rec', mode='val')
                self.evaluator.report()
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
            self.evaluator.report()

    def train_conversation(self):
        self.init_optim(self.conv_optim_opt, self.conv_model.parameters())

        for epoch in range(self.conv_epoch):
            self.evaluator.reset_metrics()
            logger.info(f'[Conversation epoch {str(epoch)}]')
            for batch in self.train_dataloader['conv'].get_conv_data(
                    batch_size=self.conv_batch_size, shuffle=True):
                self.step(batch, stage='conv', mode='train')
            self.evaluator.report()
            # val
            with torch.no_grad():
                self.evaluator.reset_metrics()
                for batch in self.valid_dataloader['conv'].get_conv_data(
                        batch_size=self.conv_batch_size, shuffle=False):
                    self.step(batch, stage='conv', mode='val')
                self.evaluator.report()
        # test
        with torch.no_grad():
            self.evaluator.reset_metrics()
            for batch in self.test_dataloader['conv'].get_conv_data(
                    batch_size=self.conv_batch_size, shuffle=False):
                self.step(batch, stage='conv', mode='test')
            self.evaluator.report()

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

        for epoch in range(self.rec_epoch):
            self.evaluator.reset_metrics()
            logger.info(f'[Policy epoch {str(epoch)}]')
            # change the shuffle to True
            for batch in self.train_dataloader['rec'].get_policy_data(
                    self.rec_batch_size, shuffle=True):
                self.step(batch, stage='policy', mode='train')
            self.evaluator.report()
            # val
            with torch.no_grad():
                self.evaluator.reset_metrics()
                for batch in self.valid_dataloader['rec'].get_policy_data(
                        self.rec_batch_size, shuffle=False):
                    self.step(batch, stage='policy', mode='val')
                self.evaluator.report()
                # early stop
                metric = self.evaluator.rec_metrics['hit@1'] + self.evaluator.rec_metrics['hit@50']
                if self.early_stop(metric):
                    break
        # test
        with torch.no_grad():
            self.evaluator.reset_metrics()
            for batch in self.test_dataloader['rec'].get_policy_data(
                    self.rec_batch_size, shuffle=False):
                self.step(batch, stage='policy', mode='test')
            self.evaluator.report()

    def fit(self):
        if hasattr(self, 'rec_model'):
            self.train_recommender()
        if hasattr(self, 'policy_model'):
            self.train_policy()
        if hasattr(self, 'conv_model'):
            self.train_conversation()

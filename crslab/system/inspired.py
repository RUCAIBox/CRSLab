# @Time   : 2021/3/1
# @Author : Beichen Zhang
# @Email  : zhangbeichen724@gmail.com

import torch
from loguru import logger
from math import floor

from crslab.data import dataset_language_map
from crslab.evaluator.metrics.base import AverageMetric
from crslab.evaluator.metrics.gen import PPLMetric
from crslab.system.base import BaseSystem
from crslab.system.utils.functions import ind2txt


class InspiredSystem(BaseSystem):
    """This is the system for Inspired model"""

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
        super(InspiredSystem, self).__init__(opt, train_dataloader, valid_dataloader,
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

    def conv_evaluate(self, prediction, response):
        """
        Args:
            prediction: torch.LongTensor, shape=(bs, response_truncate-1)
            response: (torch.LongTensor, torch.LongTensor), shape=((bs, response_truncate), (bs, response_truncate))

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
        if stage == 'rec':
            if mode == 'train':
                self.rec_model.train()
            else:
                self.rec_model.eval()

            rec_loss, rec_predict = self.rec_model.recommend(batch, mode)
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
                gen_loss, pred = self.conv_model.converse(batch, mode)
                if mode == 'train':
                    self.conv_model.train()
                    self.backward(gen_loss)
                else:
                    self.conv_model.eval()
                    self.conv_evaluate(pred, batch[-1])
                gen_loss = gen_loss.item()
                self.evaluator.optim_metrics.add("gen_loss",
                                                 AverageMetric(gen_loss))
                self.evaluator.gen_metrics.add("ppl", PPLMetric(gen_loss))
            else:
                # generate response in conv_model.step
                pred = self.conv_model.converse(batch, mode)
                self.conv_evaluate(pred, batch[-1])
        else:
            raise

    def train_recommender(self):
        if hasattr(self.rec_model, 'bert'):
            bert_param = list(self.rec_model.bert.named_parameters())
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
                    self.step((batch), stage='conv', mode='val')
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
                self.step((batch), stage='conv', mode='test')
            self.evaluator.report(mode='test')

    def fit(self):
        if hasattr(self, 'rec_model'):
            self.train_recommender()
        if hasattr(self, 'conv_model'):
            self.train_conversation()

    def interact(self):
        pass

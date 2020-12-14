# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/24, 2020/12/13
# @Author : Kun Zhou, Xiaolei Wang
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com

import torch
from loguru import logger

from crslab.evaluator.metrics.base_metrics import AverageMetric
from crslab.evaluator.metrics.gen_metrics import PPLMetric
from crslab.system.base_system import BaseSystem
from crslab.system.utils import ind2txt


class KGSFSystem(BaseSystem):
    def __init__(self, opt, train_dataloader, valid_dataloader, test_dataloader, vocab, side_data, restore=False,
                 save=False, debug=False):
        super(KGSFSystem, self).__init__(opt, train_dataloader, valid_dataloader, test_dataloader, vocab, side_data,
                                         restore, save, debug)

        self.ind2tok = vocab['ind2tok']
        self.end_token_idx = vocab['end']
        self.movie_ids = side_data['item_entity_ids']

        self.pretrain_optim_opt = self.opt['pretrain']
        self.rec_optim_opt = self.opt['rec']
        self.conv_optim_opt = self.opt['conv']
        self.pretrain_epoch = self.pretrain_optim_opt['epoch']
        self.rec_epoch = self.rec_optim_opt['epoch']
        self.conv_epoch = self.conv_optim_opt['epoch']
        self.pretrain_batch_size = self.pretrain_optim_opt['batch_size']
        self.rec_batch_size = self.rec_optim_opt['batch_size']
        self.conv_batch_size = self.conv_optim_opt['batch_size']

    def rec_evaluate(self, rec_predict, movie_label):
        rec_predict = rec_predict.cpu().detach()
        rec_predict = rec_predict[:, self.movie_ids]
        _, rec_ranks = torch.topk(rec_predict, 50, dim=-1)
        movie_label = movie_label.cpu().detach()
        for rec_rank, movie in zip(rec_ranks, movie_label):
            movie = self.movie_ids.index(movie.item())
            self.evaluator.rec_evaluate(rec_rank, movie)

    def conv_evaluate(self, prediction, response):
        prediction = prediction.cpu().detach()
        response = response.cpu().detach()
        for p, r in zip(prediction, response):
            p_str = ind2txt(p, self.ind2tok, self.end_token_idx)
            r_str = ind2txt(r, self.ind2tok, self.end_token_idx)
            self.evaluator.gen_evaluate(p_str, [r_str])

    def step(self, batch, stage, mode):
        batch = [ele.to(self.device) for ele in batch]
        if stage == 'pretrain':
            info_loss = self.model.pretrain_infomax(batch)
            if info_loss:
                self.backward(info_loss)
                info_loss = info_loss.item()
                self.evaluator.optim_metrics.add("info_loss", AverageMetric(info_loss))
        elif stage == 'rec':
            rec_loss, info_loss, rec_predict = self.model.recommend(batch, mode)
            if info_loss:
                loss = rec_loss + 0.025 * info_loss
            else:
                loss = rec_loss
            if mode == "train":
                self.backward(loss)
            else:
                self.rec_evaluate(rec_predict, batch[-1])
            rec_loss = rec_loss.item()
            self.evaluator.optim_metrics.add("rec_loss", AverageMetric(rec_loss))
            if info_loss:
                info_loss = info_loss.item()
                self.evaluator.optim_metrics.add("info_loss", AverageMetric(info_loss))
        elif stage == "conv":
            if mode != "test":
                gen_loss, pred = self.model.converse(batch, mode)
                if mode == 'train':
                    self.backward(gen_loss)
                else:
                    self.conv_evaluate(pred, batch[-1])
                gen_loss = gen_loss.item()
                self.evaluator.optim_metrics.add("gen_loss", AverageMetric(gen_loss))
                self.evaluator.gen_metrics.add("ppl", PPLMetric(gen_loss))
            else:
                pred = self.model.converse(batch, mode)
                self.conv_evaluate(pred, batch[-1])
        else:
            raise

    def pretrain(self):
        self.init_optim(self.pretrain_optim_opt, self.model.parameters())

        for epoch in range(self.pretrain_epoch):
            self.evaluator.reset_metrics()
            logger.info(f'[Pretrain epoch {str(epoch)}]')
            for batch in self.train_dataloader.get_pretrain_data(self.pretrain_batch_size, shuffle=False):
                self.step(batch, stage="pretrain", mode='train')
            self.evaluator.report()

    def train_recommender(self):
        self.init_optim(self.rec_optim_opt, self.model.parameters())

        for epoch in range(self.rec_epoch):
            self.evaluator.reset_metrics()
            logger.info(f'[Recommendation epoch {str(epoch)}]')
            logger.info('[Train]')
            for batch in self.train_dataloader.get_rec_data(self.rec_batch_size, shuffle=False):
                self.step(batch, stage='rec', mode='train')
            self.evaluator.report()
            # val
            logger.info('[Valid]')
            with torch.no_grad():
                self.evaluator.reset_metrics()
                for batch in self.valid_dataloader.get_rec_data(self.rec_batch_size, shuffle=False):
                    self.step(batch, stage='rec', mode='val')
                self.evaluator.report()
                # early stop
                metric = self.evaluator.rec_metrics['recall@1'] + self.evaluator.rec_metrics['recall@50']
                if self.early_stop(metric):
                    break
        # test
        logger.info('[Test]')
        with torch.no_grad():
            self.evaluator.reset_metrics()
            for batch in self.test_dataloader.get_rec_data(self.rec_batch_size, shuffle=False):
                self.step(batch, stage='rec', mode='test')
            self.evaluator.report()

    def train_conversation(self):
        self.model.freeze_parameters()
        self.init_optim(self.conv_optim_opt, self.model.parameters())

        for epoch in range(self.conv_epoch):
            self.evaluator.reset_metrics()
            logger.info(f'[Conversation epoch {str(epoch)}]')
            logger.info('[Train]')
            for batch in self.train_dataloader.get_conv_data(batch_size=self.conv_batch_size, shuffle=False):
                self.step(batch, stage='conv', mode='train')
            self.evaluator.report()
            # val
            logger.info('[Valid]')
            with torch.no_grad():
                self.evaluator.reset_metrics()
                for batch in self.valid_dataloader.get_conv_data(batch_size=self.conv_batch_size, shuffle=False):
                    self.step(batch, stage='conv', mode='val')
                self.evaluator.report()
        # test
        logger.info('[Test]')
        with torch.no_grad():
            self.evaluator.reset_metrics()
            for batch in self.test_dataloader.get_conv_data(batch_size=self.conv_batch_size, shuffle=False):
                self.step(batch, stage='conv', mode='test')
            self.evaluator.report()

    def fit(self):
        self.pretrain()
        self.train_recommender()
        self.train_conversation()
        if self.save:
            self.save_model()

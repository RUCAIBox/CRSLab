# @Time   : 2020/12/9
# @Author : Yuanhang Zhou
# @Email  : sdzyh002@gmail.com

# UPDATE:
# @Time   : 2020/12/13
# @Author : Xiaolei Wang
# @Email  : wxl1999@foxmail.com

import torch
import transformers
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
                 save=False,
                 debug=False):
        super(TGReDialSystem, self).__init__(opt, train_dataloader, valid_dataloader,
                                             test_dataloader, vocab, side_data, restore, save,
                                             debug)

        self.dataset = self.opt['dataset']

        self.ind2tok = vocab['conv']['ind2tok']
        self.end_token_idx = vocab['conv']['end']
        self.movie_ids = side_data['rec']['item_entity_ids']

        self.rec_optim_opt = self.opt['rec']
        self.conv_optim_opt = self.opt['conv']
        self.policy_optim_opt = self.opt['policy']
        self.rec_epoch = self.rec_optim_opt['epoch']
        self.conv_epoch = self.conv_optim_opt['epoch']
        self.policy_epoch = self.policy_optim_opt['epoch']
        self.rec_batch_size = self.rec_optim_opt['batch_size']
        self.conv_batch_size = self.conv_optim_opt['batch_size']
        self.policy_batch_size = self.policy_optim_opt['batch_size']

        # only conv model need gradient clip
        self.rec_gradient_clip = self.opt['rec'].get('gradient_clip', None)
        self.conv_gradient_clip = self.opt['conv'].get('gradient_clip', None)
        self.policy_gradient_clip = self.opt['policy'].get('gradient_clip', None)

        # self.warmup_steps = self.opt['warmup_steps']
        # self.WarmupLinearSchedule = self.opt['WarmupLinearSchedule']
        # self.total_steps = int(self.conv_optim_opt['datasset_len'] *
        #                        self.conv_epoch / self.conv_batch_size /
        #                        self.conv_optim_opt['gradient_accumulation'])

    def rec_evaluate(self, rec_predict, movie_label):
        rec_predict = rec_predict.cpu().detach()
        if self.dataset == 'ReDial':
            rec_predict = rec_predict[:, self.movie_ids]
        _, rec_ranks = torch.topk(rec_predict, 50, dim=-1)
        movie_label = movie_label.cpu().detach()
        for rec_rank, movie in zip(rec_ranks, movie_label):
            if self.dataset == 'ReDial':
                movie = self.movie_ids.index(movie.item())
            self.evaluator.rec_evaluate(rec_rank, movie)

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

    def backward(self, loss):
        """empty grad, backward loss and update params

        Args:
            loss (torch.Tensor):
        """
        self._zero_grad()

        if self.update_freq > 1:
            self._number_grad_accum = (self._number_grad_accum +
                                       1) % self.update_freq
            loss /= self.update_freq
        loss.backward()

        # shuld be True only when training conv model
        if self.gradient_clip:
            torch.nn.utils.clip_grad_norm_(self.model_parameters,
                                           self.gradient_clip)

        self._update_params()

    def _update_params(self):
        if self.update_freq > 1:
            # we're doing gradient accumulation, so we don't only want to step
            # every N updates instead
            # self._number_grad_accum is updated in backward function
            if self._number_grad_accum != 0:
                return

        self.optimizer.step()

        # keep track up number of steps, compute warmup factor
        # self._number_training_updates += 1

        # delete the args of step
        if hasattr(self, 'scheduler'):
            self.scheduler.step()

    def build_lr_scheduler(self, opt, state=None):
        """
        Add conversational model's scheduler
        """
        super(TGReDialSystem, self).build_lr_scheduler(opt, state)

        # # create scheduler for conv model
        # if self.WarmupLinearSchedule:
        #     self.scheduler = transformers.get_linear_schedule_with_warmup(
        #         self.optimizer, self.warmup_steps, self.total_steps)

    def policy_evaluate(self, rec_predict, movie_label):
        rec_predict = rec_predict.cpu().detach()
        _, rec_ranks = torch.topk(rec_predict, 50, dim=-1)
        movie_label = movie_label.cpu().detach()
        for rec_rank, movie in zip(rec_ranks, movie_label):
            self.evaluator.rec_evaluate(rec_rank, movie)

    def build_optimizer(self, opt, task):
        if task == 'rec':
            bert_param_optimizer = list(self.rec_model.bert.named_parameters())
            bert_param_name = ['bert.' + n for n, p in bert_param_optimizer]

            other_param_optimizer = [
                name_param for name_param in self.rec_model.named_parameters()
                if name_param[0] not in bert_param_name
            ]
            other_param_name = [n for n, p in other_param_optimizer]

            self.optimizer = transformers.AdamW(
                [{
                    'params': [p for n, p in bert_param_optimizer],
                    'lr': self.rec_optim_opt['lr_bert']
                }, {
                    'params': [p for n, p in other_param_optimizer]
                }],
                lr=self.rec_optim_opt['lr_sasrec'])

            logger.info("[Build optimizer: {}]", opt["optimizer"])
        elif task == 'policy':
            param_optimizer = list(self.policy_model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [{
                'params': [
                    p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)
                ],
                'weight_decay':
                    0.01
            }, {
                'params': [
                    p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)
                ],
                'weight_decay':
                    0.0
            }]

            self.optimizer = transformers.AdamW(optimizer_grouped_parameters,
                                                lr=self.policy_optim_opt['lr'])
            logger.info("[Build optimizer: {}]", opt["optimizer"])
        elif task == 'conv':
            self.optimizer = transformers.AdamW(self.conv_model.parameters(),
                                                lr=self.conv_optim_opt['lr'],
                                                correct_bias=True)

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
        # check the next two line
        self.build_optimizer(self.rec_optim_opt, 'rec')
        self.build_lr_scheduler(self.rec_optim_opt)  # do nothing
        self.reset_early_stop_state()
        self.gradient_clip = self.rec_gradient_clip

        for epoch in range(self.rec_epoch):
            self.evaluator.reset_metrics()
            logger.info(f'[Recommendation epoch {str(epoch)}]')
            # change the shuffle to True
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
                # fancy metric
                metric = self.evaluator.rec_metrics[
                             'recall@1'] + self.evaluator.rec_metrics['recall@50']
                self.early_stop(metric)
                if self.stop:
                    break
        # test
        with torch.no_grad():
            self.evaluator.reset_metrics()
            for batch in self.test_dataloader['rec'].get_rec_data(self.rec_batch_size,
                                                                  shuffle=False):
                self.step(batch, stage='rec', mode='test')
            self.evaluator.report()

    def train_conversation(self):
        # what does the next line do: freeze the param of other model
        self.build_optimizer(self.conv_optim_opt, 'conv')
        self.build_lr_scheduler(self.conv_optim_opt)
        self.gradient_clip = self.conv_gradient_clip
        self.model_parameters = self.conv_model.parameters()

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
        self.build_optimizer(self.rec_optim_opt, 'policy')
        self.build_lr_scheduler(self.rec_optim_opt)  # do nothing
        self.reset_early_stop_state()
        self.gradient_clip = self.policy_gradient_clip

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
                # fancy metric
                metric = self.evaluator.rec_metrics[
                             'recall@1'] + self.evaluator.rec_metrics['recall@50']
                self.early_stop(metric)
                if self.stop:
                    break
        # test
        with torch.no_grad():
            self.evaluator.reset_metrics()
            for batch in self.test_dataloader['rec'].get_policy_data(
                    self.rec_batch_size, shuffle=False):
                self.step(batch, stage='policy', mode='test')
            self.evaluator.report()

    def fit(self):
        self.train_recommender()
        # self.train_policy()
        self.train_conversation()

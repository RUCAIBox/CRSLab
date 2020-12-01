# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/24, 2020/11/30
# @Author : Kun Zhou, Xiaolei Wang
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com

import os
from abc import ABC, abstractmethod

import torch
from loguru import logger
from torch import optim

from crslab.config.config import SAVE_PATH
from crslab.evaluator import StandardEvaluator
from crslab.model import get_model
from crslab.system.lr_scheduler import LRScheduler


class BaseSystem(ABC):
    r"""Trainer Class is used to manage the training and evaluation processes of recommender system models.
    AbstractTrainer is an abstract class in which the fit() and evaluate() method should be implemented according
    to different training and evaluation strategies.

    For side data, it is judged by model
    """

    def __init__(self, opt, train_dataloader, valid_dataloader, test_dataloader, ind2tok, side_data=None):
        self.opt = opt
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # data
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self.ind2tok = ind2tok
        self.end_token_idx = opt['end_token_idx']
        # model
        if 'model' in opt:
            self.model = get_model(opt, opt['model'], self.device, side_data).to(self.device)
        if 'rec_model' in opt:
            self.rec_model = get_model(opt, opt['rec_model'], self.device, side_data).to(self.device)
        if 'conv_model' in opt:
            self.conv_model = get_model(opt, opt['conv_model'], self.device, side_data).to(self.device)
        if 'policy_model' in opt:
            self.policy_model = get_model(opt, opt['policy_model'], self.device, side_data).to(self.device)
        self.model_file_save_path = os.path.join(SAVE_PATH, f'{opt["model_name"]}.pth')
        self.evaluator = StandardEvaluator()
        # optim
        # gradient acumulation
        self.update_freq = opt.get('update_freq', 1)
        self._number_grad_accum = 0
        # LR scheduler
        self._number_training_updates = 0
        # early stop
        self.best_valid = None
        self.impatience = opt.get('impatience', 3)
        self.drop_cnt = 0
        self.val_optim = 1 if opt["val_mode"] == "max" else -1
        self.stop = False

    @abstractmethod
    def fit(self):
        r"""Train the model based on the train data.

        """
        raise NotImplementedError('Method [next] should be implemented.')

    def build_optimizer(self, opt, parameters):
        optimizer = opt['optimizer']
        # set up optimizer args
        lr = opt["learning_rate"]
        kwargs = {'lr': lr}
        if opt.get('weight_decay', 0):
            kwargs['weight_decay'] = opt['weight_decay']
        if opt.get('momentum', 0) > 0 and opt['optimizer'] in ['sgd', 'rmsprop', 'qhm']:
            # turn on momentum for optimizers that use it
            kwargs['momentum'] = opt['momentum']
            if opt['optimizer'] == 'sgd' and opt.get('nesterov', True):
                # for sgd, maybe nesterov
                kwargs['nesterov'] = opt.get('nesterov', True)
            elif opt['optimizer'] == 'qhm':
                # qhm needs a nu
                kwargs['nu'] = opt.get('nus', (0.7,))[0]
        elif opt['optimizer'] == 'adam':
            # turn on amsgrad for `adam`
            # amsgrad paper: https://openreview.net/forum?id=ryQu7f-RZ
            kwargs['amsgrad'] = True
        elif opt['optimizer'] == 'qhadam':
            # set nus for qhadam
            kwargs['nus'] = opt.get('nus', (0.7, 1.0))
        elif opt['optimizer'] == 'adafactor':
            # adafactor params
            kwargs['beta1'] = opt.get('betas', (0.9, 0.999))[0]
            kwargs['eps'] = opt['adafactor_eps']
            kwargs['warmup_init'] = opt.get('warmup_updates', -1) > 0

        if opt['optimizer'] in [
            'adam',
            'sparseadam',
            'fused_adam',
            'adamax',
            'qhadam',
            'fused_lamb',
        ]:
            # set betas for optims that use it
            kwargs['betas'] = opt.get('betas', (0.9, 0.999))
            # set adam optimizer, but only if user specified it
            if opt.get('adam_eps'):
                kwargs['eps'] = opt['adam_eps']

        optim_class = {k.lower(): v for k, v in optim.__dict__.items() if not k.startswith('__') and k[0].isupper()}
        self.optimizer = optim_class[optimizer](parameters, **kwargs)
        logger.info("[Build optimizer: {}]", opt["optimizer"])

    def build_lr_scheduler(self, opt, states=None, hard_reset=False):
        """
        Create the learning rate scheduler, and assign it to self.scheduler. This
        scheduler will be updated upon a call to receive_metrics. May also create
        self.warmup_scheduler, if appropriate.

        :param state_dict states: Possible state_dict provided by model
            checkpoint, for restoring LR state
        :param bool hard_reset: If true, the LR scheduler should ignore the
            state dictionary.
        """
        if states is None:
            states = {}
        if opt.get('lr_scheduler', None):
            self.scheduler = LRScheduler.lr_scheduler_factory(opt, self.optimizer, states, hard_reset)
            self._number_training_updates = self.scheduler.get_initial_number_training_updates()
            logger.info(f"[Build scheduler {opt['lr_scheduler']}]")

    def reset_training_steps(self):
        self._number_training_updates = 0

    def _zero_grad(self):
        if self._number_grad_accum != 0:
            # if we're accumulating gradients, don't actually zero things out yet.
            return
        self.optimizer.zero_grad()

    def _update_params(self):
        if self.update_freq > 1:
            # we're doing gradient accumulation, so we don't only want to step
            # every N updates instead
            # self._number_grad_accum is updated in backward function
            if self._number_grad_accum != 0:
                return

        self.optimizer.step()

        # keep track up number of steps, compute warmup factor
        self._number_training_updates += 1

        if hasattr(self, 'scheduler'):
            self.scheduler.step(self._number_training_updates)

    def backward(self, loss):
        self._zero_grad()

        if self.update_freq > 1:
            self._number_grad_accum = (self._number_grad_accum + 1) % self.update_freq
            loss /= self.update_freq
        loss.backward()

        self._update_params()

    def adjust_lr(self, metric=None):
        if not hasattr(self, 'scheduler') or self.scheduler is None:
            return
        self.scheduler.valid_step(metric)

    def early_stop(self, metric):
        if self.best_valid is None or self.best_valid < metric * self.val_optim:
            self.best_valid = metric
            self.drop_cnt = 0
        else:
            self.drop_cnt += 1
            if self.drop_cnt >= self.impatience:
                self.stop = True

    def reset_early_stop_state(self):
        self.best_valid = None
        self.drop_cnt = 0
        self.stop = False

    def ind2txt(self, inds):
        sentence = []
        for ind in inds:
            if ind == self.end_token_idx:
                break
            sentence.append(self.ind2tok.get(ind, 'unk'))
        return ' '.join(sentence)

    def save_system(self):
        r"""Store the model parameters information and training information.

        Args:
            epoch (int): the current epoch id
            saved_model_file (str): file name for saved pretrained model

        """
        state = {'config': self.opt}
        if hasattr(self, 'model'):
            state['model_state_dict'] = self.model.state_dict()
        if hasattr(self, 'rec_model'):
            state['rec_state_dict'] = self.rec_model.state_dict()
        if hasattr(self, 'conv_model'):
            state['conv_state_dict'] = self.conv_model.state_dict()
        if hasattr(self, 'policy_model'):
            state['policy_state_dict'] = self.policy_model.state_dict()
        torch.save(state, self.model_file_save_path)

    def load_system(self):
        r"""Store the model parameters information and training information.

        Args:
            epoch (int): the current epoch id
            saved_model_file (str): file name for saved pretrained model

        """
        checkpoint = torch.load(self.model_file_save_path)
        self.opt = checkpoint['config']
        if hasattr(self, 'model'):
            self.model.load_state_dict(checkpoint['model_state_dict'])
        if hasattr(self, 'rec_model'):
            self.rec_model.load_state_dict(checkpoint['rec_state_dict'])
        if hasattr(self, 'conv_model'):
            self.conv_model.load_state_dict(checkpoint['conv_state_dict'])
        if hasattr(self, 'policy_model'):
            self.policy_model.load_state_dict(checkpoint['policy_state_dict'])

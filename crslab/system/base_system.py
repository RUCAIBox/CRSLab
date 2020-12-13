# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/24, 2020/12/2
# @Author : Kun Zhou, Xiaolei Wang
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com

import os
from abc import ABC, abstractmethod

import torch
from loguru import logger
from torch import optim

from crslab.config.config import SAVE_PATH
from crslab.evaluator import get_evaluator
from crslab.model import get_model
from crslab.system.lr_scheduler import LRScheduler


class BaseSystem(ABC):
    def __init__(self, opt, train_dataloader, valid_dataloader, test_dataloader, vocab, side_data=None, restore=False,
                 save=False, debug=False):
        self.opt = opt
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # data
        if debug:
            self.train_dataloader = valid_dataloader
            self.valid_dataloader = valid_dataloader
            self.test_dataloader = test_dataloader
        else:
            self.train_dataloader = train_dataloader
            self.valid_dataloader = valid_dataloader
            self.test_dataloader = test_dataloader
        # model
        if 'model' in opt:
            self.model = get_model(opt, opt['model'], self.device, vocab, side_data).to(self.device)
        if 'rec_model' in opt:
            self.rec_model = get_model(opt, opt['rec_model'], self.device, vocab['rec'], side_data['rec']).to(
                self.device)
        if 'conv_model' in opt:
            self.conv_model = get_model(opt, opt['conv_model'], self.device, vocab['conv'], side_data['conv']).to(
                self.device)
        if 'policy_model' in opt:
            self.policy_model = get_model(opt, opt['policy_model'], self.device, vocab['policy'],
                                          side_data['policy']).to(self.device)
        model_file_name = opt.get('model_file', f'{opt["model_name"]}.pth')
        self.model_file = os.path.join(SAVE_PATH, model_file_name)
        if restore:
            self.restore_model()
        self.save = save
        self.evaluator = get_evaluator(opt.get('evaluator', 'standard'))
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
        self.valid_optim = 1 if opt["val_mode"] == "max" else -1
        self.stop = False

    @abstractmethod
    def step(self, batch, stage, mode):
        """calculate loss and prediction for batch data under certrain stage and mode

        Args:
            batch (dict or tuple): batch data
            stage (str): recommendation/policy/conversation etc.
            mode (str): train/valid/test
        """
        pass

    @abstractmethod
    def fit(self):
        """fit the whole system"""
        pass

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

    def build_lr_scheduler(self, opt, state=None):
        """
        Create the learning rate scheduler, and assign it to self.scheduler. This
        scheduler will be updated upon a call to receive_metrics. May also create
        self.warmup_scheduler, if appropriate.

        :param state_dict states: Possible state_dict provided by model
            checkpoint, for restoring LR state
        :param bool hard_reset: If true, the LR scheduler should ignore the
            state dictionary.
        """
        if state is None:
            state = {}
        if opt.get('lr_scheduler', None):
            self.scheduler = LRScheduler.lr_scheduler_factory(opt, self.optimizer, state)
            self._number_training_updates = self.scheduler.get_initial_number_training_updates()
            logger.info(f"[Build scheduler {opt['lr_scheduler']}]")

    def reset_early_stop_state(self):
        self.best_valid = None
        self.drop_cnt = 0
        self.stop = False
        logger.debug('[Reset early stop state]')

    def init_optim(self, opt, parameters):
        self.build_optimizer(opt, parameters)
        self.build_lr_scheduler(opt)
        self.reset_early_stop_state()

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
        """empty grad, backward loss and update params

        Args:
            loss (torch.Tensor):
        """
        self._zero_grad()

        if self.update_freq > 1:
            self._number_grad_accum = (self._number_grad_accum + 1) % self.update_freq
            loss /= self.update_freq
        loss.backward()

        self._update_params()

    def adjust_lr(self, metric=None):
        """adjust learning rate w/o metric by scheduler

        Args:
            metric (optional): Defaults to None.
        """
        if not hasattr(self, 'scheduler') or self.scheduler is None:
            return
        self.scheduler.valid_step(metric)
        logger.debug('[Adjust learning rate after valid epoch]')

    def early_stop(self, metric):
        if self.best_valid is None or metric * self.valid_optim > self.best_valid * self.valid_optim:
            self.best_valid = metric
            self.drop_cnt = 0
            logger.info('[Get new best model]')
        else:
            self.drop_cnt += 1
            if self.drop_cnt >= self.impatience:
                self.stop = True
                logger.info('[Early stop]')

    def save_model(self):
        r"""Store the model parameters."""
        state = {}
        if hasattr(self, 'model'):
            state['model_state_dict'] = self.model.state_dict()
        if hasattr(self, 'rec_model'):
            state['rec_state_dict'] = self.rec_model.state_dict()
        if hasattr(self, 'conv_model'):
            state['conv_state_dict'] = self.conv_model.state_dict()
        if hasattr(self, 'policy_model'):
            state['policy_state_dict'] = self.policy_model.state_dict()
        torch.save(state, self.model_file)
        logger.info(f'[Save model into {self.model_file}]')

    def restore_model(self):
        r"""Store the model parameters."""
        if not os.path.exists(self.model_file):
            raise ValueError(f'Saved model [{self.model_file}] does not exist')
        checkpoint = torch.load(self.model_file)
        if hasattr(self, 'model'):
            self.model.load_state_dict(checkpoint['model_state_dict'])
        if hasattr(self, 'rec_model'):
            self.rec_model.load_state_dict(checkpoint['rec_state_dict'])
        if hasattr(self, 'conv_model'):
            self.conv_model.load_state_dict(checkpoint['conv_state_dict'])
        if hasattr(self, 'policy_model'):
            self.policy_model.load_state_dict(checkpoint['policy_state_dict'])
        logger.info(f'[Restore model from {self.model_file}]')

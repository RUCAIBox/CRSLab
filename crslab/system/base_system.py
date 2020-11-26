# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/24
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

import math
import torch
from loguru import logger
from random import shuffle
from math import ceil
import numpy as np
from copy import deepcopy
from crslab.system import *
from crslab.model import get_model
from crslab.evaluator import Evaluator
from torch import optim
import os


class BaseSystem(object):
    r"""Trainer Class is used to manage the training and evaluation processes of recommender system models.
    AbstractTrainer is an abstract class in which the fit() and evaluate() method should be implemented according
    to different training and evaluation strategies.

    For side data, it is judged by model
    """

    def __init__(self, config, train_dataloader, valid_dataloader, test_dataloader, side_data):
        self.config = config
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._number_grad_accum = 0
        self._number_training_updates = 0
        self.evaluator=Evaluator()

        self.best_valid = None
        self.val_impatience = 0
        self.val_optim = 1 if self.config["val_mode"] == "max" else -1
        self.stop = False

        self.saved_model_file=os.path.join(config['data_path'], '{}-{}.pth'.format(self.config['rec_model'],
                                                                                   self.config['conv_model']))

        if side_data!=None:
            #no matter need or not, we prepare side data
            side_data=self.side_data_prepare(side_data)

        if config['rec_model'] in ['KGSF','KBRD']:
            self.model = get_model(config, config['rec_model'], self.device, side_data)
            self.model=self.model.to(self.device)
        else:
            self.rec_model = get_model(config, config['rec_model'], self.device, side_data)
            self.rec_model=self.rec_model.to(self.device)
            self.conv_model = get_model(config, config['conv_model'], self.device, side_data)
            self.conv_model=self.conv_model.to(self.device)

    def side_data_prepare(self, side_data):
        rgcn_data = self.test_dataloader.get_side_data(side_data[0], 'RGCN')
        gcn_data = self.test_dataloader.get_side_data(side_data[1], 'GCN')
        return (rgcn_data, gcn_data)

    def fit(self, train_data):
        r"""Train the model based on the train data.

        """
        raise NotImplementedError('Method [next] should be implemented.')

    def evaluate(self, eval_data):
        r"""Evaluate the model based on the eval data.

        """
        raise NotImplementedError('Method [next] should be implemented.')

    @staticmethod
    def build_optimizer(config, parameters):
        def _optim_opts():
            opts = {k.lower(): v for k, v in optim.__dict__.items() if not k.startswith('__') and k[0].isupper()}
            return opts
        # set up optimizer args
        lr = config["learning_rate"]
        kwargs = {'lr': lr}
        if config.get('weight_decay', 0):
            kwargs['weight_decay'] = config['weight_decay']
        if config.get('momentum', 0) > 0 and config['optimizer'] in ['sgd', 'rmsprop', 'qhm']:
            # turn on momentum for optimizers that use it
            kwargs['momentum'] = config['momentum']
            if config['optimizer'] == 'sgd' and config.get('nesterov', True):
                # for sgd, maybe nesterov
                kwargs['nesterov'] = config.get('nesterov', True)
            elif config['optimizer'] == 'qhm':
                # qhm needs a nu
                kwargs['nu'] = config.get('nus', (0.7,))[0]
        elif config['optimizer'] == 'adam':
            # turn on amsgrad for `adam`
            # amsgrad paper: https://openreview.net/forum?id=ryQu7f-RZ
            kwargs['amsgrad'] = True
        elif config['optimizer'] == 'qhadam':
            # set nus for qhadam
            kwargs['nus'] = config.get('nus', (0.7, 1.0))
        elif config['optimizer'] == 'adafactor':
            # adafactor params
            kwargs['beta1'] = config.get('betas', (0.9, 0.999))[0]
            kwargs['eps'] = config['adafactor_eps']
            kwargs['warmup_init'] = config.get('warmup_updates', -1) > 0

        if config['optimizer'] in [
            'adam',
            'sparseadam',
            'fused_adam',
            'adamax',
            'qhadam',
            'fused_lamb',
        ]:
            # set betas for optims that use it
            kwargs['betas'] = config.get('betas', (0.9, 0.999))
            # set adam optimizer, but only if user specified it
            if config.get('adam_eps'):
                kwargs['eps'] = config['adam_eps']

        optim_class = _optim_opts()[config['optimizer']]
        logger.info("[Build optimizer: {}]", config["optimizer"])
        return optim_class(parameters, **kwargs)

    @staticmethod
    def build_lr_scheduler(config, optimizer):
        """
        Create the learning rate scheduler, and assign it to self.scheduler.
        This scheduler will be updated upon a call to receive_metrics.

        May also create self.warmup_scheduler, if appropriate.
        """

        if config.get('warmup_updates', -1) > 0:
            def _warmup_lr(step):
                start = config['warmup_rate']
                end = 1.0
                progress = min(1.0, step / config['warmup_updates'])
                lr_mult = start + (end - start) * progress
                return lr_mult

            warmup_scheduler = optim.lr_scheduler.LambdaLR(
                optimizer,
                _warmup_lr
            )
            logger.info("[Build warmup_scheduler]")
        else:
            warmup_scheduler = None

        patience = config.get('lr_scheduler_patience', 3)
        decay = config.get('lr_scheduler_decay', 0.5)

        if config.get('lr_scheduler', "none") == 'none':
            scheduler = None
        elif decay == 1.0:
            # warn_once(
            #     "Your LR decay is set to 1.0. Assuming you meant you wanted "
            #     "to disable learning rate scheduling. Adjust --lr-scheduler-decay "
            #     "if this is not correct."
            # )
            scheduler = None
        elif config.get('lr_scheduler') == 'reduceonplateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                'min',
                factor=decay,
                patience=patience,
                verbose=True
            )
        elif config.get('lr_scheduler') == 'fixed':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                patience,
                gamma=decay,
            )
        elif config.get('lr_scheduler') == 'invsqrt':
            if config.get('warmup_updates', -1) <= 0:
                raise ValueError(
                    '--lr-scheduler invsqrt requires setting --warmup-updates'
                )
            warmup_updates = config['warmup_updates']
            decay_factor = np.sqrt(max(1, warmup_updates))

            def _invsqrt_lr(step):
                return decay_factor / np.sqrt(max(1, step))

            scheduler = optim.lr_scheduler.LambdaLR(
                optimizer,
                _invsqrt_lr,
            )
        else:
            raise ValueError(
                "Don't know what to do with lr_scheduler '{}'"
                .format(config.get('lr_scheduler'))
            )
        logger.info("[Build lr_scheduler: {}]", config.get("lr_scheduler"))
        return scheduler, warmup_scheduler

    def _zero_grad(self, optimizer):
        if self._number_grad_accum != 0:
            # if we're accumulating gradients, don't actually zero things out yet.
            return
        optimizer.zero_grad()

    def _reset_training_steps(self):
        self._number_training_updates=0

    def _is_lr_warming_up(self, warmup_scheduler=None):
        """Checks if we're warming up the learning rate."""
        return (
                warmup_scheduler is not None and
                self._number_training_updates <= self.config['warmup_updates']
        )

    def _update_params(self, optimizer, warmup_scheduler=None, scheduler=None):
        update_freq = self.config.get('update_freq', 1)
        if update_freq > 1:
            # we're doing gradient accumulation, so we don't only want to step
            # every N updates instead
            # self._number_grad_accum is updated in backward function
            if self._number_grad_accum != 0:
                return

        # keep track up number of steps, compute warmup factor
        self._number_training_updates += 1
        # compute warmup adjustment if needed
        if self.config.get('warmup_updates', -1) > 0:
            if not hasattr(self, 'warmup_scheduler'):
                raise RuntimeError(
                    'Looks like you forgot to call build_lr_scheduler'
                )
            if self._is_lr_warming_up(warmup_scheduler):
                warmup_scheduler.step(epoch=self._number_training_updates)
        if self.config.get('lr_scheduler') == 'invsqrt' and not self._is_lr_warming_up():
            # training step scheduler
            scheduler.step(self._number_training_updates)

        optimizer.step()

    def backward(self, optimizer, loss, warmup_scheduler=None, scheduler=None):
        self._zero_grad(optimizer)

        update_freq = self.config.get('update_freq', 1)
        if update_freq > 1:
            self._number_grad_accum = (self._number_grad_accum + 1) % update_freq
            if self._number_grad_accum == 0:
                # gradient accumulation, but still need to average across the minibatches
                loss = loss / update_freq
                loss.backward()
        else:
            loss.backward()

        self._update_params(optimizer, warmup_scheduler, scheduler)

    def adjust_lr(self, scheduler=None, metric=None):
        """
        Use the metric to decide when to adjust LR schedule.

        Override this to override the behavior.
        """
        if scheduler is None:
            return

        if self._is_lr_warming_up():
            # we're not done warming up, so don't start using validation
            # metrics to adjust schedule
            return

        if self.config['lr_scheduler'] == 'none':
            # no scheduler, nothing to adjust here
            pass
        elif self.config['lr_scheduler'] == 'reduceonplateau':
            scheduler.step(metric)
        elif self.config['lr_scheduler'] == 'fixed':
            scheduler.step()
        elif self.config['lr_scheduler'] == 'invsqrt':
            # this is a training step lr scheduler, nothing to adjust in validation
            pass
        else:
            raise ValueError(
                "Don't know how to work with lr scheduler '{}'".format(self.config['lr_scheduler'])
            )

    def early_stop(self, metric):
        if self.best_valid is None or self.best_valid < metric * self.val_optim:
            self.best_valid = metric
            self.val_impatience = 0
        else:
            self.val_impatience += 1
            if self.val_impatience >= self.config["val_impatience"]:
                self.stop = True

    def reset_early_stop_state(self):
        self.best_valid = None
        self.val_impatience = 0
        self.stop = False

    def reset_metrics(self):
        self.evaluator.reset_metrics()

    def setup_ind2token(self, ind2token):
        self.ind2token = ind2token

    def inds2txt(self, indexes):
        sentence=[]
        for index in indexes:
            if index == self.config['end_token_idx']:
                break
            sentence.append(self.ind2token.get(index, 'unk'))
        return ' '.join(sentence)

    def save_system(self):
        r"""Store the model parameters information and training information.

        Args:
            epoch (int): the current epoch id
            saved_model_file (str): file name for saved pretrained model

        """
        if hasattr(self, 'model'):
            state = {
                'config': self.config,
                'model_state_dict': self.model.state_dict(),
            }
        else:
            state = {
                'config': self.config,
                'rec_state_dict': self.rec_model.state_dict(),
                'conv_state_dict': self.conv_model.state_dict(),
            }
        torch.save(state, self.saved_model_file)

    def load_system(self):
        r"""Store the model parameters information and training information.

        Args:
            epoch (int): the current epoch id
            saved_model_file (str): file name for saved pretrained model

        """
        checkpoint = torch.load(self.saved_model_file)
        if hasattr(self, 'model'):
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.rec_model.load_state_dict(checkpoint['rec_state_dict'])
            self.conv_model.load_state_dict(checkpoint['conv_state_dict'])

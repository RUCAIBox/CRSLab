# @Time   : 2020/12/1
# @Author : Xiaolei Wang
# @Email  : wxl1999@foxmail.com

from abc import abstractmethod, ABC

# UPDATE:
# @Time   : 2020/12/14
# @Author : Xiaolei Wang
# @Email  : wxl1999@foxmail.com
import math
import numpy as np
import torch
from loguru import logger
from torch import optim


class LRScheduler(ABC):
    """
    Class for LR Schedulers.

    Includes some basic functionality by default - setting up the warmup
    scheduler, passing the correct number of steps to train_step, loading and
    saving states.
    Subclasses must implement abstract methods train_step() and valid_step().
    Schedulers should be initialized with lr_scheduler_factory().
    __init__() should not be called directly.
    """

    def __init__(self, optimizer, warmup_steps: int = 0):
        """
        Initialize warmup scheduler. Specific main schedulers should be initialized in
        the subclasses. Do not invoke this method diretly.

        :param optimizer optimizer:
            Optimizer being used for training. May be wrapped in
            fp16_optimizer_wrapper depending on whether fp16 is used.
        :param int warmup_steps:
            Number of training step updates warmup scheduler should take.
        """
        self._number_training_updates = 0
        self.warmup_steps = warmup_steps
        self._init_warmup_scheduler(optimizer)

    def _warmup_lr(self, step):
        """
        Return lr multiplier (on initial lr) for warmup scheduler.
        """
        return float(step) / float(max(1, self.warmup_steps))

    def _init_warmup_scheduler(self, optimizer):
        if self.warmup_steps > 0:
            self.warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, self._warmup_lr)
        else:
            self.warmup_scheduler = None

    def _is_lr_warming_up(self):
        """
        Check if we're warming up the learning rate.
        """
        return (
                hasattr(self, 'warmup_scheduler')
                and self.warmup_scheduler is not None
                and self._number_training_updates <= self.warmup_steps
        )

    def train_step(self):
        """
        Use the number of train steps to adjust the warmup scheduler or the main
        scheduler, depending on where in training we are.

        Override this method to override the behavior for training schedulers.
        """
        self._number_training_updates += 1
        if self._is_lr_warming_up():
            self.warmup_scheduler.step()
        else:
            self.train_adjust()

    def valid_step(self, metric=None):
        if self._is_lr_warming_up():
            # we're not done warming up, so don't start using validation
            # metrics to adjust schedule
            return
        self.valid_adjust(metric)

    @abstractmethod
    def train_adjust(self):
        """
        Use the number of train steps to decide when to adjust LR schedule.

        Override this method to override the behavior for training schedulers.
        """
        pass

    @abstractmethod
    def valid_adjust(self, metric):
        """
        Use the metrics to decide when to adjust LR schedule.

        This uses the loss as the validation metric if present, if not this
        function does nothing. Note that the model must be reporting loss for
        this to work.

        Override this method to override the behavior for validation schedulers.
        """
        pass


class ReduceLROnPlateau(LRScheduler):
    """
    Scheduler that decays by a multiplicative rate when valid loss plateaus.
    """

    def __init__(self, optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001,
                 threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, warmup_steps=0):
        super(ReduceLROnPlateau, self).__init__(optimizer, warmup_steps)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode=mode, factor=factor,
                                                              patience=patience, verbose=verbose, threshold=threshold,
                                                              threshold_mode=threshold_mode, cooldown=cooldown,
                                                              min_lr=min_lr, eps=eps)

    def train_adjust(self):
        pass

    def valid_adjust(self, metric):
        self.scheduler.step(metric)


class StepLR(LRScheduler):
    """
    Scheduler that decays by a fixed multiplicative rate at each valid step.
    """

    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1, warmup_steps=0):
        super(StepLR, self).__init__(optimizer, warmup_steps)
        self.scheduler = optim.lr_scheduler.StepLR(optimizer, step_size, gamma, last_epoch)

    def train_adjust(self):
        pass

    def valid_adjust(self, metric=None):
        self.scheduler.step()


class ConstantLR(LRScheduler):
    def __init__(self, optimizer, warmup_steps=0):
        super(ConstantLR, self).__init__(optimizer, warmup_steps)

    def train_adjust(self):
        pass

    def valid_adjust(self, metric):
        pass


class InvSqrtLR(LRScheduler):
    """
    Scheduler that decays at an inverse square root rate.
    """

    def __init__(self, optimizer, invsqrt_lr_decay_gamma=-1, last_epoch=-1, warmup_steps=0):
        """
        invsqrt_lr_decay_gamma determines the cycle length of the inverse square root
        scheduler.

        When steps taken == invsqrt_lr_decay_gamma, the lr multiplier is 1
        """
        super(InvSqrtLR, self).__init__(optimizer, warmup_steps)
        self.invsqrt_lr_decay_gamma = invsqrt_lr_decay_gamma
        if invsqrt_lr_decay_gamma <= 0:
            logger.warning(
                '--lr-scheduler invsqrt requires a value for '
                '--invsqrt-lr-decay-gamma. Defaulting to set gamma to '
                '--warmup-updates value for backwards compatibility.'
            )
            self.invsqrt_lr_decay_gamma = self.warmup_steps

        self.decay_factor = np.sqrt(max(1, self.invsqrt_lr_decay_gamma))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, self._invsqrt_lr, last_epoch)

    def _invsqrt_lr(self, step):
        return self.decay_factor / np.sqrt(max(1, self.invsqrt_lr_decay_gamma + step))

    def train_adjust(self):
        self.scheduler.step()

    def valid_adjust(self, metric):
        # this is a training step lr scheduler, nothing to adjust in validation
        pass


class CosineAnnealingLR(LRScheduler):
    """
    Scheduler that decays by a cosine function.
    """

    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, warmup_steps=0):
        """
        training_steps determines the cycle length of the cosine annealing.

        It indicates the number of steps from 1.0 multiplier to 0.0, which corresponds
        to going from cos(0) to cos(pi)
        """
        super(CosineAnnealingLR, self).__init__(optimizer, warmup_steps)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min, last_epoch)

    def train_adjust(self):
        self.scheduler.step()

    def valid_adjust(self, metric):
        pass


class CosineAnnealingWarmRestartsLR(LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1, warmup_steps=0):
        super(CosineAnnealingWarmRestartsLR, self).__init__(optimizer, warmup_steps)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0, T_mult, eta_min, last_epoch)

    def train_adjust(self):
        self.scheduler.step()

    def valid_adjust(self, metric):
        pass


class TransformersLinearLR(LRScheduler):
    """
    Scheduler that decays linearly.
    """

    def __init__(self, optimizer, training_steps, warmup_steps=0):
        """
        training_steps determines the cycle length of the linear annealing.

        It indicates the number of steps from 1.0 multiplier to 0.0
        """
        super(TransformersLinearLR, self).__init__(optimizer, warmup_steps)
        self.training_steps = training_steps - warmup_steps
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, self._linear_lr)

    def _linear_lr(self, step):
        return max(0.0, float(self.training_steps - step) / float(max(1, self.training_steps)))

    def train_adjust(self):
        self.scheduler.step()

    def valid_adjust(self, metric):
        pass


class TransformersCosineLR(LRScheduler):
    def __init__(self, optimizer, training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1,
                 warmup_steps: int = 0):
        super(TransformersCosineLR, self).__init__(optimizer, warmup_steps)
        self.training_steps = training_steps - warmup_steps
        self.num_cycles = num_cycles
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, self._cosine_lr, last_epoch)

    def _cosine_lr(self, step):
        progress = float(step) / float(max(1, self.training_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(self.num_cycles) * 2.0 * progress)))

    def train_adjust(self):
        self.scheduler.step()

    def valid_adjust(self, metric):
        pass


class TransformersCosineWithHardRestartsLR(LRScheduler):
    def __init__(self, optimizer, training_steps: int, num_cycles: int = 1, last_epoch: int = -1,
                 warmup_steps: int = 0):
        super(TransformersCosineWithHardRestartsLR, self).__init__(optimizer, warmup_steps)
        self.training_steps = training_steps - warmup_steps
        self.num_cycles = num_cycles
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, self._cosine_with_hard_restarts_lr, last_epoch)

    def _cosine_with_hard_restarts_lr(self, step):
        progress = float(step) / float(max(1, self.training_steps))
        if progress >= 1.0:
            return 0.0
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(self.num_cycles) * progress) % 1.0))))

    def train_adjust(self):
        self.scheduler.step()

    def valid_adjust(self, metric):
        pass


class TransformersPolynomialDecayLR(LRScheduler):
    def __init__(self, optimizer, training_steps, lr_end=1e-7, power=1.0, last_epoch=-1, warmup_steps=0):
        super(TransformersPolynomialDecayLR, self).__init__(optimizer, warmup_steps)
        self.training_steps = training_steps - warmup_steps
        self.lr_init = optimizer.defaults["lr"]
        self.lr_end = lr_end
        assert self.lr_init > lr_end, f"lr_end ({lr_end}) must be be smaller than initial lr ({self.lr_init})"
        self.power = power
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, self._polynomial_decay_lr, last_epoch)

    def _polynomial_decay_lr(self, step):
        if step > self.training_steps:
            return self.lr_end / self.lr_init  # as LambdaLR multiplies by lr_init
        else:
            lr_range = self.lr_init - self.lr_end
            decay_steps = self.training_steps
            pct_remaining = 1 - step / decay_steps
            decay = lr_range * pct_remaining ** self.power + self.lr_end
            return decay / self.lr_init  # as LambdaLR multiplies by lr_init

    def train_adjust(self):
        self.scheduler.step()

    def valid_adjust(self, metric):
        pass

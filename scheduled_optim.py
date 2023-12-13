import numpy as np
import torch.optim as optim

class ScheduledOptim:
    """
    A wrapper class for learning rate scheduling with an optimizer.

    Args:
        config (object): Configuration object containing hyperparameters.
            - n_warmup_steps (int): Number of warmup steps for learning rate.
            - d_model (int): Dimensionality of the model.

    Attributes:
        optimizer (torch.optim.Optimizer): The inner optimizer.
        n_warmup_steps (int): Number of warmup steps for learning rate.
        n_current_steps (int): Current number of steps.
        init_lr (float): Initial learning rate.
    """

    def __init__(self, config, optimizer):
        """
        Initialize the ScheduledOptim.
        """
        self.optimizer = optimizer
        self.n_warmup_steps: int = config.n_warmup_steps
        self.d_model = config.hidden_size
        self.n_current_steps: int = 0
        self.init_lr: float =  self.d_model** -0.5

    def step_and_update_lr(self):
        """Step with the inner optimizer and update learning rate."""
        self._update_learning_rate()
        self.optimizer.step()

    def zero_grad(self):
        """Zero out the gradients by the inner optimizer."""
        self.optimizer.zero_grad()

    def _get_lr_scale(self):
        """Calculate the learning rate scale based on the current steps."""
        return min([
            self.n_current_steps ** -0.5,
            (self.n_warmup_steps ** -1.5) * self.n_current_steps
        ])

    def _update_learning_rate(self):
        """Update learning rate based on the learning rate scale."""
        self.n_current_steps += 1
        lr: float = self.init_lr * self._get_lr_scale()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 19:41:39 2021

@author: user
"""

from typing import Optional, Any, Tuple
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Function
## implementation of domain adversarial traning. For more details, please visit: https://dalib.readthedocs.io/en/latest/index.html
def binary_accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """Computes the accuracy for binary classification"""
    with torch.no_grad():
        batch_size = target.size(0)
        pred = (output >= 0.5).float().t().view(-1)
        correct = pred.eq(target.view(-1)).float().sum()
        correct.mul_(100. / batch_size)
        return correct

class GradientReverseFunction(Function):

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None

class GradientReverseLayer(nn.Module):
    def __init__(self):
        super(GradientReverseLayer, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)

class WarmStartGradientReverseLayer(nn.Module):

    def __init__(self, alpha: Optional[float] = 1.0, lo: Optional[float] = 0.0, hi: Optional[float] = 1.,
                 max_iters: Optional[int] = 1000., auto_step: Optional[bool] = False):
        super(WarmStartGradientReverseLayer, self).__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """"""
        coeff = np.float64(
            2.0 * (self.hi - self.lo) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters))
            - (self.hi - self.lo) + self.lo
        )
        if self.auto_step:
            self.step()
        return GradientReverseFunction.apply(input, coeff)

    def step(self):
        """Increase iteration number :math:`i` by 1"""
        self.iter_num += 1

class DomainAdversarialLoss(nn.Module):
    def __init__(self, domain_discriminator: nn.Module,reduction: Optional[str] = 'mean',max_iter=1000):
        super(DomainAdversarialLoss, self).__init__()
        self.grl = WarmStartGradientReverseLayer(alpha=1.0, lo=0., hi=1., max_iters=max_iter, auto_step=True)
        self.domain_discriminator = domain_discriminator
        self.bce = nn.BCEWithLogitsLoss(reduction=reduction)
        self.domain_discriminator_accuracy = None

    def forward(self, f_s: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
        f = self.grl(torch.cat((f_s, f_t), dim=0))
        d = self.domain_discriminator(f)
        d_s, d_t = d.chunk(2, dim=0)
        d_label_s = torch.ones((f_s.size(0), 1)).to(f_s.device)
        d_label_t = torch.zeros((f_t.size(0), 1)).to(f_t.device)
        self.domain_discriminator_accuracy = 0.5 * (binary_accuracy(d_s, d_label_s) + binary_accuracy(d_t, d_label_t))
        return 0.5 * (self.bce(d_s, d_label_s) + self.bce(d_t, d_label_t))


class DAANLoss(nn.Module):
    def __init__(self, domain_discriminator: nn.Module, num_class=3, reduction: Optional[str] = 'mean',max_iter=1000):
        super(DAANLoss, self).__init__()
        self.num_class = num_class
        self.grl = WarmStartGradientReverseLayer(alpha=1.0, lo=0., hi=1., max_iters=max_iter, auto_step=True)
        self.bce = nn.BCEWithLogitsLoss(reduction=reduction)
        self.local_classifiers = torch.nn.ModuleList()
        self.global_classifiers = domain_discriminator
        for _ in range(num_class):
            self.local_classifiers.append(domain_discriminator)

        self.d_g, self.d_l = 0, 0
        self.dynamic_factor = 0.5

    def forward(self, source, target, source_logits, target_logits):

        global_loss = self.get_global_adversarial_result(source, target)

        #
        # self.d_g = self.d_g + 2 * (1 - 2 * global_loss.cpu().item())
        # self.d_l = self.d_l + 2 * (1 - 2 * (local_loss / self.num_class).cpu().item())

        # adv_loss = (1 - self.dynamic_factor) * global_loss + self.dynamic_factor * local_loss
        return global_loss

    def get_global_adversarial_result(self, f_s, f_t):
        f = self.grl(torch.cat((f_s, f_t), dim=0))
        d = self.global_classifiers(f)
        d_s, d_t = d.chunk(2, dim=0)
        d_label_s = torch.ones((f_s.size(0), 1)).to(f_s.device)
        d_label_t = torch.zeros((f_t.size(0), 1)).to(f_t.device)
        return 0.5 * (self.bce(d_s, d_label_s) + self.bce(d_t, d_label_t))


    def get_local_adversarial_result(self, feat, logits, source=True):
        loss_adv = 0.0
        for c in range(self.num_class):
            x = feat[c + 1]
            x = self.grl(x)
            softmax_logits = torch.nn.functional.softmax(logits, dim=1)
            logits_c = logits[:, c].reshape((softmax_logits.shape[0], 1)) # (B, 1)
            features_c = logits_c * x
            domain_pred = self.local_classifiers[c](features_c)
            device = domain_pred.device
            if source:
                domain_label = torch.ones(x.size(0), 1).to(device)
            else:
                domain_label = torch.zeros(x.size(0), 1).to(device)
            loss_adv = loss_adv + self.bce(domain_pred, domain_label)
        return 0.5 * loss_adv

    def update_dynamic_factor(self, epoch_length):
        if self.d_g == 0 and self.d_l == 0:
            self.dynamic_factor = 0.5
        else:
            self.d_g = self.d_g / epoch_length
            self.d_l = self.d_l / epoch_length
            self.dynamic_factor = 1 - self.d_g / (self.d_g + self.d_l)
        self.d_g, self.d_l = 0, 0
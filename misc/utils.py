from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import random



def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def build_optimizer(params, opt):
    if opt.optim == "rmsprop":
        return optim.RMSprop(
            params,
            opt.learning_rate,
            opt.optim_alpha,
            opt.optim_epsilon,
            weight_decay=opt.weight_decay,
        )
    elif opt.optim == "adagrad":
        return optim.Adagrad(params, opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optim == "sgd":
        return optim.SGD(params, opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optim == "sgdm":
        return optim.SGD(
            params, opt.learning_rate, opt.optim_alpha, weight_decay=opt.weight_decay
        )
    elif opt.optim == "sgdmom":
        return optim.SGD(
            params,
            opt.learning_rate,
            opt.optim_alpha,
            weight_decay=opt.weight_decay,
            nesterov=True,
        )
    elif opt.optim == "adam":
        return optim.Adam(
            params,
            opt.learning_rate,
            (opt.optim_alpha, opt.optim_beta),
            opt.optim_epsilon,
            weight_decay=opt.weight_decay,
        )
    else:
        raise Exception("bad option opt.optim: {}".format(opt.optim))

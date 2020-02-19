import torch
import torch.nn as nn
import os

from GreedyInfoMax.audio.models import full_model
from GreedyInfoMax.utils import model_utils


def load_model_and_optimizer(
    opt, reload_model=False, calc_accuracy=False, num_GPU=None
):

    # original dimensions given in CPC paper (Oord et al.)
    kernel_sizes = [10, 8, 4, 4, 4]
    strides = [5, 4, 2, 2, 2]
    padding = [2, 2, 2, 2, 1]
    enc_hidden = 512
    reg_hidden = 256

    ## initialize model
    model = full_model.FullModel(
        opt,
        kernel_sizes=kernel_sizes,
        strides=strides,
        padding=padding,
        enc_hidden=enc_hidden,
        reg_hidden=reg_hidden,
        calc_accuracy=calc_accuracy,
    )

    model = model.to(opt.device)

    # run on only one GPU for supervised losses
    if opt.loss == 2 or opt.loss == 1:
        num_GPU = 1

    """ initialize optimizers
    We need to have a separate optimizer for every individually trained part of the network
    as calling optimizer.step() would otherwise cause all parts of the network to be updated
    even when their respective gradients are zero (due to momentum)
    """
    optimizer = []
    for idx, layer in enumerate(model.fullmodel):
        if isinstance(opt.learning_rate, list):
            cur_lr = opt.learning_rate[idx]
        else:
            cur_lr = opt.learning_rate
        optimizer.append(torch.optim.Adam(layer.parameters(), lr=cur_lr))

    if opt.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Install the apex package from https://www.github.com/nvidia/apex to use fp16 for training"
            )

        print("### USING FP16 ###")
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=opt.fp16_opt_level
        )

    model, num_GPU = model_utils.distribute_over_GPUs(opt, model, num_GPU=num_GPU)
    model, optimizer = model_utils.reload_weights(opt, model, optimizer, reload_model)

    model.train()
    print(model)
    return model, optimizer


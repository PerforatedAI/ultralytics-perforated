################################################################################
# Author:                                                                      #
# Nicholas Mesa-Cucalon (nicholas@perforatedai.com)                            #
#                                                                              #
# PerforatedAI callbacks that drive dendrite switches inside the YOLO loop.    #
################################################################################

#
"""
Imports
"""
import math
import torch

from perforatedai import globals_perforatedai as GPA

from ultralytics.engine.trainer    import BaseTrainer
from ultralytics.utils             import LOGGER
from ultralytics.utils.torch_utils import ModelEMA

#
"""
Functions
"""
def check_correlation_batches_fit(trainer: BaseTrainer) -> None:
    '''
    Fail if correlation warmup is longer than one training epoch

    Notes:
        - PAI halts rather than clamps, so we catch it at startup

    Signature:
        trainer (BaseTrainer):
            - Trainer the callback is registered on
    '''
    iters_per_epoch = len(trainer.train_loader)
    warmup = GPA.pc.get_initial_correlation_batches()
    if warmup < iters_per_epoch:
        return
    raise ValueError(
        f'initial_correlation_batches ({warmup}) must be smaller than the '
        f'{iters_per_epoch} iterations in one training epoch. Lower it '
        f'with --initial-correlation-batches, or lower batch.'
    )

def rebuild_perforated_optimizer(trainer: BaseTrainer) -> None:
    '''
    Rebuild optimizer, scheduler, and EMA after a dendrite restructure

    Notes:
        - PAI replaces the param groups and drops initial_lr and
          param_group, which the warmup block reads, so we reseed them
        - EMA is rebuilt rather than resynced, because ModelEMA pairs
          tensors by state dict key and a restructure changes the keys

    Signature:
        trainer (BaseTrainer):
            - Trainer the callback is registered on
    '''
    args = trainer.args
    weight_decay = (
        args.weight_decay * trainer.batch_size * trainer.accumulate / args.nbs
    )
    iterations = trainer.epochs * math.ceil(
        len(trainer.train_loader.dataset) / max(trainer.batch_size, args.nbs)
    )
    trainer.optimizer = trainer.build_optimizer(
        model      = trainer.model,
        name       = args.optimizer,
        lr         = args.lr0,
        momentum   = args.momentum,
        decay      = weight_decay,
        iterations = iterations,
    )
    GPA.pai_tracker.set_optimizer_instance(trainer.optimizer)

    for group in trainer.optimizer.param_groups:
        group.setdefault('initial_lr', args.lr0)
        group.setdefault('param_group', 'weight')

    trainer._setup_scheduler()

    updates = trainer.ema.updates if trainer.ema is not None else 0
    trainer.ema = ModelEMA(trainer.model, updates = updates)

    LOGGER.info(
        f'PerforatedAI restructured at epoch {trainer.epoch}. Rebuilt the '
        f'optimizer with {len(trainer.optimizer.param_groups)} param '
        f'groups, reset the plateau schedule to lr {trainer.args.lr0}, '
        f'and rebuilt the EMA at {updates} updates.'
    )

def perforated_train_start(trainer: BaseTrainer) -> None:
    '''
    Hand PAI the optimizer and check the correlation warmup fits

    Signature:
        trainer (BaseTrainer):
            - Trainer the callback is registered on
    '''
    GPA.pai_tracker.set_optimizer_instance(trainer.optimizer)
    check_correlation_batches_fit(trainer)

def perforated_train_epoch_end(trainer: BaseTrainer) -> None:
    '''
    Report the epoch mean training loss to PAI

    Signature:
        trainer (BaseTrainer):
            - Trainer the callback is registered on
    '''
    if trainer.tloss is None:
        return
    # tloss is a running mean per loss component, so the total is the sum
    total = float(sum(float(value) for value in trainer.tloss.values()))
    GPA.pai_tracker.add_extra_score_without_graphing(total, 'train_loss')

def ema_weight_keys(model: torch.nn.Module) -> set:
    '''
    State dict keys of the weights validation scores

    Notes:
        - Parameters plus BatchNorm running stats. PAI's own buffers are
          left out, because copying EMA averages of those into the live
          model corrupts its dendrite bookkeeping

    Signature:
        model (torch.nn.Module):
            - Live perforated model
    '''
    keys = {name for name, _ in model.named_parameters()}
    keys |= {
        name for name, _ in model.named_buffers()
        if name.endswith(('running_mean', 'running_var'))
    }
    return keys

@torch.no_grad()
def swap_in_ema_weights(trainer: BaseTrainer) -> dict:
    '''
    Load the EMA weights into the live model

    Notes:
        - Validation scores the EMA, but PAI checkpoints the model it is
          handed, so without this the best model PAI restores at a
          switch was never scored
        - Returns the live weights it overwrote, so restore_live_weights
          can put them back if PAI does not restructure

    Signature:
        trainer (BaseTrainer):
            - Trainer the callback is registered on
    '''
    live = trainer.model.state_dict()
    ema  = trainer.ema.ema.state_dict()
    saved = {}
    for key in ema_weight_keys(trainer.model):
        saved[key] = live[key].clone()
        live[key].copy_(ema[key])
    return saved

@torch.no_grad()
def restore_live_weights(trainer: BaseTrainer, saved: dict) -> None:
    '''
    Put the live weights back after PAI has seen the EMA ones

    Signature:
        trainer (BaseTrainer):
            - Trainer the callback is registered on
        saved (dict):
            - Live weights returned by swap_in_ema_weights
    '''
    live = trainer.model.state_dict()
    for key, value in saved.items():
        live[key].copy_(value)

def perforated_fit_epoch_end(trainer: BaseTrainer) -> None:
    '''
    Feed validation fitness to PAI and act on what it decides

    Notes:
        - Fires after validate and the checkpoint write, so the model can
          be swapped before the next epoch
        - trainer.fitness is mAP50-95, the same score the mmyolo arms
          fed PAI

    Signature:
        trainer (BaseTrainer):
            - Trainer the callback is registered on
    '''
    if trainer.fitness is None:
        return
    for key, value in trainer.metrics.items():
        GPA.pai_tracker.add_extra_score_without_graphing(value, key)

    device = next(trainer.model.parameters()).device
    live_weights = swap_in_ema_weights(trainer)
    model, restructured, training_complete = (
        GPA.pai_tracker.add_validation_score(trainer.fitness, trainer.model)
    )
    if training_complete:
        LOGGER.info('PerforatedAI reported training complete, stopping.')
        trainer.stop = True
        return
    if not restructured:
        restore_live_weights(trainer, live_weights)
        return
    trainer.model = model.to(device)
    rebuild_perforated_optimizer(trainer)

def register_perforated_callbacks(trainer: BaseTrainer) -> None:
    '''
    Attach every PerforatedAI callback to the trainer

    Signature:
        trainer (BaseTrainer):
            - Trainer the callbacks are registered on
    '''
    trainer.add_callback('on_train_start', perforated_train_start)
    trainer.add_callback('on_train_epoch_end', perforated_train_epoch_end)
    trainer.add_callback('on_fit_epoch_end', perforated_fit_epoch_end)

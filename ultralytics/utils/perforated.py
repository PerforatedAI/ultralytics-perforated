# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""PerforatedAI dendrites for YOLO26. BaseTrainer imports from here when perforate=True."""

from __future__ import annotations

import math
import torch
import inspect

from typing import Any
from torch import nn, optim

from perforatedai import utils_perforatedai as UPA
from perforatedai import globals_perforatedai as GPA

from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import ModelEMA

# Types PAI wraps when perforate_modules is empty.
# Conv is the ultralytics conv block. Conv2d is the bare predictor at the end of each head branch.
PERFORATE_TYPE_NAMES = ["Conv", "Conv2d"]

# Shape -> [batch, channels, height, width]
OUTPUT_DIMENSIONS = [-1, 0, -1, -1]

# Norm layers outside a perforated block have to be tracked by type name
NORM_TYPE_NAMES = ["BatchNorm2d", "BatchNorm1d", "SyncBatchNorm"]

# Never perforated.
# dfl is a frozen Conv2d that decodes box distributions.
ALWAYS_TRACK_NAMES = ["dfl"]

# switch_mode value -> GPA.pc constant
SWITCH_MODES = {
    "history": "DOING_HISTORY",
    "fixed": "DOING_FIXED_SWITCH",
    "every_time": "DOING_SWITCH_EVERY_TIME",
    "none": "DOING_NO_SWITCH",
}

# param_vals_setting value -> GPA.pc constant
PARAM_VALS_SETTINGS = {
    "total_epoch": "PARAM_VALS_BY_TOTAL_EPOCH",
    "update_epoch": "PARAM_VALS_BY_UPDATE_EPOCH",
    "neuron_epoch_start": "PARAM_VALS_BY_NEURON_EPOCH_START",
}

# Trainer arg -> PAI name. Each one is pushed with GPA.pc.set_<PAI name>
PAI_SETTINGS = {
    "testing_dendrite_capacity": "testing_dendrite_capacity",
    "n_epochs_to_switch": "n_epochs_to_switch",
    "history_lookback": "history_lookback",
    "initial_history_after_switches": "initial_history_after_switches",
    "fixed_switch_num": "fixed_switch_num",
    "first_fixed_switch_num": "first_fixed_switch_num",
    "reset_best_score_on_switch": "reset_best_score_on_switch",
    "improvement_threshold": "improvement_threshold",
    "improvement_threshold_raw": "improvement_threshold_raw",
    "max_dendrites": "max_dendrites",
    "max_dendrite_tries": "max_dendrite_tries",
    "retain_all_dendrites": "retain_all_dendrites",
    "candidate_weight_initialization_multiplier": "candidate_weight_initialization_multiplier",
    "candidate_weight_init_by_main": "candidate_weight_init_by_main",
    "find_best_lr": "find_best_lr",
    "dont_give_up_unless_learning_rate_lowered": "dont_give_up_unless_learning_rate_lowered",
    "pai_verbose": "verbose",
    "pai_extra_verbose": "extra_verbose",
    "pai_silent": "silent",
    "drawing_pai": "drawing_pai",
    "drawing_extra_graphs": "drawing_extra_graphs",
    "save_old_graph_scores": "save_old_graph_scores",
    "test_saves": "test_saves",
    "using_safe_tensors": "using_safe_tensors",
}

# These functions only exist when perforatedbp is installed
PB_SETTINGS = {
    "initial_correlation_batches": "initial_correlation_batches",
    "p_epochs_to_switch": "p_epochs_to_switch",
    "cap_at_n": "cap_at_n",
    "pai_improvement_threshold": "pai_improvement_threshold",
    "pai_improvement_threshold_raw": "pai_improvement_threshold_raw",
}

class PlateauScheduler(optim.lr_scheduler.ReduceLROnPlateau):
    """
    ReduceLROnPlateau that ignores the arg-less step at the top of the epoch loop.

    BaseTrainer calls scheduler.step() once per epoch with no metric. 
    The true scheduler step happens in validate.
    """

    def step(self, metrics: float | None = None) -> None:
        """
        Step only when a score is supplied.

        Args:
            metrics (float | None): Validation fitness. None skips the step.
        """
        if metrics is None:
            return
        super().step(metrics)

def plateau_scheduler_args(args: Any) -> dict[str, Any]:
    """
    Return the ReduceLROnPlateau keyword arguments for a run.

    Args:
        args (IterableSimpleNamespace): Trainer args with lr0, lrf and the plateau_* keys.

    Returns:
        (dict[str, Any]): Keyword arguments for ReduceLROnPlateau. min_lr is lr0 * lrf.
    """
    return {
        "mode": "max",
        "factor": args.plateau_factor,
        "patience": args.plateau_patience,
        "threshold": args.plateau_threshold,
        "threshold_mode": "rel",
        "min_lr": args.lr0 * args.lrf,
        "cooldown": 0,
    }

def build_plateau_scheduler(trainer: Any) -> PlateauScheduler:
    """
    Build the plateau schedule that replaces the epoch keyed LR ramp.

    Args:
        trainer (BaseTrainer): Trainer holding the optimizer to schedule.

    Returns:
        (PlateauScheduler): Schedule on trainer.optimizer.
    """
    return PlateauScheduler(trainer.optimizer, **plateau_scheduler_args(trainer.args))

def pai_validation_score(trainer: Any) -> float:
    """
    Pick the score PAI is handed this epoch.

    Neuron Mode (GD/PB) -> Validation Fitness
    Dendrite Mode (PB) -> Mean best correlation

    Args:
        trainer (BaseTrainer): Trainer at on_fit_epoch_end, after validate.

    Returns:
        (float): Score for add_validation_score.
    """
    if not GPA.pc.get_perforated_backpropagation() or GPA.pai_tracker.member_vars["mode"] != "p":
        return trainer.fitness
    scores = list(GPA.pai_tracker.get_current_pb_scores().values())
    return sum(scores) / len(scores) if scores else 0.0

def build_module_ids(model: nn.Module, targets: list[str]) -> tuple[list[str], list[str]]:
    """
    Split the model into ids PAI perforates and ids it only tracks.

    Args:
        model (nn.Module): DetectionModel before PAI converts it.
        targets (list[str]): named_modules paths to perforate (i.e. model.23.one2one_cv2.0.0).

    Returns:
        perforate_ids (list[str]): Ids PAI wraps.
        track_ids (list[str]): Ids PAI tracks and leaves alone.
    """
    perforate_ids = []
    track_ids = ["." + name for name, _ in model.named_modules() if name.rsplit(".", 1)[-1] in ALWAYS_TRACK_NAMES]
    if not targets:
        return perforate_ids, track_ids

    names = {name for name, _ in model.named_modules()}
    if missing := [target for target in targets if target not in names]:
        raise ValueError(
            f"perforate_modules names modules this model does not have: {missing}. "
            "Valid paths come from model.named_modules()"
        )

    def walk(module: nn.Module, prefix: str) -> None:
        for child_name, child in module.named_children():
            path = f"{prefix}.{child_name}" if prefix else child_name
            module_id = "." + path
            if module_id in track_ids:
                continue
            if path in targets:
                perforate_ids.append(module_id)
            elif any(target.startswith(path + ".") for target in targets):
                walk(child, path)
            else:
                track_ids.append(module_id)

    walk(model, "")
    return perforate_ids, track_ids

def configure_perforated_ai(model: nn.Module, args: Any) -> list[str]:
    """
    Push every PAI setting for this run into GPA.pc.
    Explicit failure if perforated_backpropogation is true without perforatedbp installed. 

    Args:
        model (nn.Module): DetectionModel before PAI converts it.
        args (IterableSimpleNamespace): Trainer args with the perforate_* and PAI keys.

    Returns:
        (list[str]): Ids PAI was asked to perforate. Empty when PAI picks by type name.
    """
    if args.perforated_backpropagation and not GPA.pc.get_perforated_backpropagation():
        raise RuntimeError(
            "perforated_backpropagation=True but the perforatedbp package is not installed. Install it, or set "
            "perforated_backpropagation=False to train dendrites in open-source mode."
        )
    if args.switch_mode not in SWITCH_MODES:
        raise ValueError(f"switch_mode={args.switch_mode} is not one of {list(SWITCH_MODES)}")
    if args.param_vals_setting not in PARAM_VALS_SETTINGS:
        raise ValueError(f"param_vals_setting={args.param_vals_setting} is not one of {list(PARAM_VALS_SETTINGS)}")

    GPA.pc.set_perforated_backpropagation(args.perforated_backpropagation)
    GPA.pc.set_module_names_to_perforate(PERFORATE_TYPE_NAMES)
    GPA.pc.set_output_dimensions(OUTPUT_DIMENSIONS)
    GPA.pc.set_unwrapped_modules_confirmed(True)
    GPA.pc.set_switch_mode(getattr(GPA.pc, SWITCH_MODES[args.switch_mode]))
    GPA.pc.set_param_vals_setting(getattr(GPA.pc, PARAM_VALS_SETTINGS[args.param_vals_setting]))
    settings = {**PAI_SETTINGS, **PB_SETTINGS} if args.perforated_backpropagation else PAI_SETTINGS
    for arg_name, pai_name in settings.items():
        getattr(GPA.pc, f"set_{pai_name}")(getattr(args, arg_name))

    norm_types = (nn.modules.batchnorm._NormBase, nn.LayerNorm, nn.GroupNorm)
    for name, module in model.named_modules():
        if isinstance(module, norm_types):
            GPA.pc.append_module_names_to_not_save(["." + name])
    GPA.pc.append_module_names_to_track(NORM_TYPE_NAMES)

    perforate_ids, track_ids = build_module_ids(model, args.perforate_modules)
    GPA.pc.append_module_ids_to_perforate(perforate_ids)
    GPA.pc.append_module_ids_to_track(track_ids)
    return perforate_ids

def perforate_detection_model(model: nn.Module, args: Any, save_name: str) -> nn.Module:
    """
    Configure PAI and wrap the model.

    Args:
        model (nn.Module): DetectionModel with pretrained weights loaded.
        args (IterableSimpleNamespace): Trainer args with the perforate_* and PAI keys.
        save_name (str): Folder PAI writes its saved system into.

    Returns:
        (nn.Module): Perforated model. Loaded from pai_load_folder when that is set.
    """
    perforate_ids = configure_perforated_ai(model, args)
    if perforate_ids:
        LOGGER.info(f"PerforatedAI perforating {len(perforate_ids)} modules:\n  " + "\n  ".join(perforate_ids))
    else:
        LOGGER.info(f"PerforatedAI perforating every {PERFORATE_TYPE_NAMES} in the model")
    model = UPA.perforate_model(model, save_name=save_name, maximizing_score=True)

    if args.pai_load_folder:
        model = UPA.load_system(model, args.pai_load_folder, args.pai_load_stage, switch_call=True)
        LOGGER.info(f"PerforatedAI loaded system {args.pai_load_stage} from {args.pai_load_folder}")
    return model

def check_correlation_batches_fit(trainer: Any) -> None:
    """
    Fail if the correlation warmup is longer than one training epoch.

    Args:
        trainer (BaseTrainer): Trainer at on_train_start.
    """
    if not GPA.pc.get_perforated_backpropagation():
        return
    iters_per_epoch = len(trainer.train_loader)
    warmup = GPA.pc.get_initial_correlation_batches()
    if warmup >= iters_per_epoch:
        raise ValueError(
            f"initial_correlation_batches ({warmup}) must be smaller than the {iters_per_epoch} iterations in one "
            "training epoch. Lower it, or lower batch."
        )

def setup_perforated_optimizer(trainer: Any) -> None:
    """
    Build the optimizer and the stock ReduceLROnPlateau through PAI

    This allows PAI to perform lr_serach itself

    Args:
        trainer (BaseTrainer): Trainer whose optimizer and schedule PAI takes over.
    """
    args = trainer.args
    weight_decay = args.weight_decay * trainer.batch_size * trainer.accumulate / args.nbs
    iterations = trainer.epochs * math.ceil(len(trainer.train_loader.dataset) / max(trainer.batch_size, args.nbs))
    grouped = trainer.build_optimizer(
        model=trainer.model,
        name=args.optimizer,
        lr=args.lr0,
        momentum=args.momentum,
        decay=weight_decay,
        iterations=iterations,
    )
    accepted = inspect.signature(type(grouped).__init__).parameters
    defaults = {k: v for k, v in grouped.defaults.items() if k in accepted}
    GPA.pai_tracker.set_optimizer(type(grouped))
    GPA.pai_tracker.set_scheduler(optim.lr_scheduler.ReduceLROnPlateau)
    trainer.optimizer, _ = GPA.pai_tracker.setup_optimizer(
        trainer.model,
        {"params": [dict(group) for group in grouped.param_groups], **defaults},
        plateau_scheduler_args(args),
    )
    for group in trainer.optimizer.param_groups:
        group.setdefault("initial_lr", args.lr0)
        group.setdefault("param_group", "weight")

def rebuild_perforated_optimizer(trainer: Any) -> None:
    """
    Rebuild the optimizer, scheduler, and EMA after PAI restructured.

    Args:
        trainer (BaseTrainer): Trainer whose model PAI just restructured.
    """
    setup_perforated_optimizer(trainer)
    updates = trainer.ema.updates if trainer.ema is not None else 0
    trainer.ema = ModelEMA(trainer.model, updates=updates)
    lrs = sorted({group["lr"] for group in trainer.optimizer.param_groups})
    LOGGER.info(
        f"PerforatedAI restructured at epoch {trainer.epoch}. Rebuilt the optimizer with "
        f"{len(trainer.optimizer.param_groups)} param groups at lr {lrs}, a fresh plateau schedule, and the EMA at "
        f"{updates} updates."
    )

def perforated_train_start(trainer: Any) -> None:
    """
    Replace the trainer's optimizer and schedule with PAI's, and check the correlation warmup fits.

    Args:
        trainer (BaseTrainer): Trainer at on_train_start.
    """
    setup_perforated_optimizer(trainer)
    check_correlation_batches_fit(trainer)

def perforated_train_epoch_end(trainer: Any) -> None:
    """
    Report the epoch mean training loss to PAI.

    Args:
        trainer (BaseTrainer): Trainer at on_train_epoch_end.
    """
    if trainer.tloss is None:
        return
    # tloss is a running mean per component
    total = float(sum(float(value) for value in trainer.tloss.values()))
    GPA.pai_tracker.add_extra_score_without_graphing(total, "train_loss")

def ema_weight_keys(model: nn.Module) -> set[str]:
    """
    Return the state dict keys that validation scores.

    Ignore PAI buffers and only copy for Parameters + BatchNorm running stats

    Args:
        model (nn.Module): Live perforated model.

    Returns:
        (set[str]): Keys to swap between the live model and the EMA.
    """
    keys = {name for name, _ in model.named_parameters()}
    keys |= {name for name, _ in model.named_buffers() if name.endswith(("running_mean", "running_var"))}
    return keys

@torch.no_grad()
def swap_in_ema_weights(trainer: Any) -> dict[str, torch.Tensor]:
    """
    Copy the EMA weights into the live model.

    Allows validation to score the EMA model and maintain best performance at validation time.

    Args:
        trainer (BaseTrainer): Trainer at on_fit_epoch_end.

    Returns:
        (dict[str, torch.Tensor]): The live weights that were overwritten, for restore_live_weights.
    """
    live = trainer.model.state_dict()
    ema = trainer.ema.ema.state_dict()
    saved = {}
    for key in ema_weight_keys(trainer.model):
        saved[key] = live[key].clone()
        live[key].copy_(ema[key])
    return saved

@torch.no_grad()
def restore_live_weights(trainer: Any, saved: dict[str, torch.Tensor]) -> None:
    """
    Replace live weights back after PAI has seen the EMA ones.

    Args:
        trainer (BaseTrainer): Trainer at on_fit_epoch_end.
        saved (dict[str, torch.Tensor]): Weights returned by swap_in_ema_weights.
    """
    live = trainer.model.state_dict()
    for key, value in saved.items():
        live[key].copy_(value)

def perforated_fit_epoch_end(trainer: Any) -> None:
    """
    Give PAI the validation fitness and act on what it decides.

    Args:
        trainer (BaseTrainer): Trainer at on_fit_epoch_end.
    """
    if trainer.fitness is None:
        return
    for key, value in trainer.metrics.items():
        GPA.pai_tracker.add_extra_score_without_graphing(value, key)

    device = next(trainer.model.parameters()).device
    live_weights = swap_in_ema_weights(trainer)
    score = pai_validation_score(trainer)
    model, restructured, training_complete = GPA.pai_tracker.add_validation_score(score, trainer.model)
    if training_complete:
        LOGGER.info("PerforatedAI reported training complete, stopping.")
        trainer.stop = True
        return
    if not restructured:
        restore_live_weights(trainer, live_weights)
        return
    trainer.model = model.to(device)
    rebuild_perforated_optimizer(trainer)

def register_perforated_callbacks(trainer: Any) -> None:
    """
    Attach the three PAI callbacks to the trainer.

    Args:
        trainer (BaseTrainer): Trainer being set up.
    """
    trainer.add_callback("on_train_start", perforated_train_start)
    trainer.add_callback("on_train_epoch_end", perforated_train_epoch_end)
    trainer.add_callback("on_fit_epoch_end", perforated_fit_epoch_end)

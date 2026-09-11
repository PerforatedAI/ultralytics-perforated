################################################################################
# Author:                                                                      #
# Nicholas Mesa-Cucalon (nicholas@perforatedai.com)                            #
#                                                                              #
# PerforatedAI configuration and model perforation for YOLO26.                 #
################################################################################

#
"""
Imports
"""
import torch.nn as nn

from pathlib           import Path
from ultralytics.utils import LOGGER
from datetime          import datetime
from dataclasses       import dataclass
from typing            import List, Optional, Tuple
from perforatedai      import utils_perforatedai as UPA
from perforatedai      import globals_perforatedai as GPA

#
"""
Config
"""
# Head branch that receives dendrites. The one2one branch is the one that
# runs at inference, the one2many branch is dropped by fuse
perforate_branch = 'one2one_cv2'

# Positions perforated at each pyramid level of that branch. Position 0
# is the Conv that compresses channels at its level
perforate_positions = {
    0 : (0,),
    1 : (0,),
    2 : (0,),
}

# Module type names PAI may wrap. Conv is the ultralytics conv block,
# Conv2d is the bare predictor at the end of each branch
perforate_type_names = ['Conv', 'Conv2d']

# Shape -> [batch, channels, height, width]
output_dimensions = [-1, 0, -1, -1]

# Norm layers outside a perforated block must be tracked by type name
norm_type_names = ['BatchNorm2d', 'BatchNorm1d', 'SyncBatchNorm']

# ReduceLROnPlateau settings read by PerforatedDetectionTrainer. Two drops
# fit inside n_epochs_to_switch, and threshold matches PAI's own
# improvement bar so the two counters agree
plateau_patience  = 6
plateau_factor    = 0.1
plateau_threshold = 0.001


@dataclass
class PerforatedRun:
    '''
    Settings for one PerforatedAI run

    Notes:
        - Held as a class attribute on the trainer, because ultralytics
          rejects any override key missing from default.yaml

    Signature:
        enabled (bool):
            - Whether dendrite training runs at all
        n_epochs_to_switch (int):
            - Validations without improvement before PAI switches cycle
        initial_correlation_batches (int):
            - Batches of correlation warmup before dendrite weights move
        fixed_switch_every (Optional[int]):
            - Epochs between forced switches, None for adaptive mode
        save_name (str):
            - Name of the PAI system folder written for this run
        load_folder (Optional[str]):
            - Saved PAI system folder to resume from
        load_stage (str):
            - Stage inside that folder to resume from
    '''
    enabled                     : bool          = False
    n_epochs_to_switch          : int           = 5
    initial_correlation_batches : int           = 20
    fixed_switch_every          : Optional[int] = None
    save_name                   : str           = 'yolo26_dendritic'
    load_folder                 : Optional[str] = None
    load_stage                  : str           = 'latest'

#
"""
Functions
"""
def build_perforated_save_name(model_name: str) -> str:
    '''
    Build the PAI system folder name for a run

    Signature:
        model_name (str):
            - Checkpoint or config name the run starts from
    '''
    stem = Path(model_name).stem
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # Folder PAI writes its saved system into
    return f'{stem}_dendritic_{timestamp}'

def build_head_module_ids(model: nn.Module) -> Tuple[List[str], List[str]]:
    '''
    Split the model into ids PAI perforates and ids it only tracks

    Notes:
        - PAI ids are the named_modules name with a leading dot
        - A tracked id stops PAI descending into that module, so the
          branch is listed position by position

    Signature:
        model (nn.Module):
            - DetectionModel before PAI converts it
    '''
    layers = model.model
    head_index = len(layers) - 1
    head = layers[head_index]
    if not hasattr(head, perforate_branch):
        raise ValueError(
            f'Detection head {type(head).__name__} has no '
            f'{perforate_branch}, so it is not a YOLO26 end to end head.'
        )
    base = f'.model.{head_index}'
    branch = f'{base}.{perforate_branch}'

    # Every position lands in one list, or PAI perforates it by type name
    perforate_ids = []
    track_ids = []
    for level, level_module in enumerate(getattr(head, perforate_branch)):
        positions = perforate_positions.get(level, ())
        for position in range(len(level_module)):
            module_id = f'{branch}.{level}.{position}'
            if position in positions:
                perforate_ids.append(module_id)
            else:
                track_ids.append(module_id)

    # Backbone, neck, and the other head branches only get tracked
    track_ids += [f'.model.{index}' for index in range(head_index)]
    track_ids += [
        f'{base}.cv2',
        f'{base}.cv3',
        f'{base}.one2one_cv3',
        f'{base}.dfl',
    ]
    return perforate_ids, track_ids


def configure_perforated_ai(
    model : nn.Module,
    run   : PerforatedRun,
) -> Tuple[List[str], List[str]]:
    '''
    Push every PAI setting for this run into the global config

    Notes:
        - Fixed switching ignores PAI's improvement tests, which is how
          a short run is forced through a full dendrite cycle

    Signature:
        model (nn.Module):
            - DetectionModel before PAI converts it
        run (PerforatedRun):
            - Settings for this run
    '''
    GPA.pc.set_testing_dendrite_capacity(False)
    GPA.pc.set_module_names_to_perforate(perforate_type_names)
    GPA.pc.set_output_dimensions(output_dimensions)
    GPA.pc.set_n_epochs_to_switch(run.n_epochs_to_switch)
    GPA.pc.set_initial_correlation_batches(run.initial_correlation_batches)
    GPA.pc.set_max_dendrite_tries(2)
    GPA.pc.set_perforated_backpropagation(True)
    GPA.pc.set_unwrapped_modules_confirmed(True)

    if run.fixed_switch_every is not None:
        GPA.pc.set_switch_mode(GPA.pc.DOING_FIXED_SWITCH)
        GPA.pc.set_first_fixed_switch_num(run.fixed_switch_every)
        GPA.pc.set_fixed_switch_num(run.fixed_switch_every)

    norm_types = (
        nn.modules.batchnorm._NormBase,
        nn.LayerNorm,
        nn.GroupNorm,
    )
    for name, module in model.named_modules():
        if isinstance(module, norm_types):
            GPA.pc.append_module_names_to_not_save(['.' + name])
    GPA.pc.append_module_names_to_track(norm_type_names)

    perforate_ids, track_ids = build_head_module_ids(model)
    GPA.pc.append_module_ids_to_perforate(perforate_ids)
    GPA.pc.append_module_ids_to_track(track_ids)
    return perforate_ids, track_ids


def verify_perforated_wrapping(
    model         : nn.Module,
    perforate_ids : List[str],
) -> None:
    '''
    Fail now if PAI wrapped a different number of modules than requested

    Notes:
        - Counts rather than names, because PAI renests a wrapped module
          under its own path

    Signature:
        model (nn.Module):
            - Model returned by perforate_model
        perforate_ids (List[str]):
            - Module ids the run asked PAI to perforate
    '''
    wrapped = [
        '.' + name
        for name, module in model.named_modules()
        if type(module).__name__ == 'PAINeuronModule'
    ]
    if len(wrapped) == len(perforate_ids):
        LOGGER.info(
            f'PerforatedAI wrapped {len(wrapped)} modules, as requested'
        )
        return
    raise RuntimeError(
        f'PerforatedAI wrapped {len(wrapped)} modules but '
        f'{len(perforate_ids)} were requested, which usually means a '
        f'parent Conv and its child Conv2d were both wrapped.\n'
        f'Wrapped:\n  ' + '\n  '.join(wrapped) + '\n'
        f'Requested:\n  ' + '\n  '.join(perforate_ids)
    )

def perforate_detection_model(
    model : nn.Module,
    run   : PerforatedRun,
) -> nn.Module:
    '''
    Configure PAI, wrap the model, and verify the wrapping

    Signature:
        model (nn.Module):
            - DetectionModel with pretrained weights already loaded
        run (PerforatedRun):
            - Settings for this run
    '''
    perforate_ids, _ = configure_perforated_ai(model, run)
    LOGGER.info(
        f'PerforatedAI perforating {len(perforate_ids)} modules:\n  '
        + '\n  '.join(perforate_ids)
    )
    model = UPA.perforate_model(
        model,
        save_name        = run.save_name,
        maximizing_score = True,
    )
    verify_perforated_wrapping(model, perforate_ids)

    if run.load_folder is not None:
        model = UPA.load_system(
            model,
            run.load_folder,
            run.load_stage,
            switch_call = True,
        )
        LOGGER.info(
            f'PerforatedAI loaded system {run.load_stage} from '
            f'{run.load_folder}'
        )
    return model

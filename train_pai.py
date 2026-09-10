################################################################################
# Author:                                                                      #
# Nicholas Mesa-Cucalon (nicholas@perforatedai.com)                            #
#                                                                              #
# Entry point for YOLO26 training with PerforatedAI.                           #
################################################################################

#
"""
Imports
"""
import argparse
import yaml

from ultralytics import YOLO

from pai import (
    PerforatedDetectionTrainer,
    PerforatedRun,
    build_perforated_save_name,
)

#
"""
Functions
"""
def parse_args() -> argparse.Namespace:
    '''
    Parse the command line arguments for a YOLO26 PerforatedAI run
    '''
    parser = argparse.ArgumentParser(
        description = 'Train YOLO26 with optional PerforatedAI dendrites'
    )
    parser.add_argument(
        'config',
        type = str,
        help = 'YAML file of ultralytics train arguments',
    )
    parser.add_argument(
        '--perforate-model',
        action  = 'store_true',
        default = False,
        help    = 'Enable PerforatedAI dendrite training',
    )
    parser.add_argument(
        '--n-epochs-to-switch',
        type    = int,
        default = 5,
        help    = 'Validations without improvement before PAI switches '
                  'cycle. Counted in validations rather than train '
                  'epochs, so pair it with val True.',
    )
    parser.add_argument(
        '--initial-correlation-batches',
        type    = int,
        default = 20,
        help    = 'Batches of correlation warmup before dendrite weights '
                  'start updating. Must be smaller than the iterations '
                  'in one training epoch.',
    )
    parser.add_argument(
        '--fixed-switch-every',
        type    = int,
        default = 0,
        help    = 'Switch cycle every N epochs regardless of score. 0 '
                  'leaves PAI in its adaptive history mode, where a '
                  'dendrite cycle only ends once no node improves its '
                  'correlation.',
    )
    parser.add_argument(
        '--name',
        type    = str,
        default = '',
        help    = 'Run name, overriding the name in the config. Lets '
                  'both arms of a comparison share one config file.',
    )
    parser.add_argument(
        '--load-folder',
        type    = str,
        default = '',
        help    = 'Saved PerforatedAI system folder to resume from. An '
                  'empty value starts a fresh dendrite cycle.',
    )
    parser.add_argument(
        '--load-stage',
        type    = str,
        default = 'latest',
        help    = 'Stage inside --load-folder to resume from, naming a '
                  'file PAI wrote there without its .pt suffix, i.e. '
                  'latest, best_model, or switch_3.',
    )
    # Parsed arguments for this run
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.load_folder and not args.perforate_model:
        raise ValueError(
            '--load-folder needs --perforate-model, because PAI modules '
            'must exist before a saved system can be loaded into them.'
        )

    with open(args.config, 'r') as handle:
        train_args = yaml.safe_load(handle)

    if 'model' not in train_args:
        raise ValueError(
            f'{args.config} has no model key, so there is nothing to '
            f'load weights from. Add one, i.e. model: yolo26n.pt'
        )
    model_name = train_args.pop('model')

    if args.name:
        train_args['name'] = args.name

    # PAI owns stopping, so the ultralytics early stopper is disabled.
    # patience 0 becomes float('inf') in EarlyStopping
    if args.perforate_model:
        train_args['patience'] = 0

    PerforatedDetectionTrainer.run = PerforatedRun(
        enabled                     = args.perforate_model,
        n_epochs_to_switch          = args.n_epochs_to_switch,
        initial_correlation_batches = args.initial_correlation_batches,
        fixed_switch_every          = args.fixed_switch_every or None,
        save_name                   = build_perforated_save_name(model_name),
        load_folder                 = args.load_folder or None,
        load_stage                  = args.load_stage,
    )

    YOLO(model_name).train(
        trainer = PerforatedDetectionTrainer,
        **train_args,
    )

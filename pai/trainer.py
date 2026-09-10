################################################################################
# Author:                                                                      #
# Nicholas Mesa-Cucalon (nicholas@perforatedai.com)                            #
#                                                                              #
# DetectionTrainer and DetectionValidator subclasses for YOLO26 with PAI.      #
################################################################################

#
"""
Imports
"""
from copy         import copy
from torch        import optim
from typing       import Any, Dict, Optional, Tuple
from perforatedai import utils_perforatedai as UPA
from perforatedai import globals_perforatedai as GPA

from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator

from .callback import perforated_fit_epoch_end, register_perforated_callbacks
from .setup    import (
    PerforatedRun,
    perforate_detection_model,
    plateau_factor,
    plateau_patience,
    plateau_threshold,
)

#
"""
Classes
"""
class PlateauScheduler(optim.lr_scheduler.ReduceLROnPlateau):
    '''
    ReduceLROnPlateau that ignores the trainer's arg-less epoch step

    Notes:
        - BaseTrainer calls scheduler.step() once per epoch with no
          metric. The real step happens in validate
    '''

    def step(self, metrics: Optional[float] = None) -> None:
        '''
        Step the schedule only when a validation score is supplied

        Signature:
            metrics (Optional[float]):
                - Validation fitness, or None to skip the step
        '''
        if metrics is None:
            return
        super().step(metrics)


class PerforatedDetectionValidator(DetectionValidator):
    '''
    DetectionValidator that also reports mAP75

    Notes:
        - Added to the stats dict rather than DetMetrics.keys, because
          fitness weights mean_results by a fixed length four vector
    '''

    def get_stats(self) -> Dict[str, Any]:
        '''
        Add mAP75 to the metrics a validation reports
        '''
        stats = super().get_stats()
        stats['metrics/mAP75(B)'] = float(self.metrics.box.map75)
        return stats


class PerforatedDetectionTrainer(DetectionTrainer):
    '''
    DetectionTrainer that adds PerforatedAI dendrites to the YOLO26 head

    Notes:
        - run is a class attribute, because ultralytics rejects any
          override key missing from default.yaml
        - The scheduler, validator, and validate step are not gated on
          run.enabled, so a plain arm and a dendrite arm differ only in
          the dendrites
    '''
    run = PerforatedRun()

    def setup_model(self) -> Optional[dict]:
        '''
        Load the model, then perforate it before anything binds to it

        Notes:
            - Runs first inside _setup_train, so the optimizer, EMA, and
              any DDP wrapper see the perforated model
        '''
        checkpoint = super().setup_model()
        if not self.run.enabled:
            return checkpoint
        self.model = perforate_detection_model(self.model, self.run)
        register_perforated_callbacks(self)
        return checkpoint

    def get_validator(self) -> PerforatedDetectionValidator:
        '''
        Return the validator that also reports mAP75
        '''
        return PerforatedDetectionValidator(
            self.test_loader,
            save_dir   = self.save_dir,
            args       = copy(self.args),
            _callbacks = self.callbacks,
        )

    def _setup_scheduler(self) -> None:
        '''
        Replace the epoch keyed LR ramp with a plateau on mAP

        Notes:
            - A dendrite arm restarts its schedule at every restructure,
              so on the stock ramp it never leaves lr0 while the plain
              arm decays to lr0 * lrf
            - The base call is kept for self.lf, which warmup reads, and
              it also resets every param group to lr0
            - Called again after a restructure, so each cycle starts with
              cleared plateau state
        '''
        super()._setup_scheduler()
        self.scheduler = PlateauScheduler(
            self.optimizer,
            mode           = 'max',
            factor         = plateau_factor,
            patience       = plateau_patience,
            threshold      = plateau_threshold,
            threshold_mode = 'rel',
            min_lr         = self.args.lr0 * self.args.lrf,
            cooldown       = 0,
        )

    def plateau_metric(self, fitness: Optional[float]) -> Optional[float]:
        '''
        Pick the score the plateau schedule steps on this epoch

        Notes:
            - In neuron mode this is the validation fitness. In dendrite
              mode the neurons are frozen and mAP is flat, so we use the
              mean best correlation per perforated layer instead
            - None until any layer has a score, which skips the step

        Signature:
            fitness (Optional[float]):
                - Validation fitness this epoch
        '''
        if not self.run.enabled or GPA.pai_tracker.member_vars['mode'] != 'p':
            return fitness
        scores = list(GPA.pai_tracker.get_current_pb_scores().values())
        if not scores:
            return None
        return sum(scores) / len(scores)

    def validate(self) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
        '''
        Validate, then step the plateau schedule

        Notes:
            - Runs before the PAI callback, so a restructure in the same
              epoch rebuilds the scheduler and discards this step
        '''
        metrics, fitness = super().validate()
        self.scheduler.step(self.plateau_metric(fitness))
        return metrics, fitness

    def save_model(self) -> bool:
        '''
        Write checkpoints that reload without any PerforatedAI state

        Notes:
            - A live perforated model reads GPA.pai_tracker on forward, so
              pickling it into best.pt breaks YOLO(best.pt) in a process
              that never called perforate_model
            - prepare_final_model deep copies before it strips, so the
              live model, tracker, and optimizer are untouched
            - self.model points at the same stripped copy, because the
              stock method resyncs EMA tensors by state dict key
        '''
        if not self.run.enabled:
            return super().save_model()
        live_model, live_ema = self.model, self.ema.ema
        self.model = self.ema.ema = UPA.prepare_final_model(live_ema)
        try:
            return super().save_model()
        finally:
            self.model, self.ema.ema = live_model, live_ema

    def final_eval(self) -> None:
        '''
        Reload best.pt for the final validation, as the plain arm does

        Notes:
            - The stock method fires on_fit_epoch_end after its final
              validation, which would let PAI restructure on the way out
        '''
        if self.run.enabled:
            self.callbacks['on_fit_epoch_end'].remove(
                perforated_fit_epoch_end
            )
        super().final_eval()

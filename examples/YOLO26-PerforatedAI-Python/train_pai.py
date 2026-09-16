# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Train YOLO26 with PerforatedAI dendrites on the whole model.

Every PerforatedAI key is a normal train argument, so the same names work on the CLI:

    yolo train model=yolo26n.pt data=coco8.yaml perforate=True

The values below are the defaults from ultralytics/cfg/default.yaml.
"""

from pathlib import Path

from ultralytics import YOLO

# perforate=False -> Train the plain model and none of the other keys are read.
# perforate_modules=[] -> Perforates every Conv in the model. 
# A list of named_modules paths restricts dendrites to those (i.e. ["model.23.one2one_cv2.0.0"])
# switch_mode="history" -> End a neuron cycle after n_epochs_to_switch validations without a gain. 
# switch_mode="fixed" -> End the neuron cycles every fixed_switch_num epochs
# which a short run needs to reach a dendrite cycle at all. 
# switch_mode="every_time" -> Debug Mode
# switch_mode="none" -> Never add dendrites
# pai_load_folder -> Resumes a dendrite run from a saved PAI system folder
# perforated_backpropagation=True-> Requires perforatedbp package and license
# False trains dendrites in open-source mode and ignores the keys after it.
pai_args = {
    "perforate": True,
    "perforate_modules": [],
    "testing_dendrite_capacity": False,
    "pai_load_folder": None,
    "pai_load_stage": "latest",
    "switch_mode": "history",
    "n_epochs_to_switch": 5,
    "history_lookback": 1,
    "initial_history_after_switches": 0,
    "fixed_switch_num": 15,
    "first_fixed_switch_num": 1,
    "reset_best_score_on_switch": False,
    "improvement_threshold": [0.001, 0.0001, 0.0],
    "improvement_threshold_raw": 1.0e-5,
    "max_dendrites": 5,
    "max_dendrite_tries": 2,
    "retain_all_dendrites": False,
    "candidate_weight_initialization_multiplier": 0.01,
    "candidate_weight_init_by_main": False,
    "find_best_lr": True,
    "dont_give_up_unless_learning_rate_lowered": True,
    "param_vals_setting": "update_epoch",
    "plateau_patience": 6,
    "plateau_factor": 0.1,
    "plateau_threshold": 0.001,
    "pai_verbose": False,
    "pai_extra_verbose": False,
    "pai_silent": False,
    "drawing_pai": True,
    "drawing_extra_graphs": True,
    "save_old_graph_scores": True,
    "test_saves": True,
    "using_safe_tensors": True,
    "perforated_backpropagation": True,
    "initial_correlation_batches": 20,
    "p_epochs_to_switch": 2,
    "cap_at_n": False,
    "pai_improvement_threshold": 0.1,
    "pai_improvement_threshold_raw": 1.0e-5,
}

# Ultralytics example arguments
train_args = {
    "data": str(Path(__file__).parent / "cityscapes-det.yaml"),
    "epochs": 2000,
    "batch": 16,
    "imgsz": 640,
    "optimizer": "AdamW",
    "lr0": 0.001,
    "lrf": 0.01,
    "warmup_bias_lr": 0.0,
    "weight_decay": 0.0,
    "name": "yolo26n_pai",
}

if __name__ == "__main__":
    YOLO("yolo26n.pt").train(**train_args, **pai_args)

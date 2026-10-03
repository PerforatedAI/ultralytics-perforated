# YOLO26 with PerforatedAI Dendrites

This guide adds [PerforatedAI](https://www.perforatedai.com/) artificial dendrites to an existing [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26) training setup. You keep your own dataset, config, and training script. Dendrite training is switched on with the `perforate=True` train argument, and every PerforatedAI setting is a normal train argument listed under "PerforatedAI settings" in `ultralytics/cfg/default.yaml`, so it works from the CLI, from `YOLO(...).train(...)`, and from a `cfg=` YAML. No custom trainer script is needed.

The Cityscapes examples are in: [CITYSCAPES_EXAMPLE.md](CITYSCAPES_EXAMPLE.md).

## Required

The steps below turn your existing YOLO26 training run into a Baseline and Perforated pair. `path/to/your_config.yaml` stands for your own config.

### 1. Install

This fork replaces the `ultralytics` package, so install it in editable mode from a clone of the `pai-yolo26` branch.

If you already have a clone of Ultralytics, add this fork as a remote. Commit or stash any local changes first.

```bash
cd path/to/ultralytics
git remote add perforated https://github.com/PerforatedAI/ultralytics-perforated.git
git fetch perforated
git checkout -b perforated-main perforated/pai-yolo26
pip install -e .
```

If you do not, clone the fork. Uninstall any PyPI `ultralytics` first, because it can shadow the editable install.

```bash
pip uninstall -y ultralytics
git clone -b pai-yolo26 https://github.com/PerforatedAI/ultralytics-perforated.git
cd ultralytics-perforated
pip install -e .
```

Then, in both cases, install PerforatedAI:

```bash
pip install perforatedai perforatedbp
```

Both packages and a `perforatedbp` license are required. The trainer imports `perforatedbp` whenever `perforate=True`, and dendrites currently always train with Perforated Backpropagation. 

Check the install without starting a run. The first command should print a path inside your clone, and the second should list `perforate: false`.

```bash
python -c "import ultralytics, perforatedai; print(ultralytics.__file__)"
yolo cfg | grep perforate
```

### 2. New datasets

Nothing has to be done for a new dataset. Keep your current dataset YAML and config. 

### 3. Train

Run your usual training command once without and once with `perforate=True`. Keep everything else the same, so the comparison shows only the effect of the dendrites.

```bash
# Baseline
yolo train cfg=path/to/your_config.yaml name=my_baseline

# Perforated
yolo train cfg=path/to/your_config.yaml name=my_pai perforate=True
```

If you train from Python, add `perforate=True` to your existing `train()` call. Callbacks you add with `model.add_callback(...)` and trainer subclasses still work, because the PerforatedAI code is in the base trainer.

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
model.train(cfg="path/to/your_config.yaml", name="my_pai", perforate=True)
```

### 4. Read the results

`<task>` below is the task folder of your run, for example `detect`.

- `runs/<task>/<name>/results.csv` has the per-epoch validation metrics. The Perforated run's score is flat during dendrite phases because the neurons are frozen.
- `runs/<task>/<name>/weights/best.pt` is the best epoch, with dendrites folded into a deep copy of the EMA model. Loading it needs `perforatedai` installed.
- The PerforatedAI folder `<name>_pai/` holds the score graphs, `*_best_arch_scores.csv` with the best score per dendrite count, and `*param_counts.csv` with the parameter count at each dendrite addition. Use those two CSVs to compare against your Baseline.

## Optional Details and Hyperparameters

### Settings the trainer overrides

With `perforate=True`, the trainer changes these settings for you. Your values for them will not apply:

- `patience` is set to 0 (PerforatedAI stops the run when the last dendrite does not improve the validation score).
- `channels_last` is set to `False`, because it currently breaks fused AdamW.
- The learning rate schedule (`cos_lr` or linear) is replaced by the PerforatedAI plateau scheduler. `lr0` is the start value and `lr0 * lrf` is the floor.
- `epochs` defaults to 400 instead of 100, because a Perforated run takes about 3 times as many epochs as its Baseline. PerforatedAI decides when to stop, so this is only an upper bound. Set `epochs` yourself to change it.
- `time` is not supported and raises an error.

### Recommended training settings

Our Cityscapes runs used these values, but these are not required to run.

```yaml
optimizer: AdamW
weight_decay: 0.0
warmup_bias_lr: 0.0
amp: False
close_mosaic: 0
compile: False
```

### PerforatedAI hyperparameters

The defaults are in `ultralytics/cfg/default.yaml`. These are the ones we changed:

```yaml
switch_mode: history
n_epochs_to_switch: 15 # Validations without improvement before dendrites are added
p_epochs_to_switch: 4 # Dendrite epochs without a correlation gain before switching back to neurons
max_dendrites: 3
max_dendrite_tries: 2
initial_correlation_batches: 50 # Must be less than the number of batches per epoch on your dataset
plateau_patience: 6 # Keep below n_epochs_to_switch
plateau_factor: 0.1
plateau_threshold: 0.001
```

If your dataset is much smaller than Cityscapes (2,975 training images), lower `initial_correlation_batches` so it stays below the batches per epoch.

### Other target modules

The default `perforate_modules` is `[model.23.one2one_cv2.0.0, model.23.one2one_cv2.1.0, model.23.one2one_cv2.2.0]`, the first Conv of the one-to-one box branch at each pyramid level. These paths assume the Detect head is layer 23, as in every YOLO26 detection model, so set `perforate_modules` yourself for other architectures or tasks. `perforate_modules: []` perforates every Conv in the model. To target other modules, print the Conv paths of your model and copy the ones you want:

```bash
python -c "from ultralytics import YOLO; [print(n) for n, m in YOLO('yolo26n.pt').model.named_modules() if type(m).__name__ == 'Conv']"
```

More target modules add more parameters per dendrite and more GPU memory during dendrite phases. Lower `batch` if you add a lot of dendrites.

### Check your setup

Before a full run, you can run a few epochs with `testing_dendrite_capacity=True`. In this debug mode, PerforatedAI adds a dendrite every epoch and keeps each one, so it shows quickly that dendrites attach to your modules and that your GPU memory is sufficient.

```bash
yolo train cfg=path/to/your_config.yaml perforate=True testing_dendrite_capacity=True epochs=6 name=pai_wiring_check
```

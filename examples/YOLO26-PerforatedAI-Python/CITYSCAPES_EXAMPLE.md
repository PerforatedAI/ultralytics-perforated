# YOLO26 with PerforatedAI Dendrites

This example trains [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26) with [PerforatedAI](https://www.perforatedai.com/) artificial dendrites on Cityscapes (set up as an 8-class object detection task). Each of the four YOLO26 sizes is fine-tuned twice from its COCO checkpoint, once as a Baseline and once Perforated, so the two runs differ only in whether dendrites are on. Dendrite training is switched on with the `perforate=True` train argument. Every PerforatedAI setting is a normal train argument, listed under "PerforatedAI settings" in `ultralytics/cfg/default.yaml`, so it works from the CLI, from `YOLO(...).train(...)`, and from a `cfg=` YAML.

## Dataset

Cityscapes is a segmentation benchmark: 5,000 finely annotated street scenes at 2048×1024 with pixel masks for 30 classes. The `cityscapes.yaml` that ships with Ultralytics is the 19-class semantic set built from those masks. For detection, this training script uses the standard mmdetection conversion: the 8 instance classes (person, rider, car, truck, bus, train, motorcycle, bicycle), a tight box drawn around each instance mask, and crowd/group regions dropped. We run it in the YOLO layout on the 2,975 training and 500 validation images, so results are broadly comparable to typical mmdetection-style baselines.

`prepare_cityscapes_det.py` reproduces the conversion without going through COCO JSON. It reads each `_gtFine_instanceIds.png`, takes instance ids at or above 24 as objects, decodes the class as `id // 1000`, drops ids under 1000 as crowd regions, boxes each mask, and writes YOLO labels in mmdetection's class order.

## Results

Cityscapes 8-class detection, best mAP50-95 on the 500 validation images. Each row is one Baseline run and one Perforated run trained from the same config in this folder, differing only in `perforate`. Perforated scores are reported at 2 dendrites, except YOLO26s where the run kept only 1 dendrite.

| Model   | Baseline mAP50-95 | Perforated mAP50-95 | Dendrites Added | Baseline Params | Perforated Params | Param Increase | Gain (Points) |
| ------- | ----------------- | ------------------- | --------------- | --------------- | ----------------- | -------------- | ------------- |
| YOLO26n | 0.2620            | 0.2732              | 2               | 2,506,920       | 2,636,136         | +5.2%          | +1.1          |
| YOLO26s | 0.3181            | 0.3292              | 1               | 9,954,056       | 10,212,296        | +2.6%          | +1.1          |
| YOLO26m | 0.3597            | 0.3779              | 2               | 21,785,224      | 23,260,552        | +6.8%          | +1.8          |
| YOLO26l | 0.3924            | 0.4102              | 2               | 26,188,680      | 27,664,008        | +5.6%          | +1.8          |

The Baseline rows trace the normal size-versus-accuracy curve, where going from n to s is a 300% parameter increase for 5.6 points. Perforated YOLO26n gets 1.1 of those points for a 5.2% parameter increase, closing 20% of the accuracy gap to YOLO26s with 1.7% of the parameters that gap costs. The lift holds at every size, so it is a shift of the whole Pareto frontier.

## Files

- `train_pai.py`: Minimal example that loads the perforation config from a YAML file.
- `yolo26n_cityscapes_pai.yaml`: Configuration YAML including Ultralytics hyperparameters and Perforated hyperparameters.
- `cityscapes-det.yaml`: The 8-class Cityscapes detection set (not the same as the 19-class semantic cityscapes.yaml that Ultralytics has).
- `prepare_cityscapes_det.py`: Downloads Cityscapes and converts the instance masks to YOLO labels for `cityscapes-det.yaml`.

## Walkthrough

All commands run from the repository root.

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

`perforatedai` and `perforatedbp` are not dependencies of the package, so install them separately as above. Both packages and a `perforatedbp` license are required. The trainer imports `perforatedbp` whenever `perforate=True`, and dendrites always train with Perforated Backpropagation: `perforated_backpropagation=False` does not currently switch it off.

Check the install without starting a run. The first command should print a path inside your clone, and the second should list `perforate: false`.

```bash
python -c "import ultralytics, perforatedai; print(ultralytics.__file__)"
yolo cfg | grep perforate
```

To use your own dataset, config, or training script, see [README.md](README.md).

### 2. Build Cityscapes

Cityscapes needs a free account at [cityscapes-dataset.com](https://www.cityscapes-dataset.com/). The script downloads `leftImg8bit` and `gtFine`, then writes the 8-class detection labels.

```bash
python examples/YOLO26-PerforatedAI-Python/prepare_cityscapes_det.py --download \
    --cityscapes-root datasets/cityscapes-raw \
    --out-root datasets/cityscapes-det
```

`--out-root` must match the `path` key in `cityscapes-det.yaml`, resolved against the Ultralytics datasets directory.

### 3. Perforate and Train

```bash
# YOLO26n (Perforated)
yolo train cfg=examples/YOLO26-PerforatedAI-Python/yolo26n_cityscapes_pai.yaml name=yolo26n_cityscapes_pai perforate=True

# YOLO26n (Baseline)
yolo train cfg=examples/YOLO26-PerforatedAI-Python/yolo26n_cityscapes_pai.yaml name=yolo26n_cityscapes_baseline

# YOLO26s (Baseline) (Next size for parameter comparison)
yolo train cfg=examples/YOLO26-PerforatedAI-Python/yolo26n_cityscapes_pai.yaml model=yolo26s.pt batch=32 name=yolo26s_cityscapes_baseline
```

The Baseline run stops on `patience`, meaning it stops once more epochs no longer improve the validation score. The Perforated run stops when PerforatedAI reports that the last dendrite did not improve the score. It usually takes 75 - 150 epochs for the Baseline runs to complete and 250 - 450 epochs for the Perforated runs to complete.

### 4. Read the results

- `runs/detect/<name>/results.csv` has the per-epoch mAP50-95. The Perforated run's mAP is flat during dendrite phases because the neurons are frozen.
- `runs/detect/<name>/weights/best.pt` is the best epoch, with dendrites folded into a deep copy of the EMA model.
- The PerforatedAI folder `<name>_pai/` holds the score graphs, `*_best_arch_scores.csv` with the best score per dendrite count, and `*param_counts.csv` with the parameter count at each switch. Those two CSVs are the source of the params and mAP columns in the results table above.

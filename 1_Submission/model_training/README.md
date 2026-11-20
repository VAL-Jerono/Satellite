# Model Training Code and Serialized Artifacts

This directory contains the modeling code, configuration, and serialized artifacts referenced by the capstone submission. Artifact presence does not independently establish that a given reported metric can be reproduced; the raw observation table, patch archive, and split manifests are not included in this package.

### `code/`

- `train.py`: training loop with spatial GroupKFold cross-validation and MLflow logging; includes `FocalLossWithSmoothing` (label smoothing and class-frequency weighting).
- `model.py`: SpectralChannelAttention EfficientNet-B0 CNN definition for 10-channel patches.
- `dataset.py`: PyTorch dataset loader for 10-channel Sentinel-2 patches, including class-weight helpers.

### `artifacts/`

- `lgbm_zone_*.pkl`: five serialized zone-stratified LightGBM classifiers (highland, rift, lake, coastal, ASAL).
- `land_cover.onnx`: exported ONNX graph for CPU-oriented inference (approximately 387 KB as packaged).
- `best_spectral_cnn.pt`: PyTorch checkpoint for the spectral CNN.
- `stacking_meta_learner.pkl`: serialized stacking meta-learner combining model probability outputs.

### `configs/`

- `lgbm_tuned_params.yaml`: tuned LightGBM hyperparameters by zone.
- `config.yaml`: shared model and training constants.
- `regions.yaml`: study-region and bioclimatic-zone configuration.

## Reproduction notes

Training from scratch requires the observation table, patch archive, split assignments, and an environment with the project dependencies. These inputs are not all present here, and notebook paths may refer to the original author's machine. The notebook's reported metrics are therefore documented as notebook-reported rather than independently recomputed.

The API service can generate a placeholder ONNX graph if `land_cover.onnx` is missing. For any operational use, verify the artifact checksum and fail closed rather than substituting a placeholder model.

# Model Training Code & Serialized Artifacts

This folder contains the complete modeling codebase and versioned production artifacts:

### `code/`
- `train.py`: LightGBM training loop with spatial GroupKFold cross-validation and MLflow logging.
- `model.py`: SpectralChannelAttention EfficientNet-B0 CNN PyTorch definition.
- `dataset.py`: PyTorch dataset loader for 10-channel Sentinel-2 spatial patches.
- `loss.py`: Label-Smoothing Focal Loss implementation.

### `artifacts/`
- `lgbm_zone_*.pkl`: 5 serialized Zone-Stratified LightGBM specialist binaries.
- `land_cover.onnx`: Exported ONNX graph for high-throughput CPU inference.
- `best_spectral_cnn.pt`: PyTorch model checkpoint.
- `stacking_meta_learner.pkl`: Calibrated Logistic Regression ensemble combiner.

### `configs/`
- `lgbm_tuned_params.yaml`: Optimal Optuna Bayesian hyperparameters per zone.
- `config.yaml`: Global model constants.

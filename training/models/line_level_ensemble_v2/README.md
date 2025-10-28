# Model Artifacts - Line Level Ensemble v2

## Overview
Advanced trained models with hyperparameter optimization and improved performance metrics.

## Files
- `dt_model_v2.joblib` - Optimized Decision Tree classifier
- `mlp_model_v2.joblib` - Tuned Multi-Layer Perceptron neural network
- `sgd_model_v2.joblib` - Optimized Stochastic Gradient Descent classifier
- `scaler_v2.joblib` - Feature scaling transformer
- `line_level_ensemble_v2_results.joblib` - Complete optimized ensemble
- `training_metadata_v2.json` - Training metrics and hyperparameters



## Usage
Current production models used by the anomaly detection service:
```python
import joblib
ensemble_model = joblib.load('line_level_ensemble_v2_results.joblib')
```



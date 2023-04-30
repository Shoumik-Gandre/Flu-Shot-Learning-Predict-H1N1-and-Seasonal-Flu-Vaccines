from xgb_module.hyperparam_tune import tune
from decouple import config

# tune(
#     config('TRAINING_FEATURES_PATH'),
#     config('TRAINING_LABELS_PATH'),
#     'h1n1_vaccine',
#     500
# )

tune(
    config('TRAINING_FEATURES_PATH'),
    config('TRAINING_LABELS_PATH'),
    'seasonal_vaccine',
    100
)
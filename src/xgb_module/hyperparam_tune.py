"""
Optuna example that demonstrates a pruner for XGBoost.cv.
In this example, we optimize the validation auc of cancer detection using XGBoost.
We optimize both the choice of booster model and their hyperparameters. Throughout
training of models, a pruner observes intermediate results and stop unpromising trials.
You can run this example as follows:
    $ python xgboost_cv_integration.py
"""

from typing import Literal
import optuna
from optuna.pruners import MedianPruner
import pandas as pd
import xgboost as xgb
from pathlib import Path
from .preprocessing import preprocessor


def objective(trial, X, y):
    dtrain = xgb.DMatrix(X, label=y)

    param = {
        "tree_method": 'gpu_hist',
        "verbosity": 0,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
    }

    param['n_estimators'] = trial.suggest_categorical('n_estimators', [2, 4, 16, 32, 64, 128, 256, 512, 1024, 2048])
    
    if param["booster"] == "gbtree" or param["booster"] == "dart":
        param["max_depth"] = trial.suggest_int("max_depth", 1, 9)
        param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "test-auc")
    history = xgb.cv(param, dtrain, callbacks=[pruning_callback], nfold=10, stratified=True, metrics=['auc'])

    mean_auc = history["test-auc-mean"].values[-1]
    return mean_auc


def tune(
        features_path,
        labels_path,
        output: Literal['seasonal_vaccine', 'h1n1_vaccine'],
        n_trials: int=100):

    features_df         = pd.read_csv(features_path,   index_col='respondent_id')
    labels_df           = pd.read_csv(labels_path,     index_col='respondent_id')

    X = preprocessor.fit_transform(features_df, labels_df['h1n1_vaccine'])
    y = labels_df[output]

    pruner = MedianPruner(n_warmup_steps=100)
    study = optuna.create_study(pruner=pruner, direction="maximize")
    study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

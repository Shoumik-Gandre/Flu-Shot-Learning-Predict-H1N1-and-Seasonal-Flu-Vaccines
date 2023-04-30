from catboost import Pool, cv
from constants import *
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import optuna


def objective(trial: optuna.Trial, pool: Pool) -> float:
    params = {
        'iterations': trial.suggest_categorical('iterations', [100, 200, 300, 500, 1000, 1200, 1500]),
        'learning_rate': trial.suggest_float("learning_rate", 0.001, 0.3),
        'random_strength': trial.suggest_int("random_strength", 1, 10),
        'bagging_temperature': trial.suggest_int("bagging_temperature", 0, 10),
        'max_bin': trial.suggest_categorical('max_bin', [4, 5, 6, 8, 10, 20, 30]),
        'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
        'min_data_in_leaf': trial.suggest_int("min_data_in_leaf", 1, 10),
        'od_type': "Iter",
        'od_wait': 100,
        "depth": trial.suggest_int("max_depth", 2, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 100),
        'one_hot_max_size': trial.suggest_categorical('one_hot_max_size', [5, 10, 12, 100, 500, 1024]),
        'custom_metric': ['AUC'],
        "loss_function": "Logloss",
        'auto_class_weights': trial.suggest_categorical('auto_class_weights', ['Balanced', 'SqrtBalanced']),
    }

    scores = cv(
        pool=pool,
        params=params,
        fold_count=5,
        early_stopping_rounds=10,
        plot=False,
        logging_level='Silent'
    )

    return scores['test-AUC-mean'].max()


if __name__ == '__main__':
    # Load Data
    features_df = pd.read_csv(TRAINING_FEATURES_PATH,
                              index_col='respondent_id')
    labels_df = pd.read_csv(TRAINING_LABELS_PATH, index_col='respondent_id')
    test_features_df = pd.read_csv(
        TEST_FEATURES_PATH, index_col="respondent_id")

    # Column Types
    num_cols = features_df.select_dtypes('number').columns
    cat_cols = [
        'race',
        'sex',
        'marital_status',
        'rent_or_own',
        'hhs_geo_region',
        'census_msa',
        'employment_industry',
        'employment_occupation'
    ]
    ord_cols = [
        'age_group',
        'education',
        'income_poverty',
        'employment_status'
    ]

    # Impute columns
    for col in (cat_cols + ord_cols):
        features_df[col] = features_df[col].fillna(value='None')
        test_features_df[col] = test_features_df[col].fillna(value='None')

    for col in num_cols:
        features_df[col] = features_df[col].fillna(value=-1)
        test_features_df[col] = test_features_df[col].fillna(value=-1)

    # Get Categorical Columns
    cat_cols_indices = np.where(features_df.dtypes != float)[0]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        features_df,
        labels_df,
        test_size=0.3,
        random_state=68
    )

    # Create Catboost Pool
    seasonal_vaccine_pool = Pool(
        data=X_train,
        label=y_train.seasonal_vaccine,
        cat_features=cat_cols_indices
    )

    sampler = optuna.samplers.TPESampler(seed=68)

    study_seasonal = optuna.create_study(
        study_name='seasonal-study',
        storage='sqlite:///seasonal.db',
        load_if_exists=True,
        direction="maximize",
        sampler=sampler
    )

    study_seasonal.optimize(
        lambda trial: objective(trial, seasonal_vaccine_pool),
        n_trials=100)

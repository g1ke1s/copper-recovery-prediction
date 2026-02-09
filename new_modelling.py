import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from scipy.stats import shapiro, normaltest
import joblib
import pickle
from xgboost import XGBRegressor

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from new_preprocessing_logic import clean, get_train_test, impute_inter_fbfill, handle_outliers
from config import feats
import yaml

with open("features.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

TARGET = cfg["target"]
RANDOM_SEED = 42
RECOVERY_MODEL_PATH = cfg["model_path"]
DATA_PATH = r"C:\Users\Kenessary.Garifulla\Desktop\recov_train\data_master_till_csv.xls"

datamaster = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)

data_master = datamaster[feats+[TARGET]]

df = clean(data_master, timefrom='2020-12-01')

train, test_june, test_july, test_summer = get_train_test(df)

# Fill in N/A
for feat in feats:
    train = impute_inter_fbfill(train, feat, method='time', verbose=False)

print(f"[impute_inter_fbfill] finished.")

# Replacing outliers with daily median 
train = handle_outliers(
    train,
    iqr_multiplier=10,
    verbose=False,
)

# Divide into sets: Train, June set, July set, both June/July set
X_train = train.drop(columns=TARGET)
y_train = train[TARGET]

X_june = test_june.drop(columns=TARGET)
y_june = test_june[TARGET]

X_july = test_july.drop(columns=TARGET)
y_july = test_july[TARGET]


# Fill in N/A for test sets
for feat in feats:
    X_june = impute_inter_fbfill(X_june, feat, method='time', verbose=False)
    X_july = impute_inter_fbfill(X_july, feat, method='time', verbose=False)

X_summer = pd.concat([X_june, X_july])
y_summer = pd.concat([y_june, y_july])

# Model building

params = {
    'max_depth': 3,
    'learning_rate': 0.0655, # 0. 07655 0655
    'n_estimators': 590, 
    'subsample': 0.10003376113481072, 
    'colsample_bytree': 0.6352123526926039, 
    'gamma': 4.019055148141487, 
    'min_child_weight': 7, 
    'reg_alpha': 1.5923686922190168, 
    'reg_lambda': 3.2506547257522196, 
    'max_delta_step': 10,
    'random_state': RANDOM_SEED,
}

xgbm = XGBRegressor(**params)

xgbm.fit(X_train, y_train)

# June set
y_pred_xgb = xgbm.predict(X_june)

mape_xgb = mean_absolute_percentage_error(y_june, y_pred_xgb)
r2_xgb = r2_score(y_june, y_pred_xgb)

print(f"[June] MAPE on Test: {mape_xgb:.4f}")
print(f"[June] R2 on Test: {r2_xgb:.4f}")

# July set
y_pred_xgb2 = xgbm.predict(X_july)

mape_xgb2 = mean_absolute_percentage_error(y_july, y_pred_xgb2)
r2_xgb2 = r2_score(y_july, y_pred_xgb2)

print(f"[July] MAPE on Test: {mape_xgb2:.4f}")
print(f"[July] R2 on Test: {r2_xgb2:.4f}")

# June & July set
y_pred_xgb3 = xgbm.predict(X_summer)

mape_xgb3 = mean_absolute_percentage_error(y_summer, y_pred_xgb3)
r2_xgb3 = r2_score(y_summer, y_pred_xgb3)

print(f"[June & July] MAPE on Test: {mape_xgb3:.4f}")
print(f"[June & July] R2 on Test: {r2_xgb3:.4f}")

with open(MODEL_PATH, 'wb') as f:
    pickle.dump(xgbm, f)
print(f"[Model Saved] XGBoost model saved to {MODEL_PATH} using pickle")

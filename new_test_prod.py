from copper_oai_2022.pipelines.data_modeling.utils import IdentityTransformer
from optimus_core.core.transformers.select import SelectColumns
from copper_oai_2022.pipelines.data_modeling.online_modeling.utils import RecoveryModel

from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from xgboost import XGBRegressor
import pandas as pd
import joblib
import pickle

from config import feats
from new_preprocessing_logic import clean, get_train_test, impute_inter_fbfill, handle_outliers
import yaml

with open("features.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

TARGET = cfg["target"]
RECOVERY_MODEL_PATH = cfg["model_path"]
cntrl = cfg["controls"]
sts = cfg["stats"]

with open("xgb_model.pkl", 'rb') as f:
    test_model = pickle.load(f)


model = RecoveryModel(
    pipeline = Pipeline(steps=[
        ('select_columns', SelectColumns(items=feats)),
        ('identity_transformer', IdentityTransformer()),
        ('estimator', test_model)
    ]),
    features=feats,
    target=TARGET,
    controls=cntrl,
    states=sts
)

if __name__ == '__main__':

    data_master= pd.read_csv(r"C:\Users\Kenessary.Garifulla\Desktop\recov_train\data_master_till_csv.xls", index_col=0, parse_dates=True)[feats+[TARGET]]
 
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

    model.pipeline.fit(
        X_train, 
        y_train,
        estimator__eval_set=[(X_june, y_june)],
        estimator__eval_metric="rmse",
        # estimator__early_stopping_rounds=4,
    )

    y_pred_train = model.predict(X_train)

    print(f"[Train] MAPE on Test: {mean_absolute_percentage_error(y_train, y_pred_train):.4f}")
    print(f"[Train] R2 on Test: {r2_score(y_train, y_pred_train):.4f}")
    
    y_pred_june = model.predict(X_june)

    print(f"[June] MAPE on Test: {mean_absolute_percentage_error(y_june, y_pred_june):.4f}")
    print(f"[June] R2 on Test: {r2_score(y_june, y_pred_june):.4f}")

    y_pred_july = model.predict(X_july)

    print(f"[July] MAPE on Test: {mean_absolute_percentage_error(y_july, y_pred_july):.4f}")
    print(f"[July] R2 on Test: {r2_score(y_july, y_pred_july):.4f}")

    y_pred_summer = model.predict(X_summer)

    print(f"[June&July] MAPE on Test: {mean_absolute_percentage_error(y_summer, y_pred_summer):.4f}")
    print(f"[June&July] R2 on Test: {r2_score(y_summer, y_pred_summer):.4f}")

    print(f"[Model] {model}")
    print(f"[HIGHEST_PROTOCOL] {pickle.HIGHEST_PROTOCOL}")

    with open(RECOVERY_MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print(f"[Model Saved] RecoveryModel saved to {RECOVERY_MODEL_PATH} using pickle")

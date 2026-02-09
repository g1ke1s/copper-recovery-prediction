import pandas as pd
import numpy as np
from scipy.stats import shapiro, normaltest
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype
import yaml

with open("features.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
    
TARGET = cfg["target"]

def clean(
    data: pd.DataFrame,
    timefrom: str,
    iqr_mult: float = 1.5,
    outlier_method: str = "iqr",
) -> pd.DataFrame:
    
    data = data.loc[timefrom:].drop_duplicates().copy()
    data['s_per_cu'] = [i if i >= 0.5 else 1 / i for i in data['s_per_cu'].values]
    data = data[data[TARGET].notna()]
    
    if outlier_method == "iqr":
        Q1 = data[TARGET].quantile(0.25)
        Q3 = data[TARGET].quantile(0.75)
        IQR = Q3 - Q1
        lb, ub = Q1 - iqr_mult * IQR, Q3 + iqr_mult * IQR
        data = data[(data[TARGET] >= lb) & (data[TARGET] <= ub)]

    print(f"[clean] Finished cleaning. Final shape: {data.shape}")
    return data

def get_train_test(df_copy):
    
    train = df_copy.loc[:'2025-05-31']
    test_june = df_copy.loc['2025-06-01':'2025-06-30']
    test_july = df_copy.loc['2025-07-01':]
    test_summer = df_copy.loc['2025-06-01':]

    print(f"Training data proportion: {round(len(train)/len(df_copy), 3)}, shape: {train.shape}")
    print(f"June data proportion: {round(len(test_june)/len(df_copy), 3)}, shape: {test_june.shape}")
    print(f"Jule data proportion: {round(len(test_july)/len(df_copy), 3)}, shape: {test_july.shape}")

    return train, test_june, test_july, test_summer

def impute_inter_fbfill(
    df,
    col,
    method='time',
    limit=None,
    verbose=False,
):
    df_copy = df.copy()
    series_orig = df_copy[col].copy()

    if not series_orig.isna().any():
        if verbose:
            print(f"[impute_inter_fbfill] No missing values in '{col}', skipping.")
        return df_copy

    if col.startswith("assay_"):
        df_copy[col] = df_copy[col].ffill().bfill()
        if verbose:
            print(f"[impute_inter_fbfill] '{col}' filled using forward/backward fill.")
    else:
        df_copy[col] = df_copy[col].interpolate(
            method=method,
            limit=limit,
            limit_direction='both'
        )
        if verbose:
            print(f"[impute_inter_fbfill] '{col}' interpolated using method='{method}'.")

    return df_copy

def handle_outliers(
    df,
    method="day_median",
    std_thresh=3,
    iqr_multiplier=1.5,
    verbose=True,
):
    df = df.copy()
    df["__date__"] = df.index.normalize()

    numeric_cols = [
        col for col in df.columns
        if is_numeric_dtype(df[col]) and col != "__date__"
    ]

    for col in numeric_cols:
        series = df[col]

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr

        outlier_mask = (series < lower) | (series > upper)
        n_outliers = outlier_mask.sum()

        if n_outliers > 0 and method == "day_median":
            replacement = df.groupby("__date__")[col].transform("median")
            df.loc[outlier_mask, col] = replacement[outlier_mask]

    df.drop(columns="__date__", inplace=True)
    print(f"[handle_outliers] Outlier handling completed.")
    return df



def handle_outliers(
    df,
    method="day_median",
    std_thresh=3,
    iqr_multiplier=1.5,
    verbose=True,
):
    df = df.copy()
    df["__date__"] = df.index.normalize()

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != "__date__"]

    for col in numeric_cols:
        series = df[col]

        try:
            if len(series) < 5000:
                stat, p = shapiro(series)
                test_name = "Shapiro-Wilk"
            else:
                stat, p = normaltest(series)
                test_name = "D’Agostino-Pearson"
        except Exception as e:
            if verbose:
                print(f"Column '{col}': Normality test failed: {e}")
            continue

        is_normal = p > 0.05
        outlier_rule = "std" if is_normal else "iqr"
        if verbose:
            print(f"Column '{col}': {'Normal' if is_normal else 'Non-normal'} "
                  f"(p={p:.4f}, test={test_name}) ⇒ using {outlier_rule.upper()} rule")

        if outlier_rule == "std":
            mean_val = series.mean()
            std_val = series.std()
            lower = mean_val - std_thresh * std_val
            upper = mean_val + std_thresh * std_val
        else:
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - iqr_multiplier * iqr
            upper = q3 + iqr_multiplier * iqr

        outlier_mask = (series < lower) | (series > upper)
        n_outliers = outlier_mask.sum()

        if n_outliers > 0 and method == "day_median":
            replacement = df.groupby("__date__")[col].transform("median")
            df.loc[outlier_mask, col] = replacement[outlier_mask]

    df.drop(columns="__date__", inplace=True)
    print(f"[handle_outliers] Outlier handling completed.")
    return df


if __name__ == '__main__':
    df = clean(pd.read_parquet('data_master_till_.parquet'), timefrom='2020-12-01')
    print(df.shape)
    print('OKAY!')
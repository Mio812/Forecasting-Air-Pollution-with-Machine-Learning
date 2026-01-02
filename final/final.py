import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from itertools import cycle
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.ensemble import IsolationForest
import xgboost as xgb
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib

matplotlib.use('Agg')
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 6)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


OUTPUT_DIR = ensure_dir('output')
REG_OUTPUT_DIR = ensure_dir(os.path.join(OUTPUT_DIR, 'regression'))
CLS_OUTPUT_DIR = ensure_dir(os.path.join(OUTPUT_DIR, 'classification'))
REG_DATA_DIR = ensure_dir(os.path.join(REG_OUTPUT_DIR, 'data'))
REG_PLOT_DIR = ensure_dir(os.path.join(REG_OUTPUT_DIR, 'plots'))
EDA_DIR = ensure_dir(os.path.join(REG_PLOT_DIR, 'eda'))
DIAG_DIR = ensure_dir(os.path.join(REG_PLOT_DIR, 'diagnostics'))
REG_IMP_DIR = ensure_dir(os.path.join(REG_PLOT_DIR, 'feature_importance'))
CLS_DATA_DIR = ensure_dir(os.path.join(CLS_OUTPUT_DIR, 'data'))
CLS_PLOT_DIR = ensure_dir(os.path.join(CLS_OUTPUT_DIR, 'plots'))
CM_DIR = ensure_dir(os.path.join(CLS_PLOT_DIR, "confusion_matrices"))
ROC_DIR = ensure_dir(os.path.join(CLS_PLOT_DIR, "roc_curves"))
PR_DIR = ensure_dir(os.path.join(CLS_PLOT_DIR, "pr_curves"))  # Added for PR plots
BAR_DIR = ensure_dir(os.path.join(CLS_PLOT_DIR, "summary_bars"))
CLS_IMP_DIR = ensure_dir(os.path.join(CLS_PLOT_DIR, "feature_importance"))
COEF_DIR = ensure_dir(os.path.join(CLS_PLOT_DIR, "logreg_coefficients"))

ALL_POLLUTANTS_RAW = ['CO(GT)', 'NMHC(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)']
ALL_POLLUTANTS_CLEAN = [p.replace('.', '_').replace('(', '_').replace(')', '') for p in ALL_POLLUTANTS_RAW]
CLASS_NAMES = ["Low", "Medium", "High"]

def clean_time_interpolate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()

    df['Time'] = df['Time'].astype(str).str.replace('.', ':', regex=False)

    df['Datetime'] = pd.to_datetime(
        df['Date'].astype(str) + ' ' + df['Time'].astype(str),
        dayfirst=True,
        errors='coerce'
    )

    df = df.dropna(subset=['Datetime']).drop(columns=['Date', 'Time'])

    df = df.sort_values('Datetime').set_index('Datetime')
    df.index.name = 'Datetime'

    df = df.replace(-200, np.nan)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    df = df.interpolate(method='time')

    return df


def clean_ffill_bfill_like_other(df: pd.DataFrame, keep_original_names: bool = True) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()

    if {'Date', 'Time'}.issubset(df.columns):
        try:
            df['DateTime'] = pd.to_datetime(
                df['Date'].astype(str) + ' ' + df['Time'].astype(str),
                format='%d/%m/%Y %H.%M.%S',
                errors='coerce'
            )
        except Exception:
            df['Time'] = df['Time'].astype(str).str.replace('.', ':', regex=False)
            df['DateTime'] = pd.to_datetime(
                df['Date'].astype(str) + ' ' + df['Time'].astype(str),
                dayfirst=True,
                errors='coerce'
            )
        df = df.dropna(subset=['DateTime']).drop(columns=['Date', 'Time'])
    else:
        raise ValueError("原始数据缺少 Date / Time 列，无法解析时间。")

    df = df.set_index('DateTime').sort_index()
    df.index.name = 'Datetime'

    if not keep_original_names:
        df.columns = [c.replace('.', '_').replace('(', '_').replace(')', '') for c in df.columns]

    df = df.replace([-200, -200.0], np.nan)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    threshold = int(len(df.columns) * 0.5)
    df = df.dropna(thresh=threshold)

    df = df.ffill().bfill()

    return df

def load_prepared_data(
    raw_path="data/AirQualityUCI.csv",
    cleaning_strategy="ffill",
    save_cleaned=True,
    cleaned_output_path="cleaned_air_quality.csv",
):

    try:
        raw = pd.read_csv(raw_path, sep=';', decimal=',').dropna(axis=1, how='all')
    except FileNotFoundError:
        print(f"[Error] file not found: {raw_path}")
        return None

    cleaning_strategy = cleaning_strategy.lower()
    if cleaning_strategy == "interpolate":
        df_clean = clean_time_interpolate(raw)
        tag = "clean_interpolate"
    elif cleaning_strategy == "ffill":
        df_clean = clean_ffill_bfill_like_other(raw, keep_original_names=True)
        tag = "clean_ffill"
    else:
        raise ValueError("cleaning_strategy only 'interpolate' or 'ffill'")

    if save_cleaned:
        cleaned_dir = os.path.dirname(cleaned_output_path)
        if cleaned_dir:
            ensure_dir(cleaned_dir)
        df_clean.to_csv(cleaned_output_path)
        print(f"Cleaned dataset saved to: {cleaned_output_path}  (strategy={tag})")

    df = df_clean.copy()
    df.columns = [col.replace('.', '_').replace('(', '_').replace(')', '') for col in df.columns]

    df['IsWeekend'] = (df.index.dayofweek >= 5).astype(int)
    df['Hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['Month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df.index.month / 12)

    if 'NMHC_GT' in df.columns and df['NMHC_GT'].count() / len(df) < 0.2:
        df = df.drop(columns=['NMHC_GT'])

    df = df.dropna()

    return df


def run_exploratory_analysis(df):
    pollutants_in_df = [p for p in ALL_POLLUTANTS_CLEAN if p in df.columns]
    meteo_features = ['T', 'RH', 'AH']
    meteo_features_in_df = [f for f in meteo_features if f in df.columns]

    if pollutants_in_df:
        fig, axes = plt.subplots(len(pollutants_in_df), 1, figsize=(20, 2.5 * len(pollutants_in_df)), sharex=True)
        if len(pollutants_in_df) == 1: axes = [axes]
        for i, pol in enumerate(pollutants_in_df):
            df[pol].plot(ax=axes[i], title=f"{pol} Over Time")
            axes[i].set_ylabel("Concentration")
        plt.xlabel("Datetime")
        plt.tight_layout()
        plt.savefig(os.path.join(EDA_DIR, 'eda_hourly_timeseries_pollutants.png'), dpi=300, bbox_inches='tight')
        plt.close()

    if meteo_features_in_df:
        fig, axes = plt.subplots(len(meteo_features_in_df), 1, figsize=(20, 8), sharex=True)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        for i, (feat, color) in enumerate(zip(meteo_features_in_df, colors)):
            df[feat].plot(ax=axes[i], label=feat, color=color)
            axes[i].legend(loc='upper left')
        plt.xlabel("Datetime")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.suptitle("Environmental Features Over Time (Hourly)", fontsize=16)
        plt.savefig(os.path.join(EDA_DIR, 'eda_hourly_timeseries_environmental.png'), dpi=300, bbox_inches='tight')
        plt.close()

    df_daily = df.resample('D').mean()

    if pollutants_in_df:
        fig, axes = plt.subplots(len(pollutants_in_df), 1, figsize=(20, 12), sharex=True)
        if len(pollutants_in_df) == 1: axes = [axes]
        for i, pol in enumerate(pollutants_in_df):
            df_daily[pol].plot(ax=axes[i], label=pol)
            axes[i].legend(loc='upper right')
        plt.xlabel("Datetime")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.suptitle("Pollutant Concentrations (Daily Average)", fontsize=16)
        plt.savefig(os.path.join(EDA_DIR, 'eda_daily_avg_pollutants.png'), dpi=300, bbox_inches='tight')
        plt.close()

    if meteo_features_in_df:
        fig, axes = plt.subplots(len(meteo_features_in_df), 1, figsize=(20, 8), sharex=True)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        for i, (feat, color) in enumerate(zip(meteo_features_in_df, colors)):
            df_daily[feat].plot(ax=axes[i], label=feat, color=color)
            axes[i].legend(loc='upper left')
        plt.xlabel("Datetime")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.suptitle("Environmental Features (Daily Average)", fontsize=16)
        plt.savefig(os.path.join(EDA_DIR, 'eda_daily_avg_environmental.png'), dpi=300, bbox_inches='tight')
        plt.close()

    df_hourly_avg = df.groupby(df.index.hour).mean()

    if pollutants_in_df:
        fig, axes = plt.subplots(len(pollutants_in_df), 1, figsize=(20, 12), sharex=True)
        if len(pollutants_in_df) == 1: axes = [axes]
        for i, pol in enumerate(pollutants_in_df):
            df_hourly_avg[pol].plot(ax=axes[i], title=f"Average Intraday Pattern: {pol}")
            axes[i].set_ylabel("Concentration")
        axes[-1].set_xlabel("Hour of Day")
        axes[-1].set_xticks(range(0, 24, 2))
        plt.tight_layout()
        plt.savefig(os.path.join(EDA_DIR, 'eda_intraday_pattern_pollutants.png'), dpi=300, bbox_inches='tight')
        plt.close()

    if meteo_features_in_df:
        fig, axes = plt.subplots(len(meteo_features_in_df), 1, figsize=(20, 8), sharex=True)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        for i, (feat, color) in enumerate(zip(meteo_features_in_df, colors)):
            df_hourly_avg[feat].plot(ax=axes[i], label=feat, color=color)
            axes[i].legend(loc='upper left')
        axes[-1].set_xlabel("Hour of Day")
        axes[-1].set_xticks(range(0, 24))
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.suptitle("Average Intraday Pattern of Environmental Features (by Hour)", fontsize=16)
        plt.savefig(os.path.join(EDA_DIR, 'eda_intraday_pattern_environmental.png'), dpi=300, bbox_inches='tight')
        plt.close()

    key_features = pollutants_in_df + meteo_features_in_df
    if key_features:
        plt.figure(figsize=(14, 12))
        corr = df[key_features].corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True,
                    cbar_kws={'label': 'Correlation Coefficient'})
        plt.title("Correlation Matrix: Pollutants and Environmental Features", fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(EDA_DIR, 'eda_correlation_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()

    if pollutants_in_df:
        fig, axes = plt.subplots(2, 3, figsize=(20, 10))
        axes = axes.flatten()
        for i, pol in enumerate(pollutants_in_df):
            if i < len(axes):
                df[pol].hist(bins=50, ax=axes[i], edgecolor='black')
                axes[i].set_title(f"Distribution of {pol}")
                axes[i].set_xlabel("Concentration")
                axes[i].set_ylabel("Frequency")
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        plt.savefig(os.path.join(EDA_DIR, 'eda_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()


def run_anomaly_detection(df, features_to_check, output_path):
    X = df[features_to_check]
    iso_forest = IsolationForest(contamination='auto', random_state=42, n_jobs=-1)
    predictions = iso_forest.fit_predict(X)
    df['anomaly'] = predictions
    anomalies = df[df['anomaly'] == -1]
    if not anomalies.empty:
        anomalies_to_save = anomalies.drop(columns=['anomaly'])
        anomalies_to_save.to_csv(output_path)
    df.drop(columns=['anomaly'], inplace=True)
    return anomalies


def temporal_split(df, train_end='2004-12-31'):
    train = df[df.index <= train_end]
    test = df[df.index > train_end]
    return train, test


def run_regression_task(df, target_pollutant_raw, horizons=[1, 6, 12, 24]):
    target_col = target_pollutant_raw.replace('.', '_').replace('(', '_').replace(')', '')
    if target_col not in df.columns:
        return None
    results = {}
    for H in horizons:
        y = df[target_col].shift(-H).dropna()
        X = df.loc[y.index].drop(columns=ALL_POLLUTANTS_CLEAN, errors='ignore')
        train_mask = X.index <= '2004-12-31'
        test_mask = X.index > '2004-12-31'
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        if X_train.empty or X_test.empty:
            continue
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        models = {
            'LinearRegression': LinearRegression(),
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        }
        horizon_results = {}
        for model_name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            horizon_results[model_name] = {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'y_test': y_test, 'y_pred': y_pred,
                                           'model': model}
            _plot_regression_diagnostics(y_test, y_pred, model_name, target_pollutant_raw, H)
            if hasattr(model, 'feature_importances_'):
                _plot_feature_importance(model, X_train.columns, model_name, f"reg_{target_col}_H{H}", topk=20)
        y_naive = df[target_col].loc[X_test.index]
        rmse_naive = np.sqrt(mean_squared_error(y_test, y_naive))
        mae_naive = mean_absolute_error(y_test, y_naive)
        r2_naive = r2_score(y_test, y_naive)
        horizon_results['Naive_Baseline'] = {'RMSE': rmse_naive, 'MAE': mae_naive, 'R2': r2_naive}
        results[f"{H}h"] = horizon_results
    return results


def _plot_regression_diagnostics(y_true, y_pred, model_name, pollutant, horizon):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(y_true, y_pred, alpha=0.5, s=10)
    axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual')
    axes[0].set_ylabel('Predicted')
    axes[0].set_title(f'{model_name}: Predicted vs Actual\n{pollutant}, H={horizon}h')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    residuals = y_true - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.5, s=10)
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title(f'{model_name}: Residual Plot\n{pollutant}, H={horizon}h')
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    safe_pollutant = pollutant.replace('(', '_').replace(')', '').replace('.', '_')
    plt.savefig(os.path.join(DIAG_DIR, f'diag_{model_name}_{safe_pollutant}_H{horizon}.png'), dpi=300,
                bbox_inches='tight')
    plt.close()


def _plot_feature_importance(model, feature_names, model_name, task_id, topk=20):
    output_dir = CLS_IMP_DIR if 'cls' in task_id else REG_IMP_DIR
    if not hasattr(model, "feature_importances_"):
        return
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:topk]
    plt.figure(figsize=(10, 8))
    plt.barh(np.array(feature_names)[idx][::-1], importances[idx][::-1])
    plt.xlabel("Importance")
    plt.title(f"Top {topk} Features - {model_name}\n{task_id}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"fi_{model_name}_{task_id}.png"), dpi=300)
    plt.close()


def summarize_regression_results(all_results):
    summary_data = []
    for pollutant, results in all_results.items():
        for horizon, models in results.items():
            for model_name, metrics in models.items():
                if 'RMSE' in metrics:
                    summary_data.append(
                        {'Pollutant': pollutant, 'Horizon': horizon, 'Model': model_name, 'RMSE': metrics['RMSE'],
                         'MAE': metrics.get('MAE', np.nan), 'R2': metrics.get('R2', np.nan)})
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(REG_DATA_DIR, 'regression_summary.csv'), index=False)
    print(summary_df.to_string(index=False))
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    horizons = ['1h', '6h', '12h', '24h']
    for i, h in enumerate(horizons):
        h_data = summary_df[summary_df['Horizon'] == h]
        if not h_data.empty:
            pivot = h_data.pivot(index='Pollutant', columns='Model', values='RMSE')
            pivot.plot(kind='bar', ax=axes[i], rot=45)
            axes[i].set_title(f'RMSE Comparison: {h} Ahead')
            axes[i].set_ylabel('RMSE')
            axes[i].set_xlabel('Pollutant')
            axes[i].legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[i].grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(REG_PLOT_DIR, 'regression_rmse_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def run_classification_task(df, target_pollutant_raw='CO(GT)', horizons=[1, 6, 12, 24]):
    target_col = target_pollutant_raw.replace('.', '_').replace('(', '_').replace(')', '')
    if target_col not in df.columns:
        print(f"[Warning] {target_col} is not in df")
        return None

    def discretize_co(val: float) -> int:
        if val < 1.5:
            return 0   # Low
        elif val < 2.5:
            return 1   # Medium
        else:
            return 2   # High

    df_feat = df.copy()

    num_cols = df_feat.select_dtypes(include=np.number).columns.tolist()
    feature_cols = [c for c in num_cols if c != target_col]

    lags = [1, 6, 12, 24]
    for c in feature_cols:
        for L in lags:
            df_feat[f'{c}_lag{L}'] = df_feat[c].shift(L)
        df_feat[f'{c}_roll3'] = df_feat[c].rolling(3).mean()
        df_feat[f'{c}_roll6'] = df_feat[c].rolling(6).mean()

    df_feat['hour'] = df_feat.index.hour
    df_feat['weekday'] = df_feat.index.weekday
    df_feat['month'] = df_feat.index.month

    results = {}

    for H in horizons:
        # y_t = discretize( CO[t+H] )
        y = df_feat[target_col].shift(-H).apply(discretize_co)
        X = df_feat.drop(columns=[target_col])

        data = pd.concat([X, y.rename('target')], axis=1).dropna()

        train_mask = data.index.year == 2004
        test_mask = data.index.year == 2005
        train_df = data[train_mask]
        test_df = data[test_mask]

        if train_df.empty or test_df.empty:
            print(f"[Warning] H={H}h 没有足够的 train/test 数据，跳过该步长。")
            continue

        split_idx = int(len(train_df) * 0.8)
        tr_df = train_df.iloc[:split_idx]
        val_df = train_df.iloc[split_idx:]

        X_tr, y_tr = tr_df.drop(columns=['target']), tr_df['target']
        X_val, y_val = val_df.drop(columns=['target']), val_df['target']
        X_te, y_te = test_df.drop(columns=['target']), test_df['target']

        X_trval = pd.concat([X_tr, X_val], axis=0)
        y_trval = pd.concat([y_tr, y_val], axis=0)

        logreg = Pipeline([
            ('scaler', StandardScaler(with_mean=True)),
            ('clf', LogisticRegression(
                max_iter=2000,
                class_weight='balanced',
                solver='lbfgs'
            )),
        ])

        xgb_clf = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            n_estimators=600,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            eval_metric='mlogloss',
            tree_method='hist',
            random_state=42,
        )

        rf_clf = RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            max_features='sqrt',
            class_weight='balanced',
            n_jobs=-1,
            random_state=42,
        )

        models = {
            'LogisticRegression': logreg,
            'XGBoost': xgb_clf,
            'RandomForest': rf_clf,
        }

        horizon_results = {}

        y_test_array = y_te.to_numpy(dtype=int)

        for model_name, model in models.items():
            model.fit(X_trval, y_trval)
            y_pred = model.predict(X_te)
            y_proba = model.predict_proba(X_te)

            acc = accuracy_score(y_te, y_pred)
            f1 = f1_score(y_te, y_pred, average='macro')

            horizon_results[model_name] = {'Accuracy': acc, 'F1': f1}

            _plot_confusion_matrix(y_test_array, y_pred, model_name, H)
            _plot_multiclass_roc(y_test_array, y_proba, model_name, H)
            _plot_multiclass_pr(y_test_array, y_proba, model_name, H)

            if hasattr(model, 'feature_importances_'):
                _plot_feature_importance(
                    model,
                    X_trval.columns,
                    model_name,
                    f"cls_CO_H{H}",
                    topk=20,
                )

            if model_name == 'LogisticRegression':
                _plot_logreg_coefficients(model, X_trval.columns, H)

        y_now = df[target_col].apply(discretize_co)
        y_future = df[target_col].shift(-H).apply(discretize_co)
        test_mask_nb = (y_future.index.year == 2005)

        y_true_nb = y_future[test_mask_nb].dropna()
        y_pred_nb = y_now[test_mask_nb].loc[y_true_nb.index]

        acc_nb = accuracy_score(y_true_nb, y_pred_nb)
        f1_nb = f1_score(y_true_nb, y_pred_nb, average='macro')

        horizon_results['Naive_Baseline'] = {'Accuracy': acc_nb, 'F1': f1_nb}

        results[f"{H}h"] = horizon_results

    if results:
        summarize_classification_results(results)
    return results


def _plot_confusion_matrix(y_true, y_pred, model_name, H):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(CLASS_NAMES)), normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=140)
    disp.plot(values_format='.2f', cmap='Blues', ax=ax, colorbar=True)
    ax.set_title(f"Confusion Matrix: {model_name}\nH={H}h")
    fig.tight_layout()
    plt.savefig(os.path.join(CM_DIR, f"cm_{model_name}_H{H}.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)


def _plot_multiclass_roc(y_true, proba, model_name, H):
    y_bin = label_binarize(y_true, classes=np.arange(len(CLASS_NAMES)))
    fpr, tpr, roc_auc = dict(), dict(), dict()
    for i in range(len(CLASS_NAMES)):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    roc_auc["macro"] = np.mean([roc_auc[i] for i in range(len(CLASS_NAMES))])

    fig, ax = plt.subplots(figsize=(8, 7), dpi=140)
    ax.plot(fpr["micro"], tpr["micro"], lw=2.5, label=f"micro-avg (AUC={roc_auc['micro']:.3f})", color='deeppink',
            linestyle=':')
    colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c'])
    for i, color in zip(range(len(CLASS_NAMES)), colors):
        ax.plot(fpr[i], tpr[i], color=color, lw=2, label=f"Class '{CLASS_NAMES[i]}' (AUC={roc_auc[i]:.3f})")
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Classifier')
    ax.set_xlim([0, 1]);
    ax.set_ylim([0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12);
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f"ROC Curves: {model_name}\nH={H}h (macroAUC={roc_auc['macro']:.3f})", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    plt.savefig(os.path.join(ROC_DIR, f"roc_{model_name}_H{H}.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)


def _plot_multiclass_pr(y_true, proba, model_name, H):
    y_bin = label_binarize(y_true, classes=np.arange(len(CLASS_NAMES)))
    precision, recall, ap = dict(), dict(), dict()
    for i in range(len(CLASS_NAMES)):
        precision[i], recall[i], _ = precision_recall_curve(y_bin[:, i], proba[:, i])
        ap[i] = average_precision_score(y_bin[:, i], proba[:, i])
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_bin.ravel(), proba.ravel())
    ap["micro"] = average_precision_score(y_bin, proba, average="micro")

    fig, ax = plt.subplots(figsize=(8, 7), dpi=140)
    ax.plot(recall["micro"], precision["micro"], lw=2.5, label=f"micro-avg (AP={ap['micro']:.3f})", color='deeppink',
            linestyle=':')
    colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c'])
    for i, color in zip(range(len(CLASS_NAMES)), colors):
        ax.plot(recall[i], precision[i], color=color, lw=2, label=f"Class '{CLASS_NAMES[i]}' (AP={ap[i]:.3f})")
    ax.set_xlim([0, 1]);
    ax.set_ylim([0, 1.05])
    ax.set_xlabel('Recall', fontsize=12);
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(f"Precision-Recall Curves: {model_name}\nH={H}h", fontsize=14)
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    plt.savefig(os.path.join(PR_DIR, f"pr_{model_name}_H{H}.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)


def _plot_logreg_coefficients(model, feature_names, H, topk=20):
    if not hasattr(model, "coef_"): return
    for k, class_name in enumerate(CLASS_NAMES):
        coefs = model.coef_[k]
        idx = np.argsort(np.abs(coefs))[::-1][:topk]
        plt.figure(figsize=(10, 8))
        colors = ['red' if c < 0 else 'blue' for c in coefs[idx][::-1]]
        plt.barh(np.array(feature_names)[idx][::-1], coefs[idx][::-1], color=colors)
        plt.xlabel("Coefficient Value", fontsize=12)
        plt.title(f"Top {topk} LogReg Coefficients\nClass: {class_name}, H={H}h", fontsize=14)
        plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        plt.tight_layout()
        plt.savefig(os.path.join(COEF_DIR, f"coef_LogReg_{class_name}_H{H}.png"), dpi=300, bbox_inches='tight')
        plt.close()


def summarize_classification_results(results):
    summary_data = []
    for horizon, models in results.items():
        for model_name, metrics in models.items():
            if 'Accuracy' in metrics:
                summary_data.append({'Horizon': horizon, 'Model': model_name, 'Accuracy': metrics['Accuracy'],
                                     'F1': metrics.get('F1', np.nan)})
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(CLS_DATA_DIR, 'classification_summary.csv'), index=False)
    print(summary_df.to_string(index=False))

    for metric in ["Accuracy", "F1"]:
        fig, ax = plt.subplots(figsize=(12, 7), dpi=140)
        pivot_df = summary_df.pivot(index='Horizon', columns='Model', values=metric)
        order = [m for m in ['Naive_Baseline', 'LogisticRegression', 'RandomForest', 'XGBoost'] if
                 m in pivot_df.columns]
        pivot_df[order].plot(kind='bar', ax=ax, width=0.8, rot=0)
        ax.set_ylabel(f'Weighted {metric}' if metric == "F1" else metric, fontsize=12)
        ax.set_xlabel('Prediction Horizon', fontsize=12)
        ax.set_title(f'Classification {metric} Comparison Across Models and Horizons', fontsize=14)
        ax.legend(title='Model')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.0)
        plt.tight_layout()
        plt.savefig(os.path.join(BAR_DIR, f'classification_{metric.lower()}_comparison.png'), dpi=300,
                    bbox_inches='tight')
        plt.close()


def main():
    raw_data_path = os.path.join("data", "AirQualityUCI.csv")

    cleaning_strategy = "ffill"

    pollutants_to_predict = ['CO(GT)', 'NMHC(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)']
    horizons = [1, 6, 12, 24]

    df = load_prepared_data(
        raw_path=raw_data_path,
        cleaning_strategy=cleaning_strategy,
        save_cleaned=True,
        cleaned_output_path="cleaned_air_quality.csv",
    )
    if df is None:
        print("Can not load data!")
        return

    run_exploratory_analysis(df)

    anomaly_features = [p for p in ALL_POLLUTANTS_CLEAN if p in df.columns]
    anomaly_output_path = os.path.join(OUTPUT_DIR, 'anomalies_detected.csv')
    run_anomaly_detection(df, anomaly_features, anomaly_output_path)

    all_regression_results = {}
    for pollutant in pollutants_to_predict:
        results = run_regression_task(df, pollutant, horizons)
        if results:
            pollutant_clean = pollutant.replace('.', '_').replace('(', '_').replace(')', '')
            all_regression_results[pollutant_clean] = results

    if all_regression_results:
        summarize_regression_results(all_regression_results)

    run_classification_task(df, 'CO(GT)', horizons)


if __name__ == "__main__":
    main()
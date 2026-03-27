import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.stats import linregress, skew, kurtosis
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

N_FOLDS = 7
SEED = 42
RANDOM_STATE = SEED

def load_data():
    print("Reading data...")
    train_log = pd.read_csv('data/lab_reports_train.csv')
    test_log = pd.read_csv('data/lab_reports_test.csv')
    train_label = pd.read_csv('data/results_train.csv')
    sample_submission = pd.read_csv('data/sample_submission.csv')
    print("Data read complete.")
    return train_log, test_log, train_label, sample_submission

def get_slope(series):
    if len(series) < 2:
        return 0.0
    slope, _, _, _, _ = linregress(range(len(series)), series)
    return float(slope) if not np.isnan(slope) else 0.0

def safe_dt_parse(df, started_col='Started on', completed_col='Completed'):
    if started_col not in df.columns and 'Started' in df.columns:
        df = df.rename(columns={'Started':'Started on'})
    if completed_col not in df.columns and 'Completed on' in df.columns:
        df = df.rename(columns={'Completed on':'Completed'})
    df[started_col] = pd.to_datetime(df[started_col], dayfirst=True, errors='coerce')
    df[completed_col] = pd.to_datetime(df[completed_col], dayfirst=True, errors='coerce')
    return df

def skew_safe(x):
    return skew(x) if len(x) > 2 else 0

def kurt_safe(x):
    return kurtosis(x) if len(x) > 2 else 0

def pos_count(x):
    return (x > 0).sum()

def neg_count(x):
    return (x < 0).sum()

def mode_safe(x):
    return x.mode()[0] if len(x.mode()) > 0 else -1

def feature_engineering(df):
    print("Advanced Feature Engineering...")
    df = safe_dt_parse(df, 'Started on', 'Completed')
    
    # Duration features
    df['duration_minutes'] = (df['Completed'] - df['Started on']).dt.total_seconds() / 60
    df['duration_minutes'] = df['duration_minutes'].replace([np.inf, -np.inf], np.nan)
    df['is_time_missing'] = df['Completed'].isna().astype(int)
    
    if 'labType' in df.columns:
        df['duration_minutes'] = df.groupby('labType')['duration_minutes'].transform(lambda x: x.fillna(x.median()))
    df['duration_minutes'] = df['duration_minutes'].fillna(df['duration_minutes'].median()).fillna(0)
    
    # Time features
    time_ref = df['Completed'].fillna(df['Started on'])
    df['hour_submit'] = time_ref.dt.hour.fillna(-1).astype(int)
    df['day_of_week'] = time_ref.dt.dayofweek.fillna(-1).astype(int)
    df['day_of_month'] = time_ref.dt.day.fillna(-1).astype(int)
    df['month'] = time_ref.dt.month.fillna(-1).astype(int)
    df['is_night'] = ((df['hour_submit'] >= 22) | (df['hour_submit'] <= 4)).astype(int)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_business_hours'] = ((df['hour_submit'] >= 9) & (df['hour_submit'] <= 17)).astype(int)
    
    # Sort for temporal features
    df = df.sort_values(by=['student_pid', 'Completed'])
    
    # Calculate rolling statistics
    df['grade_roll_mean_3'] = df.groupby('student_pid')['Grade/10.00'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df['grade_roll_mean_5'] = df.groupby('student_pid')['Grade/10.00'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df['grade_roll_std_3'] = df.groupby('student_pid')['Grade/10.00'].transform(lambda x: x.rolling(3, min_periods=1).std()).fillna(0)
    
    # Cumulative features
    df['cumulative_avg_grade'] = df.groupby('student_pid')['Grade/10.00'].transform(lambda x: x.expanding().mean())
    df['cumulative_submissions'] = df.groupby('student_pid').cumcount() + 1
    
    # Grade change features
    df['grade_diff'] = df.groupby('student_pid')['Grade/10.00'].diff().fillna(0)
    df['grade_pct_change'] = df.groupby('student_pid')['Grade/10.00'].pct_change().fillna(0).replace([np.inf, -np.inf], 0)
    
    # Duration change features
    df['duration_diff'] = df.groupby('student_pid')['duration_minutes'].diff().fillna(0)
    
    # Time gaps between submissions
    df['time_gap_hours'] = df.groupby('student_pid')['Completed'].diff().dt.total_seconds() / 3600
    df['time_gap_hours'] = df['time_gap_hours'].fillna(0).replace([np.inf, -np.inf], 0)
    

    # Comprehensive aggregations
    agg_funcs = {
        'Grade/10.00': ['mean', 'max', 'min', 'std', 'sum', 'median', 'first', 'last', 'count', 
                         skew_safe, kurt_safe],
        'duration_minutes': ['mean', 'sum', 'max', 'min', 'std', 'median', 
                            skew_safe],
        'topic': ['nunique', 'count'],
        'is_night': ['mean', 'sum'],
        'is_weekend': ['mean', 'sum'],
        'is_business_hours': ['mean'],
        'is_time_missing': ['mean', 'sum'],
        'grade_roll_mean_3': ['last', 'mean'],
        'grade_roll_std_3': ['last', 'mean'],
        'grade_diff': ['mean', 'std', 'min', 'max', pos_count, neg_count],
        'grade_pct_change': ['mean', 'std'],
        'duration_diff': ['mean', 'std'],
        'time_gap_hours': ['mean', 'std', 'max', 'min'],
        'cumulative_avg_grade': ['last'],
        'cumulative_submissions': ['last'],
        'hour_submit': ['mean', mode_safe],
        'day_of_week': ['mean', mode_safe],
    }
    
    for c in list(agg_funcs.keys()):
        if c not in df.columns:
            df[c] = 0
    
    features = df.groupby("student_pid").agg(agg_funcs)
    features.columns = ['_'.join([str(col[0]), str(i)]) if isinstance(col[1], type(lambda: None)) 
                        else '_'.join(map(str, col)) for i, col in enumerate(features.columns.values)]
    features = features.reset_index()
    print (features.head(5))

    
    # Grade trend analysis
    print("Calculating advanced trends...")
    grade_slope = df.groupby("student_pid")['Grade/10.00'].apply(list).apply(get_slope)
    features = features.merge(grade_slope.rename('grade_trend'), on='student_pid', how='left')
    
    # Performance momentum (recent vs early performance)
    def get_momentum(grades):
        if len(grades) < 4:
            return 0
        recent = np.mean(grades[-3:])
        early = np.mean(grades[:3])
        return recent - early
    
    grade_momentum = df.groupby("student_pid")['Grade/10.00'].apply(list).apply(get_momentum)
    features = features.merge(grade_momentum.rename('grade_momentum'), on='student_pid', how='left')
    
    # Streaks (consecutive improvements/declines)
    def get_max_streak(changes):
        if len(changes) == 0:
            return 0, 0
        pos_streak = neg_streak = 0
        max_pos = max_neg = 0
        for c in changes:
            if c > 0:
                pos_streak += 1
                neg_streak = 0
                max_pos = max(max_pos, pos_streak)
            elif c < 0:
                neg_streak += 1
                pos_streak = 0
                max_neg = max(max_neg, neg_streak)
            else:
                pos_streak = neg_streak = 0
        return max_pos, max_neg
    
    streaks = df.groupby("student_pid")['grade_diff'].apply(list).apply(get_max_streak)
    features['max_improvement_streak'] = streaks.apply(lambda x: x[0])
    features['max_decline_streak'] = streaks.apply(lambda x: x[1])
    
    # Time-based features
    last_time = df.groupby('student_pid')['Completed'].max().rename('last_completed')
    first_time = df.groupby('student_pid')['Started on'].min().rename('first_started')
    features = features.merge(last_time.reset_index(), on='student_pid', how='left')
    features = features.merge(first_time.reset_index(), on='student_pid', how='left')
    
    features['time_since_last_hours'] = (pd.Timestamp.now() - features['last_completed']).dt.total_seconds() / 3600
    features['time_between_first_last_hours'] = (features['last_completed'] - features['first_started']).dt.total_seconds() / 3600
    features['avg_submission_rate_per_day'] = features['Grade/10.00_count'] / (features['time_between_first_last_hours'] / 24 + 1)
    
    features['time_since_last_hours'] = features['time_since_last_hours'].fillna(features['time_since_last_hours'].median())
    features['time_between_first_last_hours'] = features['time_between_first_last_hours'].fillna(0)
    
    # Lab type pivot features
    if 'labType' in df.columns:
        pivot_grade = df.pivot_table(index='student_pid', columns='labType', values='Grade/10.00', aggfunc='mean').fillna(0)
        pivot_grade.columns = [f'avg_grade_{str(col)}' for col in pivot_grade.columns]
        pivot_count = df.pivot_table(index='student_pid', columns='labType', values='Grade/10.00', aggfunc='count').fillna(0)
        pivot_count.columns = [f'count_{str(col)}' for col in pivot_count.columns]
        pivot_last = df.groupby('student_pid').last()['labType'].reset_index()
        pivot_last.columns = ['student_pid', 'last_labType']
        
        features = features.merge(pivot_grade.reset_index(), on='student_pid', how='left')
        features = features.merge(pivot_count.reset_index(), on='student_pid', how='left')
    
    # Advanced engineered features
    features['efficiency_score'] = features['Grade/10.00_sum'] / (features['duration_minutes_sum'] + 1e-6)
    features['missing_time_rate'] = features['is_time_missing_sum'] / (features['topic_count'] + 1)
    features['intensity'] = features['topic_count'] / (features['topic_nunique'] + 1)
    features['grade_consistency'] = 1.0 / (features['Grade/10.00_std'].fillna(0.0) + 0.1)
    features['improvement_rate'] = features['grade_trend'] * features['Grade/10.00_count']
    features['early_performance'] = features['Grade/10.00_first']
    features['late_performance'] = features['Grade/10.00_last']
    features['performance_range'] = features['Grade/10.00_max'] - features['Grade/10.00_min']
    features['grade_cv'] = features['Grade/10.00_std'] / (features['Grade/10.00_mean'] + 0.1)
    
    # Relative performance (compared to median)
    median_grade = df['Grade/10.00'].median()
    features['grade_vs_median'] = features['Grade/10.00_mean'] - median_grade
    
    # Replace inf/nan
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Drop datetime columns
    drop_cols = [c for c in ['last_completed', 'first_started'] if c in features.columns]
    features = features.drop(columns=drop_cols, errors='ignore')
    
    print("Feature Engineering Done.")
    return features

def create_meta_features(train_feats, test_feats):
    """Create interaction and polynomial features"""
    for df in [train_feats, test_feats]:
        # Key interactions
        if 'Grade/10.00_mean' in df.columns and 'efficiency_score' in df.columns:
            df['grade_eff_inter'] = df['Grade/10.00_mean'] * df['efficiency_score']
        if 'Grade/10.00_mean' in df.columns and 'grade_trend' in df.columns:
            df['grade_trend_inter'] = df['Grade/10.00_mean'] * df['grade_trend']
        if 'topic_count' in df.columns and 'Grade/10.00_mean' in df.columns:
            df['volume_quality'] = df['topic_count'] * df['Grade/10.00_mean']
        if 'grade_consistency' in df.columns and 'Grade/10.00_mean' in df.columns:
            df['consistent_high_performer'] = df['grade_consistency'] * df['Grade/10.00_mean']
        
        # Polynomial features for key metrics
        key_features = ['Grade/10.00_mean', 'duration_minutes_mean', 'efficiency_score', 
                       'grade_trend', 'topic_count', 'grade_momentum']
        for feat in key_features:
            if feat in df.columns:
                df[f'{feat}_sq'] = df[feat] ** 2
                df[f'{feat}_sqrt'] = np.sqrt(np.abs(df[feat]))
    
    return train_feats, test_feats

def training(train_df, test_df, label):
    # Merge labels
    train_data = train_df.merge(label.rename(columns={'student_id':'student_pid'}) if 'student_id' in label.columns else label, 
                                 on='student_pid', how='left')
    X = train_data.drop(columns=['student_pid', 'Pass'], errors='ignore')
    y = train_data['Pass'].fillna(0).astype(int)
    
    X_test = test_df.set_index('student_pid').reindex(columns=X.columns).fillna(0).reset_index()
    test_ids = X_test['student_pid'].values
    X_test = X_test.drop(columns=['student_pid'], errors='ignore')
    
    print(f"\n--- Training with {N_FOLDS}-Fold CV ---")
    print(f"Training samples: {len(X)}, Test samples: {len(X_test)}")
    print(f"Positive rate: {y.mean():.4f}")
    
    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    oof_preds = np.zeros(len(X))
    test_preds_lgb = np.zeros(len(X_test))
    test_preds_xgb = np.zeros(len(X_test))
    test_preds_cat = np.zeros(len(X_test))
    fold_scores = []
    
    stack_train = np.zeros((len(X), 3))
    stack_test = np.zeros((len(X_test), 3))
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"\nFold {fold+1}/{N_FOLDS}")
        X_tr, X_val = X.iloc[tr_idx].copy(), X.iloc[val_idx].copy()
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        
        # LightGBM with optimized parameters
        lgb_params = {
            'n_estimators': 4000,
            'learning_rate': 0.015,
            'num_leaves': 30,
            'max_depth': 4,
            'bagging_fraction': 0.75,
            'feature_fraction': 0.75,
            'bagging_freq': 5,
            'reg_alpha': 1.0,
            'reg_lambda': 1.0,
            'min_child_samples': 40,
            'min_split_gain': 0.01,
            'objective': 'binary',
            'metric': 'auc',
            'random_state': RANDOM_STATE + fold,
            'n_jobs': -1,
            'verbosity': -1,
            'colsample_bytree': 0.6
        }
        model_lgb = lgb.LGBMClassifier(**lgb_params)
        model_lgb.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=150, verbose=False)]
        )
        
        # XGBoost with optimized parameters
        xgb_params = {
            'n_estimators': 3500,
            'learning_rate': 0.015,
            'max_depth': 4,
            'subsample': 0.7,
            'colsample_bytree': 0.6,
            'gamma': 1.5,
            'reg_alpha': 1.0,
            'reg_lambda': 1.0,
            'min_child_weight': 3,
            'objective': 'binary:logistic',
            'eval_metric': 'aucpr',
            'random_state': RANDOM_STATE + fold,
            'n_jobs': -1
        }
        model_xgb = xgb.XGBClassifier(**xgb_params)
        model_xgb.fit(
            X_tr, y_tr, 
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # CatBoost with optimized parameters
        model_cat = CatBoostClassifier(
            iterations=3000,
            learning_rate=0.02,
            depth=4,
            l2_leaf_reg=3,
            bagging_temperature=0.5,
            random_strength=0.5,
            eval_metric='AUC',
            random_seed=RANDOM_STATE + fold,
            verbose=False,
            task_type='CPU'
        )
        model_cat.fit(
            X_tr, y_tr, 
            eval_set=(X_val, y_val), 
            early_stopping_rounds=250, 
            verbose=False
        )
        
        # Predictions
        p_lgb = model_lgb.predict_proba(X_val)[:,1]
        p_xgb = model_xgb.predict_proba(X_val)[:,1]
        p_cat = model_cat.predict_proba(X_val)[:,1]
        
        # Optimized weights
        val_pred = 0.35 * p_lgb + 0.35 * p_xgb + 0.30 * p_cat
        oof_preds[val_idx] = val_pred
        
        stack_train[val_idx, 0] = p_lgb
        stack_train[val_idx, 1] = p_xgb
        stack_train[val_idx, 2] = p_cat
        
        # Test predictions
        p_lgb_test = model_lgb.predict_proba(X_test)[:,1]
        p_xgb_test = model_xgb.predict_proba(X_test)[:,1]
        p_cat_test = model_cat.predict_proba(X_test)[:,1]
        
        test_preds_lgb += p_lgb_test / N_FOLDS
        test_preds_xgb += p_xgb_test / N_FOLDS
        test_preds_cat += p_cat_test / N_FOLDS
        
        stack_test[:,0] += p_lgb_test / N_FOLDS
        stack_test[:,1] += p_xgb_test / N_FOLDS
        stack_test[:,2] += p_cat_test / N_FOLDS
        
        # Score
        fold_score = average_precision_score(y_val, val_pred)
        fold_scores.append(fold_score)
        print(f"Fold {fold+1} AUPRC: {fold_score:.5f}")
    
    oof_score = average_precision_score(y, oof_preds)
    print(f"\n{'='*50}")
    print(f"OOF AUPRC: {oof_score:.5f}")
    print(f"Mean Fold AUPRC: {np.mean(fold_scores):.5f} ± {np.std(fold_scores):.5f}")
    print(f"{'='*50}")
    
    # Stacking meta-learner
    print("\nTraining stacking meta-learner...")
    meta_clf = LogisticRegression(random_state=RANDOM_STATE, max_iter=3000, C=0.1)
    meta_clf.fit(stack_train, y)
    final_test_pred_stack = meta_clf.predict_proba(stack_test)[:,1]
    
    # Final ensemble with weighted average and stacking
    test_preds_weighted = 0.35 * test_preds_lgb + 0.35 * test_preds_xgb + 0.30 * test_preds_cat
    final_test_preds = 0.5 * test_preds_weighted + 0.5 * final_test_pred_stack
    
    return test_ids, final_test_preds, oof_preds

def main():
    train_log, test_log, label, sample_submission = load_data()
    
    train_features = feature_engineering(train_log)
    test_features = feature_engineering(test_log)
    
    # train_features, test_features = create_meta_features(train_features, test_features)
    
    test_ids, predictions, oof = training(train_features, test_features, label)
    
    submission = pd.DataFrame({
        'student_id': test_ids,
        'Pass': predictions
    })
    
    if 'student_id' in sample_submission.columns:
        submission = sample_submission[['student_id']].merge(submission, on='student_id', how='left')
    submission['Pass'] = submission['Pass'].fillna(submission['Pass'].mean())
    
    print("\nSubmission preview:")
    print(submission.head(10))
    print(f"\nPrediction statistics:")
    print(f"Mean: {submission['Pass'].mean():.5f}")
    print(f"Std: {submission['Pass'].std():.5f}")
    print(f"Min: {submission['Pass'].min():.5f}")
    print(f"Max: {submission['Pass'].max():.5f}")
    
    submission.to_csv('submission.csv', index=False)
    print("\nSubmission file saved successfully!")

if __name__ == "__main__":
    main()
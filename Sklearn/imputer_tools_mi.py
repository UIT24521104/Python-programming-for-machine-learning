import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, mean_squared_error, mean_absolute_error
from sklearn.impute import MissingIndicator

def calculate_nrmse(df_original, df_imputed, df_miss):
    errors = []
    
    for col in df_original.columns:
        # 1. Tìm vị trí mà dữ liệu ban đầu bị thiếu (True tại vị trí NaN)
        mask = df_miss[col].isna()
        # Chỉ tính toán nếu cột đó thực sự có dữ liệu thiếu
        if mask.sum() > 0:
            actual = df_original.loc[mask, col]
            predicted = df_imputed.loc[mask, col]
            rmse = np.sqrt(np.mean((actual - predicted)**2))
            
            # Tính NRMSE (để so sánh giữa các cột khác đơn vị)
            range_val = df_original[col].max() - df_original[col].min()
            nrmse = rmse / range_val if range_val != 0 else rmse      
            errors.append(nrmse)
            
    return errors if len(errors) > 0 else [0]

def missing_imputer(df: pd.DataFrame, fold_method, clf_method, imputer_name, 
                    imputer_method=None, cols=None, target=None, missing_rate=0.25, 
                    is_missing=False, add_indicator=False, type='classification'):
    """
    Hàm thực hiện cross-validation cho các phương pháp impute.
    add_indicator=True sẽ thêm cột chỉ báo (MissingIndicator) vào dữ liệu trước khi impute.
    """
    n = len(df)
    features = cols if cols else df.columns[:-1].tolist()
    label = target if target else df.columns[-1]
    
    # Create missing-value features nếu chưa có
    df_miss = df.copy()
    if not is_missing:
        for col in features:
            df_miss.loc[np.random.rand(n) < missing_rate, col] = np.nan

    nrmse_scores = []
    f1_scores = []
    acc_scores = []
    mae_scores = []
    mse_scores = []

    for train_idx, test_idx in fold_method.split(df[features], df[label]):
        X_train_org = df.loc[train_idx, features]
        y_train_org = df.loc[train_idx, label]
        X_train_miss = df_miss.loc[train_idx, features]
        
        X_test_miss = df_miss.loc[test_idx, features]
        y_test_org = df.loc[test_idx, label]
        
        # ---------------------------------------------------------
        # XỬ LÝ MISSING INDICATOR
        # ---------------------------------------------------------
        current_train_features = X_train_miss.copy()
        current_test_features = X_test_miss.copy()
        
        if add_indicator:
            indicator = MissingIndicator(features='missing-only')
            indicator.fit(X_train_miss)
            
            # Tạo tên cột mới cho indicator (VD: 'var_name_missing')
            missing_features_names = [f"{col}_missing" for col in indicator.features_]
            
            # Chuyển đổi và tạo dataframe
            train_indicators = pd.DataFrame(indicator.transform(X_train_miss).astype(int), 
                                            columns=missing_features_names, index=train_idx)
            test_indicators = pd.DataFrame(indicator.transform(X_test_miss).astype(int), 
                                           columns=missing_features_names, index=test_idx)
            
            # Ghép vào features
            current_train_features = pd.concat([current_train_features, train_indicators], axis=1)
            current_test_features = pd.concat([current_test_features, test_indicators], axis=1)

        # ---------------------------------------------------------
        # XỬ LÝ IMPUTATION THÔNG THƯỜNG
        # ---------------------------------------------------------
        if imputer_name.lower() != 'baseline\n(drop)':
            # Fit imputer trên dữ liệu (đã có indicator nếu add_indicator=True)
            # Lưu ý: Imputer phải xử lý được các cột indicator (thường là 0/1)
            train_imp = imputer_method.fit_transform(current_train_features)
            test_imp = imputer_method.transform(current_test_features)
            
            X_train_imp = pd.DataFrame(train_imp, index=train_idx, columns=current_train_features.columns)
            X_test_imp = pd.DataFrame(test_imp, index=test_idx, columns=current_test_features.columns)
            
            # Tính NRMSE (chỉ tính trên phần dữ liệu gốc, không tính trên cột indicator)
            # Chúng ta cần cắt lại các cột gốc để tính lỗi
            nrmse = calculate_nrmse(X_train_org, X_train_imp[features], X_train_miss)
            nrmse_scores.append(np.mean(nrmse))
            
        else:
            # Baseline Drop
            combined_train = pd.concat([current_train_features, y_train_org], axis=1).dropna()
            X_train_imp = combined_train.drop(columns=[label])
            y_train_clean = combined_train[label]
            
            combined_test = pd.concat([current_test_features, y_test_org], axis=1).dropna()
            X_test_imp = combined_test.drop(columns=[label])
            y_test_clean = combined_test[label]
            
            # Cập nhật nhãn
            y_train_org, y_test_org = y_train_clean, y_test_clean
            
        # Predict
        clf_method.fit(X_train_imp, y_train_org)
        y_pred = clf_method.predict(X_test_imp)
        
        if type == 'classification':
            f1_scores.append(f1_score(y_test_org, y_pred, average='macro'))
            acc_scores.append(accuracy_score(y_test_org, y_pred))
        else:
            mae_scores.append(mean_absolute_error(y_test_org, y_pred))
            mse_scores.append(mean_squared_error(y_test_org, y_pred))

    # Kết quả
    res = {
        'Method name': imputer_name, 
        'Avg_NRMSE': np.mean(nrmse_scores) if nrmse_scores else -1e-3
    }
    
    if type == 'classification':
        res.update({'F1': np.mean(f1_scores), 'Accuracy': np.mean(acc_scores)})
    else:
        res.update({'MAE': np.mean(mae_scores), 'MSE': np.mean(mse_scores)})
        
    return res
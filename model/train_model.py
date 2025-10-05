import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def train_with_feature_selection(
    df,
    feature_cols,
    label_col,
    n_splits=5,
    n_drop=30,
    params=None,
    verbose_eval=0,
    model_save_path="lgbm_final_model.txt"
):
    # =======================
    # 交差検証+特徴量選択+最終モデルfit＆保存までを一つの関数で実行する
    # =======================
    
    if params is None:
        params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 34,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'class_weight': 'balanced',
            'max_depth': 6,
            'verbose': -1,
            'random_state': 42,
            'min_child_samples': 10,
        }

    X = df[feature_cols].values
    y = df[label_col].astype(int).values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = []
    cv_reports = []
    fold_importances = []
    conf_matrices = []

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), 1):
        print(f"===== Fold {fold} =====")
        X_tr, X_val = X[train_idx], X[valid_idx]
        y_tr, y_val = y[train_idx], y[valid_idx]
        feature_list = feature_cols.copy()

        # 1st fit: 全特徴量でfitし重要度算出
        dtrain_full = lgb.Dataset(X_tr, label=y_tr)
        dvalid_full = lgb.Dataset(X_val, label=y_val)
        gbm_full = lgb.train(
            params, dtrain_full,
            valid_sets=[dvalid_full],
            num_boost_round=10000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=20, verbose=False),
                lgb.log_evaluation(verbose_eval)
            ]
        )
        importances = gbm_full.feature_importance(importance_type='gain')
        importance_df = pd.DataFrame({'feature': feature_list, 'importance': importances})
        low_features = importance_df.sort_values('importance').head(n_drop)['feature'].tolist()
        fold_importances.append(importance_df)
        # 低重要度特徴量を削除
        feature_list_red = [f for f in feature_list if f not in low_features]
        print(f"特徴量数: {len(feature_list_red)} (削除: {n_drop}個)")

        # 2nd fit: 削除後特徴量だけで再fit
        X_tr_red = pd.DataFrame(X_tr, columns=feature_list)[feature_list_red].values
        X_val_red = pd.DataFrame(X_val, columns=feature_list)[feature_list_red].values
        dtrain_red = lgb.Dataset(X_tr_red, label=y_tr)
        dvalid_red = lgb.Dataset(X_val_red, label=y_val)
        gbm_red = lgb.train(
            params, dtrain_red,
            valid_sets=[dvalid_red],
            num_boost_round=10000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=20, verbose=False),
                lgb.log_evaluation(verbose_eval)
            ]
        )
        y_pred = gbm_red.predict(X_val_red, num_iteration=gbm_red.best_iteration)
        y_pred_classes = np.argmax(y_pred, axis=1)
        acc = accuracy_score(y_val, y_pred_classes)
        print(f"[Fold {fold}] Accuracy: {acc:.5f}")
        print(classification_report(y_val, y_pred_classes, digits=5))
        cv_scores.append(acc)
        cv_reports.append(classification_report(y_val, y_pred_classes, digits=5, output_dict=True))
        conf_matrices.append(confusion_matrix(y_val, y_pred_classes))

    # 重要度の低い特徴量の頻度可視化（foldをまたいでよく落とされた特徴量の分析など）
    feature_drop_counts = {}
    for imp in fold_importances:
        dropped = imp.sort_values('importance').head(n_drop)['feature'].tolist()
        for f in dropped:
            feature_drop_counts[f] = feature_drop_counts.get(f, 0) + 1
    drop_summary = pd.DataFrame(list(feature_drop_counts.items()), columns=['feature', 'drop_count']).sort_values('drop_count', ascending=False)
    most_dropped_features = drop_summary.head(n_drop)['feature'].tolist()
    final_feature_list = [f for f in feature_cols if f not in most_dropped_features]
    print("最終モデルで使用する特徴量数:", len(final_feature_list))
    print("最終モデル特徴量リスト:", final_feature_list)

    # 削除後特徴量で全データfit→モデル保存
    X_final = df[final_feature_list].values
    y_final = df[label_col].astype(int).values

    dtrain_final = lgb.Dataset(X_final, label=y_final)
    final_gbm = lgb.train(
        params, dtrain_final,
        num_boost_round=10000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=20, verbose=False),
            lgb.log_evaluation(verbose_eval)
        ]
    )

    final_gbm.save_model(model_save_path)
    print(f"全データで学習した最終モデルを {model_save_path} として保存しました。")

    # 可視化（棒グラフやヒートマップ、スコア推移）
    # 必要なら元コードのまま呼び出し可

    return {
        "final_model": final_gbm,
        "final_feature_list": final_feature_list,
        "cv_scores": cv_scores,
        "cv_reports": cv_reports,
        "conf_matrices": conf_matrices,
        "drop_summary": drop_summary,
        "fold_importances": fold_importances
    }
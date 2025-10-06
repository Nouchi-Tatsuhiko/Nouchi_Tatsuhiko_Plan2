import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from features.feature_engineering import feature_engineering_pipeline
from features.labeling import make_long_short_label_and_drop_nan
from model.train_model import train_with_feature_selection
from backtest.backtest import backtest_pips

# データ読み込み
df = pd.read_csv('data/2003_GBPUSD_M1_M1_UTCPlus02-M5-No Session.csv')

# 特徴量生成
df = feature_engineering_pipeline(df)

# ラベル生成
df = make_long_short_label_and_drop_nan(df)

# 学習＆モデル保存
label_col = 'y'
exclude_cols = [label_col]
feature_cols = [
    col for col in df.columns
    if (col not in exclude_cols) and (pd.api.types.is_numeric_dtype(df[col]))
]
train_with_feature_selection(df, feature_cols, label_col)

# 結果はmodel/train_model.py内でpickle保存される
# 可視化はscripts/plot_cv_results.pyを参照
# もしくはnotebooks/visualize_cv_results.ipynbを参照

import pickle

results = train_with_feature_selection(df, feature_cols, label_col)
with open("results_cv.pkl", "wb") as f:
    pickle.dump(results, f)

print("学習と結果保存が完了しました。results_cv.pklを確認してください。")


test_df = backtest_pips(df)
test_df.to_csv("test_backtest.csv", index=False)
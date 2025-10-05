import pandas as pd
from features.feature_engineering import feature_engineering_pipeline
from features.labeling import make_long_short_label_and_drop_nan
from model.train_model import train_with_feature_selection

# データ読み込み
df = pd.read_csv('data/2003_GBPUSD_M1_M1_UTCPlus02-M5-No Session.csv')

# 特徴量生成
df = feature_engineering_pipeline(df)

# ラベル生成
df = make_long_short_label_and_drop_nan(df)

# 学習＆モデル保存
label_col = 'y'
exclude_cols = [label_col]
feature_cols = [col for col in df.columns if (col not in exclude_cols)]
train_with_feature_selection(df, feature_cols, label_col, n_drop=20)
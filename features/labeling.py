import numpy as np

# =======================
# 目的変数の作成
# =======================
def make_long_short_label_and_drop_nan(
    df,
    profit_pips=0.0015,
    loss_pips=0.0015,
    max_horizon=48
):
    y = np.full(len(df), np.nan)
    for i in range(len(df)):
        entry = df['Close'].iloc[i]
        for j in range(i+1, min(i+1+max_horizon, len(df))):
            high = df['High'].iloc[j]
            low = df['Low'].iloc[j]
            if high - entry >= profit_pips:
                y[i] = 1
                break
            if entry - low >= loss_pips:
                y[i] = 2
                break
        else:
            y[i] = 0
    df = df.copy()
    df['y'] = y
    df = df.dropna(axis=0, how='any')
    return df

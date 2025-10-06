import pandas as pd
import numpy as np

def backtest_pips(
    test_df, 
    pred_col='pred', 
    close_col='Close', 
    entry_col='Close', 
    high_col='High', 
    low_col='Low', 
    pip_unit=0.0001, 
    max_bars=48, 
    profit_pips=13, 
    loss_pips=16, 
    reach_pips=15
):
    """モデル予測と実際の価格推移からPIPSを計算するバックテスト
    ±15pips動いた時点で利益・損失を判定し、利益なら+13pips、損失なら-16pipsとする。
    48本以内にどちらにも到達しなければ、その時点の損益で計算。
    """

    results = []
    test_df = test_df.reset_index(drop=True)

    for idx, row in test_df.iterrows():
        entry_signal = row[pred_col] # 1:買い, -1:売り, 0:様子見
        entry_price = row[entry_col]

        # 様子見ならスキップ
        if entry_signal == 0:
            results.append(0)
            continue

        # バックテスト期間超過しないように調整
        end_idx = min(idx + max_bars, len(test_df)-1)

        resolved = False

        if entry_signal == 1:
            # 買いの場合
            for future_idx in range(idx+1, end_idx+1):
                high = test_df.loc[future_idx, high_col]
                low = test_df.loc[future_idx, low_col]

                # 利確判定（+15pips到達）
                if (high - entry_price)/pip_unit >= reach_pips:
                    results.append(profit_pips)
                    resolved = True
                    break
                # 損切判定（-15pips到達）
                if (entry_price - low)/pip_unit >= reach_pips:
                    results.append(-loss_pips)
                    resolved = True
                    break

            if not resolved:
                close = test_df.loc[end_idx, close_col]
                pips = (close - entry_price)/pip_unit
                results.append(pips)

        elif entry_signal == -1:
            # 売りの場合
            for future_idx in range(idx+1, end_idx+1):
                high = test_df.loc[future_idx, high_col]
                low = test_df.loc[future_idx, low_col]

                # 利確判定（-15pips到達）
                if (entry_price - low)/pip_unit >= reach_pips:
                    results.append(profit_pips)
                    resolved = True
                    break
                # 損切判定（+15pips到達）
                if (high - entry_price)/pip_unit >= reach_pips:
                    results.append(-loss_pips)
                    resolved = True
                    break

            if not resolved:
                close = test_df.loc[end_idx, close_col]
                pips = (entry_price - close)/pip_unit
                results.append(pips)

    test_df['backtest_pips'] = results
    print(f"バックテスト総獲得PIPS: {np.sum(results):.2f}")
    return test_df

print("backtest.py is loaded.")
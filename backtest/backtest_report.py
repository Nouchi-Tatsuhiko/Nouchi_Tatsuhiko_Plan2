import numpy as np
import matplotlib.pyplot as plt

def analyze_backtest(test_df, pips_col='backtest_pips', pred_col='pred', profit_pips=13, loss_pips=16):
    
    # test_df: バックテスト済みデータ（pred, backtest_pips カラム必須）
    
    # 総トレード数
    trade_mask = test_df[pred_col] != 0
    total_trades = trade_mask.sum()
    
    # 勝ち・負け数
    win_mask = (test_df[pips_col] > 0) & trade_mask
    win_count = win_mask.sum()
    lose_mask = (test_df[pips_col] < 0) & trade_mask
    lose_count = lose_mask.sum()
    win_rate = win_count / total_trades if total_trades > 0 else 0

    # 獲得・損失PIPS
    total_win_pips = test_df.loc[win_mask, pips_col].sum()
    total_lose_pips = test_df.loc[lose_mask, pips_col].sum()
    net_pips = test_df.loc[trade_mask, pips_col].sum()

    # リスクリワードレシオ
    avg_win = test_df.loc[win_mask, pips_col].mean() if win_count > 0 else 0
    avg_lose = abs(test_df.loc[lose_mask, pips_col].mean()) if lose_count > 0 else 0
    rr_ratio = avg_win / avg_lose if avg_lose > 0 else np.nan

    # 買い・売り・様子見マスク
    buy_mask = test_df[pred_col] == 1
    sell_mask = test_df[pred_col] == -1
    hold_mask = test_df[pred_col] == 0
    hold_count = hold_mask.sum()

    # 買いで勝ち（48本以内に利益決着）、負け（同損切決着）、未決着
    buy_win = ((test_df[pips_col] == profit_pips) & buy_mask).sum()
    buy_lose = ((test_df[pips_col] == -loss_pips) & buy_mask).sum()
    buy_undecided = buy_mask & ~((test_df[pips_col] == profit_pips) | (test_df[pips_col] == -loss_pips))
    buy_undecided_win = (test_df[pips_col][buy_undecided] > 0).sum()
    buy_undecided_lose = (test_df[pips_col][buy_undecided] < 0).sum()

    # 売りで勝ち（48本以内に利益決着）、負け（同損切決着）、未決着
    sell_win = ((test_df[pips_col] == profit_pips) & sell_mask).sum()
    sell_lose = ((test_df[pips_col] == -loss_pips) & sell_mask).sum()
    sell_undecided = sell_mask & ~((test_df[pips_col] == profit_pips) | (test_df[pips_col] == -loss_pips))
    sell_undecided_win = (test_df[pips_col][sell_undecided] > 0).sum()
    sell_undecided_lose = (test_df[pips_col][sell_undecided] < 0).sum()

    # バックテストに使用したローソク足の本数
    candle_count = len(test_df)

    # 損益曲線
    equity_curve = test_df[pips_col][trade_mask].cumsum().values

    # 最大ドローダウン
    if len(equity_curve) > 0:
        cummax = np.maximum.accumulate(equity_curve)
        drawdowns = cummax - equity_curve
        max_drawdown = np.max(drawdowns)
    else:
        max_drawdown = 0

    # 最大連勝・最大連敗
    def max_streak(series, cond_func):
        streak = max_streak = 0
        for v in series:
            if cond_func(v):
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0
        return max_streak

    max_consecutive_wins = max_streak(test_df[pips_col][trade_mask], lambda x: x > 0)
    max_consecutive_losses = max_streak(test_df[pips_col][trade_mask], lambda x: x < 0)

    # 結果表示
    print("【バックテスト集計結果】")
    print(f"1. 総トレード数: {total_trades}")
    print(f"2. 勝ち数: {win_count}")
    print(f"3. 負け数: {lose_count}")
    print(f"4. 勝率: {win_rate:.2%}")
    print(f"5. 総獲得PIPS: {total_win_pips:.2f}")
    print(f"6. 総損失PIPS: {total_lose_pips:.2f}")
    print(f"7. 最終的な獲得PIPS: {net_pips:.2f}")
    print(f"8. リスクリワードレシオ: {rr_ratio:.2f}")
    print(f"9. 買いで利益になった数（48本以内決着）: {buy_win}")
    print(f"10. 買いで損失になった数（48本以内決着）: {buy_lose}")
    print(f"11. 買いで未決着（勝ち／負け）: {buy_undecided_win}/{buy_undecided_lose}")
    print(f"12. 売りで利益になった数（48本以内決着）: {sell_win}")
    print(f"13. 売りで損失になった数（48本以内決着）: {sell_lose}")
    print(f"14. 売りで未決着（勝ち／負け）: {sell_undecided_win}/{sell_undecided_lose}")
    print(f"15. 様子見数: {hold_count}")
    print(f"16. 損益曲線（グラフ表示）")
    print(f"17. 最大ドローダウン（PIPS）: {max_drawdown:.2f}")
    print(f"18. 最大連勝数: {max_consecutive_wins}")
    print(f"19. 最大連敗数: {max_consecutive_losses}")
    print(f"20. バックテストに使用したローソク足の本数: {candle_count}")

    # 損益曲線プロット
    plt.figure(figsize=(10,4))
    plt.plot(equity_curve, label='Equity Curve')
    plt.title("損益曲線")
    plt.xlabel("Trade #")
    plt.ylabel("PIPS")
    plt.grid(True)
    plt.legend()
    plt.show()

    # 結果を辞書で返すことも可能
    return {
        "total_trades": total_trades,
        "win_count": win_count,
        "lose_count": lose_count,
        "win_rate": win_rate,
        "total_win_pips": total_win_pips,
        "total_lose_pips": total_lose_pips,
        "net_pips": net_pips,
        "rr_ratio": rr_ratio,
        "buy_win": buy_win,
        "buy_lose": buy_lose,
        "buy_undecided_win": buy_undecided_win,
        "buy_undecided_lose": buy_undecided_lose,
        "sell_win": sell_win,
        "sell_lose": sell_lose,
        "sell_undecided_win": sell_undecided_win,
        "sell_undecided_lose": sell_undecided_lose,
        "hold_count": hold_count,
        "equity_curve": equity_curve,
        "max_drawdown": max_drawdown,
        "max_consecutive_wins": max_consecutive_wins,
        "max_consecutive_losses": max_consecutive_losses,
        "candle_count": candle_count
    }

import pandas as pd

if __name__ == "__main__":
    # テスト用のダミーデータ作成
    np.random.seed(42)
    n = 1000
    test_df = pd.DataFrame({
        'pred': np.random.choice([1, -1, 0], size=n, p=[0.4, 0.4, 0.2]),
        'backtest_pips': np.random.choice([13, -16, 5, -5, 0], size=n, p=[0.3, 0.3, 0.2, 0.1, 0.1])
    })

    # バックテスト分析実行
    analyze_backtest(test_df)
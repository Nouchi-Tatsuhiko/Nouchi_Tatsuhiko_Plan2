import numpy as np
import talib

# =======================
# 特徴量自動生成パイプライン（省略せず記載）
# =======================

def add_talib_features(df):
    df['ATR_14'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['RSI_21'] = talib.RSI(df['Close'], timeperiod=21)
    macd, macd_signal, macd_hist = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'] = macd
    df['MACD_SIGNAL'] = macd_signal
    df['MACD_HIST'] = macd_hist
    df['EMA_14'] = talib.EMA(df['Close'], timeperiod=14)
    df['EMA_21'] = talib.EMA(df['Close'], timeperiod=21)
    df['SMA_14'] = talib.SMA(df['Close'], timeperiod=14)
    df['SMA_42'] = talib.SMA(df['Close'], timeperiod=42)
    df['SMA_252'] = talib.SMA(df['Close'], timeperiod=252)
    df['WILLR_14'] = talib.WILLR(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['ADX_14'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['ADX_21'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=21)
    df['APO'] = talib.APO(df['Close'], fastperiod=12, slowperiod=26, matype=0)
    df['AROONOSC_14'] = talib.AROONOSC(df['High'], df['Low'], timeperiod=14)
    df['AROONOSC_21'] = talib.AROONOSC(df['High'], df['Low'], timeperiod=21)
    df['CCI_14'] = talib.CCI(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['STDDEV_14'] = talib.STDDEV(df['Close'], timeperiod=14)
    df['STDDEV_21'] = talib.STDDEV(df['Close'], timeperiod=21)
    df['SAR'] = talib.SAR(df['High'], df['Low'], acceleration=0.02, maximum=0.2)
    df['SAREXT'] = talib.SAREXT(df['High'], df['Low'], startvalue=0, offsetonreverse=0,
                                accelerationinitlong=0.02, accelerationlong=0.02, accelerationmaxlong=0.2,
                                accelerationinitshort=0.02, accelerationshort=0.02, accelerationmaxshort=0.2)
    df['KAMA_14'] = talib.KAMA(df['Close'], timeperiod=14)
    df['KAMA_21'] = talib.KAMA(df['Close'], timeperiod=21)
    df['PPO'] = talib.PPO(df['Close'], fastperiod=12, slowperiod=26, matype=0)
    df['TRIX_30'] = talib.TRIX(df['Close'], timeperiod=30)
    df['TYP_PRICE'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['AVG_PRICE'] = (df['High'] + df['Low'] + df['Close'] + df['Open']) / 4
    df['ADXR_14'] = talib.ADXR(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['CMO_14'] = talib.CMO(df['Close'], timeperiod=14)
    df['ROCP_14'] = talib.ROCP(df['Close'], timeperiod=14)
    df['HT_TRENDLINE'] = talib.HT_TRENDLINE(df['Close'])

    for n in [1, 2, 3]:
        upper, middle, lower = talib.BBANDS(df['Close'], timeperiod=20, nbdevup=n, nbdevdn=n, matype=0)
        df[f'BB_upper_{n}'] = upper
        df[f'BB_middle_{n}'] = middle
        df[f'BB_lower_{n}'] = lower
    return df

def add_candle_features(df):
    df['Body_Upper'] = df[['Open', 'Close']].max(axis=1)
    df['Body_Lower'] = df[['Open', 'Close']].min(axis=1)
    df['PRICE_DIFF'] = abs(df['Close'] - df['Open'])
    df['Body_Upper_Top5Avg_235'] = df['Body_Upper'].rolling(window=235, min_periods=235).apply(lambda x: np.mean(np.sort(x)[-5:]), raw=True)
    df['Body_Lower_Bottom5Avg_235'] = df['Body_Lower'].rolling(window=235, min_periods=235).apply(lambda x: np.mean(np.sort(x)[:5]), raw=True)
    return df

def add_diff_features(df, base_cols, rolling_windows=[48]):
    for window in rolling_windows:
        for col in base_cols:
            df[f'{col}_mean_diff'] = df[col] - df[col].rolling(window=window).mean()
            df[f'{col}_max_diff'] = df[col].rolling(window=window).max() - df[col]
            df[f'{col}_min_diff'] = df[col].rolling(window=window).min() - df[col]
            df[f'{col}_mean_ratio'] = df[col] / (df[col].rolling(window=window).mean() + 1e-9)
            df[f'{col}_max_ratio'] = df[col] / (df[col].rolling(window=window).max() + 1e-9)
            df[f'{col}_min_ratio'] = df[col] / (df[col].rolling(window=window).min() + 1e-9)
    return df

def add_ma_slope_trend(df):
    for ma in ['SMA_14', 'SMA_42', 'SMA_252']:
        df[f'{ma}_Slope'] = np.sign(df[ma].diff())
    def ma_trend(row):
        s14, s42, s252 = row['SMA_14_Slope'], row['SMA_42_Slope'], row['SMA_252_Slope']
        if s14 == 1 and s42 == 1 and s252 == 1:
            return 1
        elif s14 == -1 and s42 == -1 and s252 == -1:
            return -1
        else:
            return 0
    df['MA_Trend'] = df.apply(ma_trend, axis=1)
    def ma_relationship(row):
        if row['SMA_14'] > row['SMA_42'] > row['SMA_252']:
            return 1
        elif row['SMA_14'] < row['SMA_42'] < row['SMA_252']:
            return -1
        else:
            return 0
    df['MA_Relationship'] = df.apply(ma_relationship, axis=1)
    return df

def add_cdl_patterns(df):
    df['CDL_2CROWS'] = talib.CDL2CROWS(df['Open'], df['High'], df['Low'], df['Close'])
    df['CDL_3INSIDE'] = talib.CDL3INSIDE(df['Open'], df['High'], df['Low'], df['Close'])
    df['CDL_3LINESTRIKE'] = talib.CDL3LINESTRIKE(df['Open'], df['High'], df['Low'], df['Close'])
    df['CDL_3OUTSIDE'] = talib.CDL3OUTSIDE(df['Open'], df['High'], df['Low'], df['Close'])
    df['CDL_BREAKAWAY'] = talib.CDLBREAKAWAY(df['Open'], df['High'], df['Low'], df['Close'])
    df['CDL_DRAGONFLY'] = talib.CDLDRAGONFLYDOJI(df['Open'], df['High'], df['Low'], df['Close'])
    df['CDL_EVENINGSTAR'] = talib.CDLEVENINGSTAR(df['Open'], df['High'], df['Low'], df['Close'])
    df['CDL_GRAVESTONE'] = talib.CDLGRAVESTONEDOJI(df['Open'], df['High'], df['Low'], df['Close'])
    df['CDL_HIKKAKE'] = talib.CDLHIKKAKE(df['Open'], df['High'], df['Low'], df['Close'])
    df['CDL_LADDER'] = talib.CDLLADDERBOTTOM(df['Open'], df['High'], df['Low'], df['Close'])
    df['CDL_MATHOLD'] = talib.CDLMATHOLD(df['Open'], df['High'], df['Low'], df['Close'])
    return df

def add_price_lag_features(df, max_lag=5):
    for i in range(1, max_lag+1):
        for col in ['Open', 'High', 'Low', 'Close']:
            df[f'{col}_lag{i}'] = df[col].shift(i)
    return df

def feature_engineering_pipeline(df):
    df = add_talib_features(df)
    df = add_candle_features(df)
    base_cols = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'ATR_14', 'PRICE_DIFF', 'RSI_21', 'MACD', 'MACD_SIGNAL', 'MACD_HIST',
        'EMA_14', 'EMA_21', 'SMA_14', 'SMA_42', 'SMA_252'
    ]
    df = add_diff_features(df, base_cols)
    df = add_ma_slope_trend(df)
    df = add_cdl_patterns(df)
    df = add_price_lag_features(df, max_lag=5)
    return df
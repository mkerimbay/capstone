import pandas as pd
import numpy as np
import datetime
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tqdm import tqdm

# Function to calculate Welles Wilder's EMA
def wwma(values, n):
    return values.ewm(alpha=1 / n, adjust=False).mean()

# Function to calculate percentage change
def percentage(f, t):
    return round((t/f - 1) * 100, 2)

# Function to calculate upper wick percentage
def upwick(d):
    return percentage(max(d['open'], d['close']), d['high'])

# Function to calculate lower wick percentage
def downwick(d):
    return percentage(min(d['open'], d['close']), d['high'])

# Function to calculate Volume Weighted Moving Average (VWMA)
def vwma(prices, volumes, window):
    typical_price = (prices * volumes).sum() / volumes.sum()
    vwma_values = pd.Series(index=prices.index)
    for i in range(len(prices) - window + 1):
        vwma_values.iloc[i + window - 1] = (prices.iloc[i:i+window] * volumes.iloc[i:i+window]).sum() / volumes.iloc[i:i+window].sum()
    return vwma_values

# Function to generate features
def generate_features(df):
    sma_periods = [10, 20, 50, 100, 200]
    for n in sma_periods:
        df['tmp'] = df['close'].rolling(window=n).mean()
        df[f'f_sma_{n}'] = percentage(df['tmp'], df['close'])

    ema_periods = [10, 20, 50, 100, 200]
    for n in ema_periods:
        df['tmp'] = df['close'].ewm(span=n, adjust=False).mean()
        df[f'f_ema_{n}'] = percentage(df['tmp'], df['close'])

    atr_periods = [7, 10, 14, 20, 30]
    df['tr0'] = abs(df['high'] - df['low'])
    df['tr1'] = abs(df['high'] - df['close'].shift())
    df['tr2'] = abs(df['low'] - df['close'].shift())
    df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
    df.drop(['tr0', 'tr1','tr2'], axis=1, inplace=True)
    for n in atr_periods:
        df['tmp'] = wwma(df['tr'], n)
        df[f'f_atr_{n}'] = percentage(df['tmp'], df['tr'])
    df.drop('tr', axis=1, inplace=True)

    df['f_upwick'] = df.apply(upwick, axis=1)
    df['f_body'] = percentage(df['open'], df['close'])
    df['f_downwick'] = df.apply(downwick, axis=1)
    df['f_range'] = percentage(df['low'], df['high'])

    vma_periods = [10, 20, 50]
    for n in vma_periods:
        df['tmp'] = df['volume'].rolling(window=n).mean()
        df[f'f_vma_{n}'] = percentage(df['tmp'], df['volume'])

    vwma_periods = [10, 20, 50]
    for n in vwma_periods:
        df['tmp'] = vwma(df['close'], df['volume'], window=n)
        df[f'f_vwma_{n}'] = percentage(df['tmp'], df['close'])

    df.drop('tmp', axis=1, inplace=True)

    for col in df.columns:
        if not col.startswith('f'):
            continue    
        for i in range(1, 4):
            df[f"{col}_{i}"] = df[col].shift(i)
        
    df['target_clf'] = (df['close'] >= df['close'].shift(1)).astype(int)
    df['target_clf'] = df['target_clf'].shift(-1)
    df['target_reg'] = percentage(df['close'].shift(1), df['close'])
    df['target_reg'] = df['target_reg'].shift(-1)
    
    return df

# Function to get list of files in a directory
def file_list(source):
    res = []
    for root, dirnames, filenames in os.walk(source):
        for filename in filenames:
            if filename.endswith('.csv'):
                res.append(os.path.join(root, filename))
    return res

def get_all_df(main_directory_crypto):
    X_train = pd.DataFrame()
    X_test = pd.DataFrame()
    all_files = file_list(main_directory_crypto)
    for file in tqdm(all_files):
        symbol = file.split('/')[-1].split('.')[0]
        if 'DOWN' in symbol:
            continue
        df = pd.read_csv(file)
        df['symbol'] = symbol
        df.sort_values('time', inplace=True)
        
        shape_before = df.shape[0]
        df.drop_duplicates(inplace=True)
        shape_after = df.shape[0] 
        
        diff = shape_before - shape_after
        if diff > 0:
            print(f"{symbol} - {diff} duplicates")
        df = generate_features(df)
        
        X_test = pd.concat([X_test, df.tail(1)])
        
        df.dropna(inplace=True)
        if not df.empty:
            X_train = pd.concat([X_train, df])
    
    return X_train, X_test

def transform(df):
    for col in feature_cols + ['target_reg']:
        df[col] = df[col].apply(lambda x: np.log1p(abs(x)) * (-1 if x < 0 else 1))
    return df

main_directory_crypto = 'klines/crypto/'
df_train, df_test = get_all_df(main_directory_crypto)

# Feature columns
feature_cols = [col for col in df_train.columns if col.startswith("f_")]

# MinMax Scale
scaler = MinMaxScaler(feature_range=(-1, 1), clip=True)
df_train[feature_cols] = scaler.fit_transform(df_train[feature_cols])
df_test[feature_cols] = scaler.fit_transform(df_test[feature_cols])

# Train
df_train.reset_index(drop=True, inplace=True)

# Test
max_time = df_test['time'].mode()[0]
df_test = df_test[df_test['time'] == max_time].reset_index(drop=True)
ind = df_test[feature_cols].dropna().index.values
df_test = df_test.iloc[ind]

# Val
yesterday = (datetime.datetime.strptime(max_time, '%Y-%m-%d') - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
df_yesterday = df_train[df_train['time'] == yesterday]
df_train = df_train.drop(index=df_yesterday.index)

for df in [df_train, df_yesterday, df_test]:
    df = transform(df)

X = df_train[feature_cols]
y = df_train[['target_clf', 'target_reg']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape input data for LSTM
X_train = np.reshape(X_train.values, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test.values, (X_test.shape[0], 1, X_test.shape[1]))

# Build LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train['target_clf'], epochs=100, batch_size=32, validation_data=(X_test, y_test['target_clf']))

# Evaluate the model
_, accuracy = model.evaluate(X_test, y_test['target_clf'])
print(f'Accuracy: {accuracy * 100:.2f}%')

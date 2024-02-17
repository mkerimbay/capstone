# Klines collection Day-0

from my_binance import MyBinance
import pandas as pd
import time
import os
from my_logger import logger

b = MyBinance()


def get_info():
    # Get all pairs, whether they are actively being traded or in a break
    info = b.s_exchange_info()
    cols = ['symbol', 'status', 'baseAsset', 'quoteAsset']
    df = pd.DataFrame(info['symbols'])[cols]
    df = df[~df['symbol'].str.endswith('FDUSD')]    # Get rid of pairs ending with FDUSDT
    df = df[(df['symbol'].str.endswith('USDT')) | (df['symbol'].str.endswith('BTC'))]   # Take only pairs ending with USDT or BTC
    return df


def main():
    info = get_info()
    if info.empty:
        logger.info('Empty info file, exit.')
        return
    for symbol in info.symbol.values[:2]:
        try:
            time.sleep(2)
            directory = f'klines/crypto/{symbol[0]}/'
            fname = f'{symbol}.csv'

            # Create all directories in the path if they don't exist
            os.makedirs(os.path.dirname(directory), exist_ok=True)

            df = b.s_klines(symbol, '1D', 1000)  # time,symbol,open,high,low,close,volume
            df.drop('symbol', axis=1, inplace=True)  # no need for symbol column, reflected in file name
            df.drop(df.tail(1).index, inplace=True)  # delete last incomplete daily row
            df.to_csv(directory + fname, index=False)
            start = df['time'].min().strftime('%Y/%m/%d')
            end = df['time'].max().strftime('%Y/%m/%d')
            msg = f"{symbol}: {df.shape[0]} rows, from ({start}) - ({end})"
            logger.info(msg)
        except Exception as e:
            logger.error(e)
            continue


if __name__ == "__main__":
    main()

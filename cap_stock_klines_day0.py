# stock Klines collection Day-0

import os
import pandas as pd
import yfinance as yf
import time
from my_logger import logger

nasdaq = pd.read_csv('data/nasdaq.csv')


def main():
    for symbol in nasdaq.Symbol.values[:2]:
        try:
            time.sleep(1)
            directory = f'klines/stock/{symbol[0]}/'
            fname = f'{symbol}.csv'

            # Create all directories in the path if they don't exist
            os.makedirs(os.path.dirname(directory), exist_ok=True)

            ticker = yf.Ticker(symbol)
            df = ticker.history(period="max")
            df.reset_index(inplace=True)
            df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
            df.columns = [x.lower() for x in df.columns]
            cols = ['date', 'open', 'high', 'low', 'close', 'volume']
            df = df[cols]

            df.to_csv(directory + fname, index=False)
            start = df['date'].min()
            end = df['date'].max()
            msg = f"{symbol}: {df.shape[0]} rows, from ({start}) - ({end})"
            logger.info(msg)
        except Exception as e:
            logger.error(e)
            continue


if __name__ == "__main__":
    main()

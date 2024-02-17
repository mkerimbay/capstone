# stock Klines collection Day-1

import os
import pandas as pd
import yfinance as yf
import time
from my_logger import logger

nasdaq = pd.read_csv('data/nasdaq.csv')


def read_df(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        logger.error(e)
        return pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume'])


def main():
    for symbol in nasdaq.Symbol.values[:2]:
        try:
            time.sleep(2)

            # new data
            ticker = yf.Ticker(symbol)
            new = ticker.history()  # get latest 1 month of data
            new.reset_index(inplace=True)
            new['Date'] = new['Date'].dt.strftime('%Y-%m-%d')
            new.columns = [x.lower() for x in new.columns]
            cols = ['date', 'open', 'high', 'low', 'close', 'volume']
            new = new[cols]

            if new.empty:
                continue

            # existing data
            directory = f'klines/stock/{symbol[0]}/'
            fname = f'{symbol}.csv'

            # Create all directories in the path if they don't exist
            os.makedirs(os.path.dirname(directory), exist_ok=True)

            old = read_df(directory + fname)
            old_shape = old.shape[0]

            df = pd.concat([old, new])
            df.drop_duplicates(inplace=True)
            new_shape = df.shape[0]

            df.to_csv(directory + fname, index=False)
            msg = f"{symbol}: {new_shape - old_shape} rows added"
            logger.info(msg)

        except Exception as e:
            logger.error(e)
            continue


if __name__ == "__main__":
    main()

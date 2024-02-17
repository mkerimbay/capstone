# Day-1
# collect klines and append to csv file new rows

from my_binance import MyBinance
import pandas as pd
import time
import _config_spot
import os
from my_logger import logger

b = MyBinance()
precisions = pd.read_csv('data/spot_precisions.csv')
main_directory = 'klines/crypto/'


def read_df(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        logger.error(e)
        return pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume'])


def main():
    for symbol in precisions.symbol.values:
        try:
            time.sleep(1)
            new = b.s_klines(symbol, '1D', 500)  # get latest data
            new.drop('symbol', axis=1, inplace=True)  # no need for symbol column, reflected in file name
            new['time'] = new['time'].dt.strftime('%Y-%m-%d')
            new.drop(new.tail(1).index, inplace=True)  # delete last incomplete data

            if new.empty:  # example: newly added pair, df could be empty
                continue

            directory = f'klines/crypto/{symbol[0]}/'
            fname = f'{symbol}.csv'
            old = read_df(directory + fname)

            df = pd.concat([old, new])
            df.drop_duplicates(inplace=True)

            # Create all directories in the path if they don't exist
            os.makedirs(os.path.dirname(directory), exist_ok=True)

            df.to_csv(directory + fname, index=False)
            msg = f'{symbol}: new kline added'
            logger.info(msg)

        except Exception as e:
            msg = f'{symbol}: {e}'
            logger.error(msg)
            continue


if __name__ == "__main__":
    main()


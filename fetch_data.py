import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

# Define the stock symbols you want to fetch data for
stocks = ["AAPL", "MSFT", "GOOG"]

# Function to fetch data from Yahoo Finance and store it in a CSV file
def fetch_data_and_store(stocks):
    # Create a directory to store CSV files if it doesn't exist
    if not os.path.exists("stock_data"):
        os.makedirs("stock_data")

    # Fetch data for each stock
    for stock_symbol in stocks:
        stock_data = yf.download(stock_symbol, start=datetime.now() - timedelta(days=365), end=datetime.now())
        # Create a CSV file name based on the stock symbol
        csv_file_name = f"stock_data/{stock_symbol}_data.csv"
        # If file exists, append new data; otherwise, create a new file
        if os.path.exists(csv_file_name):
            existing_data = pd.read_csv(csv_file_name, index_col=0)
            updated_data = existing_data.append(stock_data)
            # Remove duplicate rows
            updated_data.drop_duplicates(inplace=True)
            updated_data.to_csv(csv_file_name)
        else:
            stock_data.to_csv(csv_file_name)

# Run the function to fetch data and store it
fetch_data_and_store(stocks)

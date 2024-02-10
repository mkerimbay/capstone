import os
import yfinance as yf

# Function to fetch stock data for a company and save it to the corresponding folder
def fetch_and_save_stock_data(company_symbol, folder_path):
    # Fetch stock data
    stock_data = yf.download(company_symbol, start="2023-01-01", end="2023-12-31")
    
    # Save stock data to a CSV file
    csv_file_path = os.path.join(folder_path, f"{company_symbol}.csv")
    stock_data.to_csv(csv_file_path)
    print(f"Stock data saved for {company_symbol} at {csv_file_path}")

# Create directories A-Z in the stock_data folder
stock_data_folder = "stock_data"
if not os.path.exists(stock_data_folder):
    os.makedirs(stock_data_folder)

for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    letter_folder_path = os.path.join(stock_data_folder, letter)
    if not os.path.exists(letter_folder_path):
        os.makedirs(letter_folder_path)

# List of companies
companies = ["AAPL", "GOOGL", "MSFT", "AMZN", "FB", "TSLA", "NFLX", "NVDA", "INTC", "CSCO"]

# Fetch and save stock data for each company in the corresponding folder
for company in companies:
    first_letter = company[0].upper()  # Get the first letter of the company symbol
    folder_path = os.path.join(stock_data_folder, first_letter)
    fetch_and_save_stock_data(company, folder_path)

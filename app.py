from flask import Flask, request, jsonify, render_template
import openai
import requests
import re

app = Flask(__name__)
openai.api_key = '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'
alpha_vantage_api_key = 'P79S99OWD2YX7418'

@app.route('/', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        data = request.get_json()
        user_input = data['query']
        
        # Preprocess user input
        processed_input = preprocess_input(user_input)
        
        # Generate bot response using NLP
        bot_response = generate_bot_response(processed_input)
        
        # Combine NLP response with stock market data
        if bot_response:
            stock_symbol = extract_stock_symbol(processed_input)
            stock_data = fetch_stock_data(stock_symbol)
            bot_response += f" {stock_data}"
        
        return jsonify({'response': bot_response})
    
    result = request.args.get("result")
    return render_template("index.html", result=result)

def preprocess_input(user_input):
    """
    Preprocess user input to remove unnecessary characters and normalize text.
    """
    # Remove special characters and extra whitespace
    processed_input = re.sub(r'[^\w\s]', '', user_input)
    # Convert text to lowercase
    processed_input = processed_input.lower()
    
    return processed_input

def extract_stock_symbol(user_input):
    """
    Extract stock symbol from user input.
    """
    # You need to implement logic to extract stock symbol from user input
    # For example, you can split the input and extract the relevant part
    stock_symbol = user_input.upper()  # Example: Assume user input is the stock symbol itself
    
    return stock_symbol

def fetch_stock_data(stock_symbol):
    """
    Fetch stock market data using Alpha Vantage API based on the stock symbol.
    """
    # Make a request to Alpha Vantage API to fetch stock market data
    url = f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={stock_symbol}&apikey={alpha_vantage_api_key}'
    response = requests.get(url)
    
    # Extract relevant data from the response (You may need to parse JSON response here)
    stock_data = response.json()
    
    return stock_data

def generate_bot_response(user_input):
    """
    Generate bot response based on user input using the OpenAI API.
    
    Args:
    - user_input (str): The user's query or prompt.
    
    Returns:
    - str: The generated bot response, or None if an error occurs.
    """
    try:
        # Make a request to the OpenAI API to generate the response
        response = openai.Completion.create(
            engine="gpt-3.5-turbo-instruct",  # Choose the language model
            prompt=user_input,  # The user's query or prompt
            max_tokens=150  # Maximum number of tokens for the response
        )

        # Check if the response was successful
        if response and 'choices' in response and response['choices']:
            # Extract the generated text completion
            bot_response = response['choices'][0]['text'].strip()
            return bot_response
        else:
            print("Error: Unable to generate bot response")
            return None
    except Exception as e:
        print("Error:", e)
        return None

if __name__ == '__main__':
    app.run(debug=True)

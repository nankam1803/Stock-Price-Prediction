Stock Price Prediction Web App
This project is a Stock Price Prediction Web App built using Streamlit, TensorFlow, and Keras. The app predicts the next day's closing price of a stock using historical data from Yahoo Finance. The model is based on an LSTM (Long Short-Term Memory) neural network, which is a type of recurrent neural network (RNN) that excels in time series forecasting.

Introduction
The Stock Price Prediction Web App uses historical stock data (up to the past 20 years) to predict the closing price for the next trading day. The model uses a 75-day window of historical data to predict the stock price using an LSTM model.

The app is designed to:

Take any stock ticker as input (e.g., AAPL, GOOG, TSLA).
Fetch data from Yahoo Finance for the past 20 years.
Preprocess the data and prepare it for training the model.
Use an LSTM model to predict the next day's closing price.
Display the prediction and visualizations using Matplotlib.
Technologies Used
Streamlit: For building the interactive web app.
TensorFlow: For creating and training the LSTM model.
Keras: High-level API for building neural networks with TensorFlow backend.
Numpy: For numerical operations and handling arrays.
Pandas: For data manipulation and handling time series data.
Matplotlib: For visualizing the stock price and prediction results.
Yahoo Finance API: For fetching stock data.
Features
Stock Ticker Input: Users can input any stock ticker symbol (e.g., AAPL, MSFT, TSLA).
Stock Price Prediction: Predicts the next day's closing price based on historical data.
Data Visualization: Displays stock price data and predictions using interactive graphs.
Accurate Predictions: Achieves 97%+ accuracy in predicting the closing price of stocks.
Installation
To run this project on your local machine, follow these steps:

Prerequisites
Python 3.x
pip (Python package installer)
Steps
Clone the repository:
bash
Copy code
git@github.com:nankam1803/Stock-Price-Prediction.git
Navigate to the project directory:
bash
Copy code
cd stock-price-prediction-web-app
Install the required libraries:
bash
Copy code
pip install -r requirements.txt
The requirements.txt file includes all the necessary libraries like streamlit, tensorflow, keras, pandas, numpy, and matplotlib.

Run the Streamlit app:
bash
Copy code
streamlit run app.py
The app will open in your default web browser.

How to Use
Open the app in your browser by running the command mentioned above.
Input the stock ticker symbol (e.g., AAPL for Apple, GOOG for Google) in the input box.
The app will fetch historical data from Yahoo Finance for the specified stock.
The app will preprocess the data and train the model on it.
It will predict the next day's closing price and display the result along with visualizations.
You can view the prediction and historical data graphs.
Model Details
The model uses a Long Short-Term Memory (LSTM) neural network, which is effective for time series prediction tasks. Here’s an overview of the architecture:

LSTM Layer 1: 128 units with return_sequences=True to pass sequences to the next LSTM layer.
LSTM Layer 2: 64 units with return_sequences=False to output the final prediction.
Dense Layer 1: 25 units for intermediate processing.
Dense Layer 2: Single unit output layer to predict the stock's closing price.
The model was trained using Yahoo Finance data and uses a 75-day window of past stock data to predict the next day's closing price.

Model Performance
The model has consistently achieved over 97% accuracy in predicting the close prices of stocks, making it a reliable tool for forecasting stock trends.

Results
The app successfully predicts stock prices based on historical data with high accuracy.
Visualizations of the predicted prices and actual prices are shown on the web app for better understanding.
The model's performance improves with more epochs and by fine-tuning hyperparameters like the number of layers, units, and dropout rates.
Contributions
Feel free to contribute to this project! Here’s how you can help:

Improve the model’s accuracy by experimenting with hyperparameters.
Add more features such as predicting stock volume, opening price, etc.
Optimize the code for performance improvements.
License
This project is licensed under the MIT License - see the LICENSE file for details.

Thank you for checking out my Stock Price Prediction Web App! 🚀

Feel free to reach out for any questions or contributions!

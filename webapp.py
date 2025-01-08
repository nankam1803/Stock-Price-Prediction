import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

st.title("Stock Price Predictor")
stock = st.text_input("Enter the Stock ID (Ticker)", "GOOGL")

from datetime import datetime
#Setting Start and End Date window for the data to be pulled
end = datetime.now()
start = datetime(end.year-20, end.month, end.day)

stock_data = yf.download(stock, start, end)
st.subheader("Stock Data")
st.write(stock_data)


# Define the plotting function
def plotgraph(figsize, values, input_data, index, color='black'):
    fig = plt.figure(figsize=figsize)
    plt.plot(input_data.index, stock_data['Close'], label='Actual Close Price', color='blue') 
    plt.plot(input_data.index, values, label=f'Moving Average ({index} days)', color=color)  
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    return fig

st.subheader('Actual Close Price v/s Moving Average for 75 days')

# Calculate the moving average for 75 days and plot it
stock_data['MA for 75 days'] = stock_data['Close'].rolling(75).mean()
st.pyplot(plotgraph((15, 5), stock_data['MA for 75 days'], stock_data, 75, color='orange'))

# Pre-processing the data
model = load_model("Stock_Price_Prediction_LSTMModel.keras")
splitting_len = int(len(stock_data)*0.7)
x_test = pd.DataFrame(stock_data.Close[splitting_len:])
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(x_test)

# Splitting the data
x_data = []
y_data = []

for i in range (75, len(scaled_data)):
    x_data.append(scaled_data[i-75:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

predictions = model.predict(x_data)

inv_pre = scaler.inverse_transform(predictions)
inv_ytest = scaler.inverse_transform(y_data)

plotting_data = pd.DataFrame(
    {
    'Actual Close Price': inv_ytest.reshape(-1),
    'Predictions': inv_pre.reshape(-1)
    },
    index = stock_data.index[splitting_len + 75:]
)

st.subheader('Actual Close Price v/s Predicted Close Price')
st.write(plotting_data)

st.subheader('Stock Graph')
fig = plt.figure(figsize=(15,6))
plt.plot(pd.concat([stock_data.Close[:splitting_len+100],plotting_data], axis=0))
plt.legend(["Original Data", "Test data", "Predicted Test data"])
st.pyplot(fig)
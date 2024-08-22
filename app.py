import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st
from datetime import datetime, timedelta


st.set_page_config(
    page_title="Stock Trend Prediction App",
    page_icon=":chart_with_upwards_trend:"
)
st.title("Stock Trend Prediction App")

end_date = datetime.today().date()
start_date = end_date - timedelta(days=365*10)
end_date = end_date.strftime('%Y-%m-%d')
start_date = start_date.strftime('%Y-%m-%d')

user_input = st.text_input("Enter Stock Ticker", 'AAPL')
df = yf.download(user_input, start=start_date, end=end_date)

# Describe the data
st.subheader("Data from last 10 years")
st.write(df.describe())

# Visualization
st.subheader("Closing Price vs Time Chart")
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader("Closing Price vs Time Chart with 100MA & 200MA")
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100, 'orange', label="100MA")
plt.plot(ma200, 'green', label="200MA")
plt.plot(df.Close)
plt.legend()
st.pyplot(fig)

# Splitting Data into Training and Testing
# first 70% data is used for training, last 30% data is used for testing
data_train = pd.DataFrame(df["Close"][0:int(len(df)*.70)])
data_test = pd.DataFrame(df["Close"][int(len(df)*.70):int(len(df))])

# Scaling data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_train_array = scaler.fit_transform(data_train)

#Loading model
model = load_model("keras_model.h5")

# Testing Data
last_100_days = data_train.tail(100)
final_df = pd.concat([last_100_days, data_test], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])


x_test = np.array(x_test)
y_test = np.array(y_test)


y_predict = model.predict(x_test)


scale_factor = 1/(scaler.scale_[0])
y_predict = y_predict*scale_factor
y_test = y_test*scale_factor


# Final Predicted Graph
st.subheader("Predictions vs Original")
fig1 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'blue', label="Original Trend Price")
plt.plot(y_predict, 'red', label="Predicted Trend Price")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()
st.pyplot(fig1)

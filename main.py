import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error

data = pd.read_csv('/content/coin_Bitcoin.csv')

data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data= data['2017':'2021']

plt.figure(figsize=(10, 6))
plt.plot(data['Close'], label='Bitcoin Close Price')
plt.title('Bitcoin Historical Prices')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

#LSTM üçün təlim və sınaq verilənlərinin hazırlanması
def create_sequences(data, seq_length):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i:i+seq_length, 0])
        y.append(data[i+seq_length, 0])
    return np.array(x), np.array(y)

seq_length = 60
x, y = create_sequences(scaled_data, seq_length)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test))


predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions.reshape(-1, 1))


plt.figure(figsize=(14,5))
plt.plot(data.index[-len(y_test):], 
scaler.inverse_transform(y_test.reshape(-1,1)), 
color='blue', label='Həqiqi qiymət')
plt.plot(data.index[-len(predictions):], 
predictions, color='red', label='Proqnozlaşdırılan qiymət')
plt.title('Bitcoinin proqronlaşdırılması')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

mae = mean_absolute_error(scaler.inverse_transform(y_test.reshape(-1,1)), predictions)
rmse = np.sqrt(mean_squared_error(scaler.inverse_transform(y_test.reshape(-1,1)), predictions))
print(f'MAE: {mae}, RMSE: {rmse}')


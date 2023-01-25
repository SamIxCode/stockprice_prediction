import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, LSTM
import yfinance as yf
import datetime as dt
from sys import argv


class StockPredictor:
    def __init__(
        self,
        company: str,
        test_start_date: datetime,
        test_end_date: datetime,
    ):
        self.company = company
        self.start_date = dt.datetime(2010, 1, 1)
        self.end_date = dt.datetime(2020, 1, 1)
        self.prediction_days = 60
        self.ohlcv = None
        self.data = None
        self.scaler = None
        self.x_train = []
        self.y_train = []
        self.model = None
        self.test_data = []
        self.x_test = []
        self.predicted_prices = None
        self.test_start_date = test_start_date
        self.test_end_date = test_end_date

    def get_data(self):
        self.ohlcv = yf.download(self.company, start=self.start_date, end=self.end_date)
        self.data = pd.DataFrame(self.ohlcv)

    def prepare_data(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = self.scaler.fit_transform(
            self.data["Close"].values.reshape(-1, 1)
        )

        for x in range(self.prediction_days, len(scaled_data)):
            self.x_train.append(scaled_data[x - self.prediction_days : x, 0])
            self.y_train.append(scaled_data[x, 0])

        self.x_train, self.y_train = np.array(self.x_train), np.array(self.y_train)
        self.x_train = np.reshape(
            self.x_train, (self.x_train.shape[0], self.x_train.shape[1], 1)
        )

    def create_model(self):
        self.model = Sequential()
        self.model.add(
            LSTM(
                units=50, return_sequences=True, input_shape=(self.x_train.shape[1], 1)
            )
        )
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=50, return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=50))
        self.model.add(Dense(units=1))  # prediction of the next close
        self.model.compile(optimizer="adam", loss="mean_squared_error")

    def train_model(self):
        self.model.fit(self.x_train, self.y_train, epochs=25, batch_size=32)

    def test_model(self):

        self.test_data = yf.download(
            self.company, start=self.test_start_date, end=self.test_end_date
        )
        self.test_data = pd.DataFrame(self.test_data)
        total_dataset = pd.concat((self.data["Close"], self.test_data["Close"]), axis=0)
        model_input = total_dataset[
            len(total_dataset) - len(self.test_data) - self.prediction_days :
        ].values
        model_input = model_input.reshape(-1, 1)
        model_input = self.scaler.transform(model_input)

        for x in range(self.prediction_days, len(model_input)):
            self.x_test.append(model_input[x - self.prediction_days : x, 0])
        self.x_test = np.array(self.x_test)
        self.x_test = np.reshape(
            self.x_test, (self.x_test.shape[0], self.x_test.shape[1], 1)
        )

        self.predicted_prices = self.model.predict(self.x_test)
        self.predicted_prices = self.scaler.inverse_transform(self.predicted_prices)

    def plot_prediction(self):
        plt.plot(
            self.test_data["Close"].values,
            color="black",
            label=f"actual {self.company} price",
        )
        plt.plot(
            self.predicted_prices,
            color="green",
            label=f"predicted {self.company} price",
        )
        plt.title(f"{self.company} share price prediction")
        plt.xlabel("Time")
        plt.ylabel(f"{self.company} share price")
        plt.legend()
        plt.show()


if __name__ == "__main__":


    symbol = input("enter a stockmarket symbol. default is 'META': ")
    if len(symbol) == 0:
        symbol = "META"

    start_date = input("Specify the start day (in DD/MM/YYYY) default 01/01/2020:  ")
    if len(start_date) == 0:
        start_date = "01/01/2020"
    test_start_date = dt.datetime.strptime(start_date, "%d/%m/%Y")

    end_date = input("Specify the end day (in DD/MM/YYYY) default 01/01/2021:  ")
    if len(start_date) == 0:
        start_date = "01/01/2021"
    test_end_date = dt.datetime.strptime(start_date, "%d/%m/%Y")

    predictor = StockPredictor(
        company=symbol, test_start_date=test_start_date, test_end_date=test_end_date
    )
    predictor.get_data()
    predictor.prepare_data()
    predictor.create_model()
    predictor.train_model()
    predictor.test_model()
    predictor.plot_prediction()

Stock Price Predictor
This is a Python app that uses a LSTM (Long Short-Term Memory) neural network to predict stock prices for a specific company. The app takes in a company name, a start date for training, and an end date for testing. It then uses the yfinance library to download historical stock data for the specified company and dates, and trains a model on that data. The model is then used to make predictions on the test data. The app also includes a method to plot the actual and predicted prices.

Requirements
Python 3.6 or higher
yfinance library
TensorFlow 2.x
pandas
numpy
matplotlib
sklearn
Usage
Create an instance of the StockPredictor class and pass in the following parameters:

company: the name of the company as a string
test_start_date: the start date for testing as a datetime object
test_end_date: the end date for testing as a datetime object
Copy code
predictor = StockPredictor("META", dt.datetime(2012, 1, 1), dt.datetime(2020, 1, 1))
Then call the following methods in order:

get_data()
prepare_data()
create_model()
train_model()
test_model()
plot_prediction()
Copy code
predictor.get_data()
predictor.prepare_data()
predictor.create_model()
predictor.train_model()
predictor.test_model()
predictor.plot_prediction()
The plot_prediction() method will display a graph of the actual and predicted stock prices.

Customization
You can change the number of days to predict by changing the "prediction_days" attribute in the init method.
You can change the start date for training by changing the "start_date" attribute in the init method.
You can change the end date for training by changing the "end_date" attribute in the init method.
Note
This is a demonstration model and should not be used for actual stock trading decisions.
This model only predicts the closing price on the day after the last day of the test data.
This model only works for the stock of the company you entered.
Please let me know if you have any question about the code or the readme.

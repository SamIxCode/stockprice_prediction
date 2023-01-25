**Stock Price Predictor**

This is a Python program that uses a LSTM (Long Short-Term Memory) neural network to predict stock prices for a specific company. The app takes in a company name, a start date for training, and an end date for testing. It then uses the yfinance library to download historical stock data for the specified company and dates, and trains a model on that data. The model is then used to make predictions on the test data. The app also includes a method to plot the actual and predicted prices.

Requirements
Python 3.6 or higher
yfinance library
TensorFlow 2.x
pandas
numpy
matplotlib
sklearn
**Usage**
For simple usage all you need to run this program, is to run it with
`python3 main.py`
Then input desired ticker, start and end date for the prediction. If none values are inputed, the default values will be used.


Customization
You can change the number of days to predict by changing the "prediction_days" attribute in the init method.
You can change the start date for training by changing the "start_date" attribute in the init method.
You can change the end date for training by changing the "end_date" attribute in the init method.
Note
This is a demonstration model and should not be used for actual stock trading decisions.
This model only predicts the closing price on the day after the last day of the test data.
This model only works for the stock of the company you entered.
Please let me know if you have any question about the code or the readme.

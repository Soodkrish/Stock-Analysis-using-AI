# Stock-Analysis-using-AI-and-ML


# Infosys Stock Price Prediction Using XGBoost

This repository contains code to predict the closing prices of Infosys Limited stocks using the XGBoost regression model. The data used for this prediction is obtained from historical stock prices.

## Table of Contents

1. [Introduction](#)
2. [Dependencies](#)
3. [Data](#)
4. [Code Explanation](#)
5. [Results](#)
6. [Conclusion](#)


## Introduction

Stock price prediction is a crucial task in financial markets. Accurate predictions can help investors make informed decisions. This project demonstrates a simple approach to predict the closing prices of Infosys Limited stocks using the XGBoost machine learning algorithm.

## Dependencies

- Python 3.x
- pandas
- matplotlib
- xgboost

Install the required packages using the following command:

```sh
pip install pandas matplotlib xgboost
```

## Data

The data used in this project is the historical stock prices of Infosys Limited, stored in a CSV file named `INFY.csv`. This file should contain columns like 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', etc.

## Code Explanation

1. **Loading the Data:**

    ```python
    import pandas as pd

    d = pd.read_csv('C:\\Users\\KRISH\\OneDrive\\Documents\\EXCEL Data pool\\INFY.csv')
    ```

2. **Visualizing the Closing Prices:**

    ```python
    import matplotlib.pyplot as plt

    plt.title('Closing Price history of Infosys')
    plt.plot(d['Close'])
    plt.ylabel('Closing Prices')
    plt.show()
    ```

3. **Splitting the Data:**

    The data is split into training and testing sets. 80% of the data is used for training, and 20% for testing.

    ```python
    train = d.iloc[:int(.80*len(d)), :]
    test = d.iloc[int(.80*len(d)):, :]
    ```

4. **Defining Features and Target Variable:**

    The features used are 'Open' and 'Volume', and the target variable is 'Close'.

    ```python
    features = ['Open', 'Volume']
    target = ['Close']
    ```

5. **Training the Model:**

    An XGBoost regressor is used for the prediction.

    ```python
    import xgboost as xgb

    m = xgb.XGBRegressor()
    m.fit(train[features], train[target])
    ```

6. **Making Predictions:**

    Predictions are made on the test set.

    ```python
    predictions = m.predict(test[features])
    ```

7. **Evaluating the Model:**

    The accuracy of the model is printed.

    ```python
    accuracy = m.score(test[features], test[target])
    print('Accuracy:', accuracy)
    ```

8. **Plotting Predictions:**

    The actual and predicted closing prices are plotted for comparison.

    ```python
    plt.plot(d['Close'], label="Close Price")
    plt.plot(test[target].index, predictions, label="Predicted")
    plt.legend()
    plt.show()
    ```

        
## Results

The model achieves an accuracy of approximately 96% in predicting the closing prices of Infosys Limited stocks. The plot comparing actual and predicted values shows how well the model performs.

## Conclusion

This project demonstrates a simple yet effective approach to stock price prediction using machine learning. While the model shows good accuracy, further improvements can be made by incorporating more features, tuning hyperparameters, and using more sophisticated models.


Feel free to contribute to this project by opening issues or submitting pull requests.

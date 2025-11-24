# Implement-Moving-Average-model-in-Python.

### AIM:
To implement Moving Average Model and Exponential smoothing Using Python.

### ALGORITHM:
1. Import necessary libraries
2. Read the electricity time series data from a CSV file,Display the shape and the first 20 rows of
the dataset
3. Set the figure size for plots
4. Suppress warnings
5. Plot the first 50 values of the 'Value' column
6. Perform rolling average transformation with a window size of 5
7. Display the first 10 values of the rolling mean
8. Perform rolling average transformation with a window size of 10
9. Create a new figure for plotting,Plot the original data and fitted value
10. Show the plot
11. Also perform exponential smoothing and plot the graph

### PROGRAM:

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings('ignore')

data = pd.read_csv('results.csv')

data['date'] = pd.to_datetime(data['date'])

data['home_score'] = pd.to_numeric(data['home_score'], errors='coerce')
ts = data['home_score'].dropna().reset_index(drop=True)

print("Shape of the dataset:", data.shape)
print("First 20 rows of the dataset:")
print(data.head(20))

plt.rcParams['figure.figsize'] = [10, 6]

plt.plot(ts[:50])
plt.title('First 50 Values of Home Score')
plt.xlabel('Index')
plt.ylabel('Home Score')
plt.grid(True)
plt.show()

rolling_mean_5 = ts.rolling(window=5).mean()
print("First 10 values of the rolling mean (window size 5):")
print(rolling_mean_5.head(10))

rolling_mean_10 = ts.rolling(window=10).mean()

plt.figure()
plt.plot(ts, label='Original Data')
plt.plot(rolling_mean_10, label='Rolling Mean (window=10)', color='orange')
plt.title('Original Home Scores and Rolling Mean (window=10)')
plt.xlabel('Index')
plt.ylabel('Home Score')
plt.legend()
plt.grid(True)
plt.show()

alpha = 0.3
exp_smooth = ts.ewm(alpha=alpha).mean()

plt.figure()
plt.plot(ts, label='Original Data')
plt.plot(exp_smooth, label='Exponential Smoothing', color='red')
plt.title('Original Home Scores and Exponential Smoothing')
plt.xlabel('Index')
plt.ylabel('Home Score')
plt.legend()
plt.grid(True)
plt.show()

q = 1
ma_model = ARIMA(ts, order=(0, 0, q))
ma_model_fit = ma_model.fit()

print(ma_model_fit.summary())

forecast = ma_model_fit.forecast(steps=10)

plt.figure()
plt.plot(ts, label='Original Data')
plt.plot(range(len(ts), len(ts) + 10), forecast, label='MA Forecast', color='green', marker='o')
plt.title('Original Home Scores and Moving Average Forecast')
plt.xlabel('Index')
plt.ylabel('Home Score')
plt.legend()
plt.grid(True)
plt.show()

print("Forecasted values for the next 10 points:")
print(forecast)
```

### OUTPUT:
## data:
<img width="800" height="771" alt="image" src="https://github.com/user-attachments/assets/030f20d4-b6f3-42d8-80bf-e497ab34c4ba" />

<img width="465" height="211" alt="image" src="https://github.com/user-attachments/assets/6078eaeb-6e5f-4f7f-8db9-0547797c157b" />


## Price Today
<img width="884" height="539" alt="image" src="https://github.com/user-attachments/assets/2c7a381a-0de6-4572-8e3f-3262ec0f8667" />

## Original Data and Rolling Mean
<img width="866" height="532" alt="image" src="https://github.com/user-attachments/assets/82c0c955-376c-47a4-8b99-a507e98f391e" />

## Exponential Smoothing
<img width="875" height="540" alt="image" src="https://github.com/user-attachments/assets/ea9f45ee-718b-4ee6-bb9e-d1e40eeeb715" />


## Moving average
<img width="876" height="534" alt="image" src="https://github.com/user-attachments/assets/6bb360df-6b36-44b1-be6a-4f3229b1a7c8" />

<img width="666" height="410" alt="image" src="https://github.com/user-attachments/assets/1d743fa2-bd37-4ecb-a61b-376410ea1e8b" />


### RESULT:
Thus we have successfully implemented the Moving Average Model and Exponential smoothing using python.

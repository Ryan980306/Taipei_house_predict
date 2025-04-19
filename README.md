Taipei House Price Prediction
This project is a simple house price prediction tool for Taipei City, using historical transaction data to analyze trends and forecast future prices.

Features
Data visualization of historical house prices.
Simple trend analysis using linear regression.
Forecast future house prices based on historical trends.
Requirements
Python 3.x
pandas
matplotlib
numpy
Installation
Clone the repository or download the source code.

Install the required Python packages using pip:

pip install pandas matplotlib numpy
Ensure you have the Taipei_house.csv file in the same directory as the script.

Usage
Run the script to perform the analysis and see the visualizations:

python
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

# 讀取資料
data = pd.read_csv('Taipei_house.csv')

# 確保交易日期是日期格式
data['交易日期'] = pd.to_datetime(data['交易日期'])

# 按日期排序
data = data.sort_values('交易日期')

# 計算每月的平均房價
data.set_index('交易日期', inplace=True)
monthly_avg_price = data['總價'].resample('M').mean()

# 繪製歷史房價趨勢
plt.figure(figsize=(10, 6))
plt.plot(monthly_avg_price, label='Historical Average Price')
plt.title('Historical Monthly Average House Price')
plt.xlabel('Date')
plt.ylabel('Average Price')
plt.legend()
plt.show()

# 簡單的趨勢線擬合（線性回歸）
# 準備數據
monthly_avg_price = monthly_avg_price.dropna()  # 移除缺失值
X = np.arange(len(monthly_avg_price)).reshape(-1, 1)  # 時間步長
y = monthly_avg_price.values

# 使用numpy進行線性回歸
A = np.vstack([X.flatten(), np.ones(len(X))]).T
m, c = np.linalg.lstsq(A, y, rcond=None)[0]

# 繪製趨勢線
plt.figure(figsize=(10, 6))
plt.plot(monthly_avg_price.index, y, label='Historical Average Price')
plt.plot(monthly_avg_price.index, m*X + c, 'r', label='Trend Line')
plt.title('House Price Trend with Linear Fit')
plt.xlabel('Date')
plt.ylabel('Average Price')
plt.legend()
plt.show()

# 簡單的未來趨勢預測
future_steps = 12  # 預測未來12個月
future_X = np.arange(len(monthly_avg_price), len(monthly_avg_price) + future_steps).reshape(-1, 1)
future_y = m * future_X + c

# 繪製未來趨勢預測
plt.figure(figsize=(10, 6))
plt.plot(monthly_avg_price.index, y, label='Historical Average Price')
plt.plot(pd.date_range(monthly_avg_price.index[-1], periods=future_steps, freq='M'), future_y, 'r--', label='Forecast')
plt.title('House Price Forecast')
plt.xlabel('Date')
plt.ylabel('Average Price')
plt.legend()
plt.show()
The script will output several plots:

Historical monthly average house prices.
Trend line fitted to the historical data.
Forecast of future house prices.
Data
The data used in this project is assumed to be in a CSV file named Taipei_house.csv, containing at least the following columns:

交易日期 (Transaction Date)
總價 (Total Price)
License
This project is licensed under the MIT License.

Acknowledgments
This project uses basic Python libraries for data analysis and visualization.
Inspired by the need to understand housing market trends in Taipei City.

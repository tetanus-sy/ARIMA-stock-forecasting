import pandas_datareader as pdr
import matplotlib.pyplot as plt 
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
import warnings
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import yfinance as yf
from pandas.plotting import autocorrelation_plot

warnings.filterwarnings("ignore")

# Download the information of closing prices from Yahoo Finance
ticker = "NTDOY"  # 示例股票代码——任天堂
start_date = "2024-01-01"  # 示例开始日期
end_date = "2024-07-01"  # 示例结束日期

# 创建 Ticker 对象
ticker_data = yf.Ticker(ticker)
ticker_data = yf.Ticker(ticker)

# 使用 history() 方法获取数据
data = ticker_data.history(start=start_date, end=end_date)['Close']

# 画出训练集合股价图像
plt.rcParams["figure.figsize"] = (10, 7)
plt.style.use('ggplot')  # 使用 ggplot 样式

data.plot()
plt.title('任天堂股票收盘价', fontproperties='KaiTi', fontsize=16)  # 楷体、16
plt.ylabel('价格（美元）', fontproperties='KaiTi', fontsize=14)  
plt.legend(['收盘价'], prop={'family': 'KaiTi', 'size': 12})  
plt.show()

# 绘制自相关图
autocorrelation_plot(data)
plt.title('任天堂股票收盘价自相关图',fontproperties='KaiTi', fontsize=16)
plt.ylabel('自相关系数',fontproperties='KaiTi',fontsize=14)
plt.show()

# Splitting data into train and test set
split = int(len(data)*0.90)
train_set, test_set = data[:split], data[split:]

plt.figure(figsize=(10, 5))  # 调整图像大小
plt.title('任天堂股票收盘价', fontproperties='KaiTi',fontsize=16)  # 使用中文楷体
plt.xlabel('日期', fontproperties='KaiTi',fontsize=14)
plt.ylabel('价格（美元）', fontproperties='KaiTi',fontsize=14)
plt.plot(train_set, 'green', label='训练集')
plt.plot(test_set, 'red', label='测试集')
plt.xlim(train_set.index.min(), test_set.index.max())  # 只显示有数据的部分
plt.legend(['收盘价'], prop={'family': 'KaiTi', 'size': 12}) # 确保图例使用中文字体
plt.show()

# 优化ARIMA模型
aic_p = []
bic_p = []
params = []  # 用于存储(p, d, q)参数
    
p = range(0,6) # [0,1,2,3,4,5]
d = range(0,2) # [0,1]
q = range(0,6) # [0,1,2,3,4,5]

# 三个循环遍历所有(p, d, q)
for i in p:
    for j in d:
        for k in q:
            model = ARIMA(train_set, order=(i,j,k)) # define ARIMA model
            model_fit = model.fit() # fit the model
            aic_temp = model_fit.aic # get aic score
            bic_temp = model_fit.bic # get bic score
            aic_p.append(aic_temp) # append aic score
            bic_p.append(bic_temp) # append bic score
            params.append((i, j, k))  # 存储参数
            print(f'ARIMA model p={i}, d={j}, q={k} AIC={aic_temp}, BIC={bic_temp}')  # 输出所有模型的AIC和BIC评分

# 找到AIC和BIC最小值的索引
min_aic_index = aic_p.index(min(aic_p))
min_bic_index = bic_p.index(min(bic_p))

# 绘制AIC图像
plt.figure(figsize=(12, 6))
plt.plot(range(len(aic_p)), aic_p, color='red', marker='o')
plt.annotate(f'({params[min_aic_index][0]},{params[min_aic_index][1]},{params[min_aic_index][2]})', 
             (min_aic_index, aic_p[min_aic_index]), textcoords="offset points", xytext=(0,10), ha='center')
plt.title('AIC优化模型图', fontproperties='KaiTi',fontsize=16)
plt.xlabel('模型参数', fontproperties='KaiTi',fontsize=14)
plt.ylabel('AIC评分', fontproperties='KaiTi',fontsize=14)
plt.show()

# 绘制BIC图像
plt.figure(figsize=(12, 6))
plt.plot(range(len(bic_p)), bic_p, color='blue', marker='o')
plt.annotate(f'({params[min_bic_index][0]},{params[min_bic_index][1]},{params[min_bic_index][2]})', 
             (min_bic_index, bic_p[min_bic_index]), textcoords="offset points", xytext=(0,10), ha='center')
plt.title('BIC优化模型图', fontproperties='KaiTi',fontsize=16)
plt.xlabel('模型参数', fontproperties='KaiTi',fontsize=14)
plt.ylabel('BIC评分', fontproperties='KaiTi',fontsize=14)
plt.show()
    
# Fitting ARIMA model
model = ARIMA(train_set, order=(0,1,0))
model_fit_0 = model.fit()
model_fit_0

# 预测
past = train_set.tolist()
predictions = []
    
for i in range(len(test_set)):
    model = ARIMA(past, order=(0,1,0)) 
    model_fit = model.fit(start_params=model_fit_0.params)
    forecast_results = model_fit.forecast() 
        
    pred = forecast_results 
    predictions.append(pred) 
    past.append(test_set[i]) 
    
# calculate mse
# Start of Selection
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

mse = mean_squared_error(test_set, predictions)
mae = mean_absolute_error(test_set, predictions)
mape = mean_absolute_percentage_error(test_set, predictions)

print('Test MSE: {mse}'.format(mse=mse))
print('Test MAE: {mae}'.format(mae=mae))
print('Test MAPE: {mape:.2%}'.format(mape=mape))

# Plot forecasted and actual values
plt.plot(test_set, color='green', label='实际值')
plt.plot(test_set.index, predictions, color='red', label='预测值')
plt.title('任天堂股票预测对比', fontproperties='KaiTi', fontsize=16)
plt.xlabel('日期', fontproperties='KaiTi', fontsize=14)
plt.ylabel('价格（美元）', fontproperties='KaiTi', fontsize=12)
plt.legend(prop={'family': 'KaiTi', 'size': 12})   # 确保图例使用中文字体
plt.show()
# ARIMA-stock-forecasting
这个代码以任天堂股票为例，基于ARIMA模型进行预测股票的收盘价，并给出相关信息。具体关于ARIMA的概念等请自行网上搜索学习

## 股票获取代码
本项目包含两种情况，但是对于国内用户都需要有VPN。如果是国外用户或者对自己VPN很有信心，可以直接运行**ARIMA股价预测**，但是如果想要离线运行或者减少运行时间可以先用**股票获取代码**生成一个Excel本地表格。
此表格中第一列是**日期**，第二列是**收盘价**，然后再去运行代码**任天堂半年股价预测**
对于代码中的股票代号和起始终止日期均可以修改
```
# 获取任天堂的股票数据
ticker = 'NTDOY'

# 获取2024年1月1日到2024年7月1日的数据
data_2024 = yf.download(ticker, start='2020-01-01', end='2024-07-01')
```

## 股价预测
本项目中代码**任天堂半年股价预测**和代码**任天堂四年股价预测**唯一的区别就是第**13**行其读取的文件不同
```
# 从本地Excel文件读取任天堂股票收盘价数据
data = pd.read_excel('C:/Users/29823/.cursor-tutor/任天堂股票上半年收盘价.xlsx', index_col='日期', parse_dates=True)
```
```
data = pd.read_excel('C:/Users/29823/.cursor-tutor/任天堂股票四年收盘价.xlsx', index_col='日期', parse_dates=True)
```
在预测代码中，生成图像的标题，横纵坐标以及图例都可以自定义修改
```
plt.title('任天堂股票收盘价', fontproperties='KaiTi', fontsize=16)  # 楷体、16
plt.ylabel('价格（美元）', fontproperties='KaiTi', fontsize=14)  
plt.legend(['收盘价'], prop={'family': 'KaiTi', 'size': 12})  
plt.show()
```

import yfinance as yf
import pandas as pd

# 获取任天堂的股票数据
ticker = 'NTDOY'

# 获取2024年1月1日到2024年7月1日的数据
data_2024 = yf.download(ticker, start='2020-01-01', end='2024-07-01')

# 创建一个DataFrame，包含日期和收盘价
df = pd.DataFrame({
    '日期': data_2024.index.tz_localize(None).strftime('%Y-%m-%d'),  # 转换为字符串格式
    '收盘价': data_2024['Close'].values.flatten()  # 将收盘价转换为一维数组
})

# 将DataFrame保存为Excel文件
df.to_excel('任天堂股票三年收盘价.xlsx', index=False)

print("数据已保存到 '任天堂股票三年收盘价.xlsx'")

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor

# 读取数据
data = pd.read_csv("./data/train.csv")

## 数据集处理
# 删除缺失行
missing_values_per_row = data.isnull().sum(axis=1)
rows_to_drop = missing_values_per_row[missing_values_per_row > 0].index
data = data.drop(rows_to_drop)

# 处理data数据，汇总每日销售量
data = data[["date", "sales"]]
sales_summary = data.groupby("date")["sales"].sum().round(1)
sales_summary = sales_summary.reset_index(name="sales")

# 使用z-scores处理异常值
z_scores = stats.zscore(sales_summary["sales"])
outliters = (abs(z_scores) > 2).astype(int)
df_clean = sales_summary[outliters == 0]
data = df_clean

# 划分训练集，测试集
train_data = data.iloc[:int(0.8*len(data)), :]
train_data['date'] = pd.to_datetime(train_data['date'])
train_data.set_index('date', inplace = True)
test_data = data.iloc[int(0.8*len(data)):, :]
test_data['date'] = pd.to_datetime(test_data['date'])
test_data.set_index('date', inplace = True)
# 定义Holt-Winters模型
model_1 = sm.tsa.ExponentialSmoothing(
    train_data["sales"],
    trend='add', 
    seasonal='add', 
    seasonal_periods=12
).fit(
    smoothing_level=0.5,
)
forecast_1 = model_1.forecast(steps=len(test_data)).round(1)
forecast_index_1 = test_data.index
forecast_1.index = test_data.index

# 定义SARIMA模型
p, d, q = 1, 1, 1
P, D, Q, S = 1, 1, 1, 12
model_2 = SARIMAX(train_data["sales"], order=(p, d, q), seasonal_order=(P, D, Q, S)).fit()
forecast_2 = model_2.forecast(steps=len(test_data)).round(1)
forecast_index_2 = test_data.index
forecast_2.index = test_data.index

# 定义随机森林模型
_train_data = train_data.copy()
_train_data['year'] = train_data.index.year
_train_data['month'] = train_data.index.month
_train_data['day'] = train_data.index.day
_train_data['weekday'] = train_data.index.weekday
_test_data = test_data.copy()
_test_data['month'] = test_data.index.month
_test_data['day'] = test_data.index.day
_test_data['weekday'] = test_data.index.weekday
_test_data['year'] = test_data.index.year
X_train = _train_data[['year', 'month', 'day', 'weekday']]
y_train = _train_data['sales']
X_test = _test_data[['year', 'month', 'day', 'weekday']]
model_3 = RandomForestRegressor(n_estimators=50, random_state=42).fit(
    X_train,
    y_train,
)
forecast_3 = model_3.predict(X_test).round(1)
forecast_3 = pd.DataFrame({
    "date": test_data.index,
    "sales": forecast_3,
})
forecast_3['date'] = pd.to_datetime(forecast_3['date'])
forecast_3.set_index('date', inplace = True)

plt.style.use('dark_background')
plt.figure(figsize=(18, 6))
# 绘制测试集的数据
plt.plot(test_data.index, test_data, label='Actual', color='#1f77b4')
# 绘制预测结果
plt.plot(forecast_1.index, forecast_1, label='Holt-Winters', color='#2ca02c', linestyle='--')
plt.plot(forecast_2.index, forecast_2, label='SARIMA', color='#d62728', linestyle='--')
plt.plot(forecast_3.index, forecast_3, label='Random-Forest', color='#ff7f0e', linestyle='--')
# 添加标题和标签
plt.title('Comparison of Predictions from Each Model')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
# 保存图表
plt.savefig("./tables/result.png", dpi = 400)
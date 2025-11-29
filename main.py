import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats

'''
train.csv
    -The training data, comprising time series of features store_nbr, family, and onpromotion as well as the target sales.
    培训数据包括功能 store_nbr、 系列 、 推广的时间序列以及目标销售 。

    -store_nbr identifies the store at which the products are sold.
    store_nbr 标识销售产品的门店。

    -family identifies the type of product sold.
    家庭指的是销售产品的类型。

    -sales gives the total sales for a product family at a particular store at a given date. Fractional values are possible since products can be sold in fractional units (1.5 kg of cheese, for instance, as opposed to 1 bag of chips).
    销售额表示某一特定门店在特定日期内某一产品系列的总销售额。由于产品可以以分数单位销售（例如 1.5 公斤奶酪，而非一包薯片）而存在分数价值。

    -onpromotion gives the total number of items in a product family that were being promoted at a store at a given date.
    OnPromotion 表示在某一日期，商店中产品系列中正在推广的商品总数。

test.csv
    -The test data, having the same features as the training data. You will predict the target sales for the dates in this file.
    测试数据，具有与训练数据相同的特征。您将预测该文件中日期的目标销售额。

    -The dates in the test data are for the 15 days after the last date in the training data.
    测试数据中的日期是训练数据最后一个日期后的15天。

sample_submission.csv  <b1001></b1001>
    -A sample submission file in the correct format.
    格式正确、提交样本文件。

stores.csv
    -Store metadata, including city, state, type, and cluster.
    存储元数据，包括城市 、 州 、 类型和集群 。

    -cluster is a grouping of similar stores.
    Cluster 是一组类似商店的集合。

oil.csv
    -Daily oil price. Includes values during both the train and test data timeframes. (Ecuador is an oil-dependent country and it's economical health is highly vulnerable to shocks in oil prices.)
    每日油价。包括训练和测试数据时间段内的数值。（厄瓜多尔是一个依赖石油的国家，其经济健康极易受到油价冲击的影响。）

holidays_events.csv  <b1001></b1001>
    -Holidays and Events, with metadata
    节日和活动，含元数据

    -NOTE: Pay special attention to the transferred column. A holiday that is transferred officially falls on that calendar day, but was moved to another date by the government. A transferred day is more like a normal day than a holiday. To find the day that it was actually celebrated, look for the corresponding row where type is Transfer. For example, the holiday Independencia de Guayaquil was transferred from 2012-10-09 to 2012-10-12, which means it was celebrated on 2012-10-12. Days that are type Bridge are extra days that are added to a holiday (e.g., to extend the break across a long weekend). These are frequently made up by the type Work Day which is a day not normally scheduled for work (e.g., Saturday) that is meant to payback the Bridge.
    注意：特别注意转移的那一栏。被转移的假期官方落在该日，但政府已将日期移至其他日期。调动日更像是普通一天，而不是假期。要查找实际庆祝日期，请查找类型为 Transfer 的对应行。例如，瓜亚基尔独立节日从 2012-10-09 移至 2012-10-12，也就是说该节日在 2012-10-12 庆祝。桥梁类型的天是加到假期的额外天数（例如，延长假期至长周末）。这些通常由“工作日”补偿，工作日通常是不安排工作的日子（例如周六），旨在偿还桥梁。
    Additional holidays are days added a regular calendar holiday, for example, as typically happens around Christmas (making Christmas Eve a holiday).
    额外假期是加在普通日历假日的日期，比如通常在圣诞节前后发生（使平安夜成为假日）。

Additional Notes  附加注释
    -Wages in the public sector are paid every two weeks on the 15 th and on the last day of the month. Supermarket sales could be affected by this.
    公共部门的工资每两周在15日和月底支付一次。超市销售可能会受到影响。

    -A magnitude 7.8 earthquake struck Ecuador on April 16, 2016. People rallied in relief efforts donating water and other first need products which greatly affected supermarket sales for several weeks after the earthquake.
    2016年4月16日，厄瓜多尔发生了一场7.8级地震。人们积极参与救援，捐赠水和其他急需物资，地震后数周内极大地影响了超市销售。
'''

# 读取数据集
folder_path = "./data"
filenames = os.listdir(folder_path)
csv_filenames = [f for f in filenames if f.endswith(".csv")]
df_dict = {}
for filename in filenames:
    filename_without_extention, _ = os.path.splitext(filename)
    file_path = os.path.join(folder_path, filename)
    df_dict[filename_without_extention] = pd.read_csv(file_path)

# 数据集处理
for filename, df in df_dict.items():
    # 删除缺失行
    missing_values_per_row = df.isnull().sum(axis=1)
    rows_to_drop = missing_values_per_row[missing_values_per_row > 0].index
    df = df.drop(rows_to_drop)

    # # 输出缺失行占比
    # missing_values_per_row = df.isnull().sum(axis=1)
    # missing_rows = (missing_values_per_row > 0).sum()
    # total_rows = df.shape[0]
    # ratio = missing_rows / total_rows
    # print(f"{filename}: {ratio}")

#使用z-scores处理异常值
df_train = df_dict["train"]
z_scores = stats.zscore(df_train["sales"])
filtered_train = df_train[np.abs(z_scores) < 3]
print(f"origin: {len(df_train)}\nfiltered: {len(filtered_train)}\n")


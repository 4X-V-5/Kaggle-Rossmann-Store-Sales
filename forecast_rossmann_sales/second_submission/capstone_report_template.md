# 机器学习纳米学位
##毕业项目
 许梓晟

2019年1月22日


## I. 问题的定义

### 项目概述

​	Rossmann在7个欧洲国家运营着300家药店，此项目的目的为利用机器学习模型，预测未来6个星期1115家Rossmann商店的销售额[^1]。将以前销售得到的历史数据（包括商店的信息以及销售额）进行一定处理后，作为训练集对模型进行训练，让模型从过往的销售中学习到一定规律，从而对未来的销售额进行大概甚至准确的预测。

​	此项目所对应的数据集可以在 Kaggle Rossmann Sales 竞赛页面下载[^2] ，Kaggle提供的数据文件有3个，分别为：

> train.csv - 包括销售额的历史数据（训练数据）
>
> test.csv - 不包括销售额的历史数据（需要进行预测的数据）
>
> sample_submission.csv - 包含正确提交格式的提交文件样本
>
> store.csv - 商店的一些补充信息



### 问题陈述
​	此项目属于机器学习中有监督学习中的回归问题，同时属于时序类预测问题。主要的问题有如何利用store.csv文件，数据清理及一些异常值处理，特征工程以及寻找模型的最优参数。

​	如何利用store.csv提供的补充信息，商店的补充信息是和商店编号（位于train.csv和test.csv的Store列）是一一对应的，可以用Pandas库中的merge方法进行合并。

​	数据清理和异常值处理，主要是清理train.csv中对预测销售额没有帮助的Customer列，可以用Pandas库中DataFrame的Drop方法进行对Customer列的丢弃。并且，由于train.csv和test.csv中Date列的格式为字符串格式而非datetime格式，机器不能理解其中的时间信息，需要利用Pandas库的to_datetime方法将Date列转换为datetime格式。有些时候商店开门，但没有顾客，销售额为0是正常的，但由于之前删除了Customer列，这样的数据可能会造成对模型的干扰，经过计数，这样的数据只有54条，直接删除后对训练集大小的影响可以忽略不计，故将其直接删除。

​	特征工程，主要是将模型不能直接利用的Date列进行拆分，创造新的列，拆分为年、月、日，并删除原来的Date列排除共线性的干扰。由于数据集中有一些数据（StateHoliday, StoreType, Assortment和PromoInterval）属于无序的定类数据，各类别之间没有关联，如果直接进行标签编码（Label Encoding），模型可能会误以为这些类别之间是有大小关系的，预测准确率可能会降低。这一问题可以利用独热编码（One-hot Encoding）解决。当然，独热编码也有缺点：数据的维度大大增加了。在后续对test.csv进行预测的时候，由于train.csv中StateHoliday列有0, a, b, c四种取值，而test.csv中StateHoliday列只有0和a两种取值，会造成mismatch的情况，需要后续添加两列弥补缺失。

​	关于如何寻找模型的最优参数，我使用了随机搜索（Randomized Search）进行参数调整，Sci-kit learn库提供了随机搜索的实现，可供直接调用使用。由于网格搜索是穷举完给定参数范围内所有参数组合的可能取值，使用随机搜索对给定参数范围进行随机取值，通常比传统的网格搜索耗时少，但如果设定的迭代次数过少，可能没有办法找到给定参数范围内的最优参数。



### 评价指标
​	对模型评估的标准由 Kaggle Rossmann Store Sales 竞赛给出的均方根百分比误差 (Root Mean Suqare Percentage Error, RMSPE) 衡量[^3]，公式为

​								RMSPE = $\sqrt{\frac{1}{n} \sum_{i=1}^{n} (\frac{y_i - \hat{y_i}}{y_i})^2}$ 

​	其中 $y_i$ 表示一间商店一天的实际销售额，$\hat{y_i}$ 表示一间商店一天的预测销售额。

​	从公式可以看出，预测值与真实值越接近，RMSPE的值越低，因此，此项目的目标也即为降低RMSPE得分。




## II. 分析
### 数据的探索
​	经过上面问题陈述提及的数据处理过程，最终得到训练用的训练集，关于训练集的信息见下方表格



| 特征                           | 特征简介                                         | 数据类型 | 数量   |
| ------------------------------ | ------------------------------------------------ | -------- | ------ |
| Store                          | 每一个商店的唯一编号                             | int64    | 844338 |
| DayOfWeek                      | 代表星期几                                       | int64    | 844338 |
| Sales                          | 给定日期的销售额 (需要预测的项目)                | int64    | 844338 |
| Open                           | 表示商店是否营业：0 = 休息，1 = 营业             | int64    | 844338 |
| Promo                          | 表示那一天商店是否在促销                         | int64    | 844338 |
| SchoolHoliday                  | 表示 (商店, 日期) 是否受到公立学校休息的影响     | int64    | 844338 |
| CompetitionDistance            | 距离最近竞争对手商店的距离，以米为单位           | float64  | 844338 |
| CompetitionOpenSinceMonth      | 提供竞争对手商店开始营业的大致月份               | float64  | 844338 |
| CompetitionOpenSinceYear       | 提供竞争对手商店开始营业的大致年份               | float64  | 844338 |
| Promo2                         | 一些商店的连续促销：0 = 商店不参加，1 = 商店参加 | int64    | 844338 |
| Promo2SinceWeek                | 描述商店参加Promo2的周数                         | float64  | 844338 |
| Promo2SinceYear                | 描述商店参加Promo2的年份                         | float64  | 844338 |
| Day                            | 营业的日期                                       | int64    | 844338 |
| Month                          | 营业的月份                                       | int64    | 844338 |
| Year                           | 营业的年份                                       | int64    | 844338 |
| StateHoliday_0                 | 表示没有州假日                                   | uint8    | 844338 |
| StateHoliday_a                 | 表示州公众假日                                   | uint8    | 844338 |
| StateHoliday_b                 | 表示州复活节假日                                 | uint8    | 844338 |
| StateHoliday_c                 | 表示州圣诞节假日                                 | uint8    | 844338 |
| Store_a                        | a商店                                            | uint8    | 844338 |
| Store_b                        | b商店                                            | uint8    | 844338 |
| Store_c                        | c商店                                            | uint8    | 844338 |
| Store_d                        | d商店                                            | uint8    | 844338 |
| Assort_a                       | 基本分类级别                                     | uint8    | 844338 |
| Assort_b                       | 额外分类级别                                     | uint8    | 844338 |
| Assort_c                       | 拓展分类级别                                     | uint8    | 844338 |
| PromoInterval_Feb,May,Aug,Nov  | 每年2、5、8、11月开始新促销                      | uint8    | 844338 |
| PromoInterval_Jan,Apr,Jul,Oct  | 每年1、4、7、10月开始新促销                      | uint8    | 844338 |
| PromoInterval_Mar,Jun,Sept,Dec | 每年3、6、9、12月开始新促销                      | uint8    | 844338 |



### 探索性可视化
​	直观上来说，可能影响销售额的因素有:
1. DayOfWeek: 如果商店不进行休息，那么周末的销售额可能会比工作日的销售额多，人们有更多的时间去商店

2. Date: 特定的时间可能受节日促销等的影响，销售额提升

3. Promo: 进行促销的日子应当能推动顾客的购买商品，从而提高销售额

4. CompetitonDistance: 附近有竞争对手，可能销售额会有所影响

   首先选取一个数据较多的商店——1023号商店进行分析，以Sales列对DayOfWeek列作图，得到下图

![图1:1023号商店一周七天中销售额的分布](/Users/xuzisheng/Desktop/School_stuff/Udacity/机器学习工程师/forecast_rossmann_sales/second_submission/DayOfWeek.png)

​								图1：1023号商店一周七天中销售额的分布

​	*可以看到，除了星期日商家休息无销售额外，星期一销售额分布比较分散，其中位数比其余日子大，为7500左右，而其余日子对销售额没有太大影响，中位数都落在6000左右。*

​	以Sales列对Date列作图，得到下图

![图2:1023号商店所有日期中销售额的分布](/Users/xuzisheng/Desktop/School_stuff/Udacity/机器学习工程师/forecast_rossmann_sales/second_submission/Dates.png)

​								图2：1023号商店所有日期中销售额的分布

​	*可以看到，有趣的是，2014年和2015年的1月份左右有几天销售额是明显比其他日子高的，可能是由于新年假期的缘故。*

​	商店是否进行促销也可能对商店营业额造成影响，以Sales列对Promo列作图，得到下图

![图3:商店是否促销和销售额的关系](/Users/xuzisheng/Desktop/School_stuff/Udacity/机器学习工程师/forecast_rossmann_sales/second_submission/Promo.png)

​								图3：商店是否促销和销售额的关系

​	*可以看到，有促销的商店销售额的中位数比没有促销的商店高一点。由于已经剔除了商店不营业，销售额为0的情况，中位数不会受极小值影响。*

​	选取a类商店进行分析，对CompetitionDistance列进行groupby分组并取均值，以Sales列对处理后数据作图，得到下图

![图4：a商店竞争对手距离和销售额的关系](/Users/xuzisheng/Desktop/School_stuff/Udacity/机器学习工程师/forecast_rossmann_sales/second_submission/CompetitionDistance.png)

​								图4：a商店竞争对手距离和销售额的关系

​	*可以看到，似乎竞争对手距离对销售额的影响并不明显*



### 算法和技术
​	由于使用传统的决策树模型进行预测面临着准确率不高的问题，需要使用一种前沿的模型提高准确率。我计划采用在 Leaderboard 上被广泛使用的 XGBoost 算法对销售额进行预测。XGBoost (eXtreme Gradient Boosting) 是一种集成学习方法，是 Gradient Boost 算法的高效实现,其原理为：

**定义树模型**

​	XGBoost使用多个CART树（Classification And Regression Tree）对数据进行预测。假设模型含有K个CART树模型，用公式表示为：

​									$\hat{y} = \sum_{k=1}^{K} f_k(x_i), f_k \in F$	

​	式中$K$表示CART树的数量，$f_k$ 表示第$k$个CART树，$F$表示包含所有回归树的函数空间。

​	定义模型的目标函数为：

​									$Obj = \sum_{i=1}^{n} l(y_i, \hat{y_i}) + \sum_{k=1}^{K} \Omega (f_K)​$

​	式中，加号前为训练损失，加号后为模型的复杂度，即正则项。

**训练过程**

​	在训练模型（寻找最优参数）的过程中，由于每个学习器为一个树模型而不是数值型的向量，因此不能使用SGD（Stochastic Gradient Descent）寻找最优参数。但是可以使用Additive Training（Boosting）来解决，其过程为：

​	分步优化，从常数预测量开始，每次添加一个新的函数，即

​								$\hat{y_i}^{(0)} = 0$

​								$\hat{y_i}^{(1)} = f_1(x_i) = \hat{y_i}^{(0)} + f_1(x_i)$

​								$\hat{y_i}^{(2)} = f_1(x_i) + f_2(x_i) = \hat{y_i}^{(1)} + f_2(x_i)$

​								$...$

​								$\hat{y_i}^{(t)} = \sum_{k=1}^{t} f_k(x_i) = \hat{y_i}^{(t-1)} + f_t(x_i)$

​	其中$\hat{y_i}^{(t)}$为训练第$t$轮时的模型，$\hat{y_i}^{(t-1)}$为第$t-1$轮训练添加的函数，$f_t(x_i)$为第$t$轮添加的函数。

​	添加$f_t$的原则为$f_t$能够最小化目标函数。假设第$t$轮训练时模型的预测为$\hat{y_i}^{(t)} = \hat{y_i}^{(t-1)} + f_t(x_i)$，代入到目标函数为

​								$Obj^{(t)} = \sum_{i=1}^{n} l(y_i, \hat{y_i}^{(t)}) + \sum_{i=1}^{t} \Omega(f_i)$

​									    $= \sum_{i=1}^{n} l(y_i, \hat{y_i}^{(t-1)} + f_i(x_i)) + \Omega (f_t) +constant$

​	对其进行泰勒展开，目标函数变为

​					$Obj^{(t)} \simeq \sum_{i=1}^{n} [l(y_i, \hat{y_i}^{(t-1)}) + g_if_t(x_i) + \frac{1}{2} h_if_t^2(x_i)] + \Omega (f_t) + constant​$

​	其中$g_i = \partial_{\hat{y}^{(t-1)}} l(y_i, \hat{y}^{(t-1)}), h_i = \partial_{\hat{y}^{(t-1)}}^{2} l(y_i, \hat{y}^{(t-1)})$

​	去除所有常数项后，变为

​						  		$\sum_{i=1}^{n} [g_if_t(x_i) + \frac{1}{2} h_if_t^2(x_i)] + \Omega (f_t)$

**完善树模型定义**

​	为了进行正则化，需要对树模型作另一个定义：

​								$f_t(x) = w_{q(x)}, w \in R^T, q : R^d \to \{1,2,\cdots , T\}​$

​	其中$R$为叶节点的权重，$q$为树的结构

​	于是，可以定义模型复杂度为（可以有其他定义形式）：

​									$\Omega (f_t) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^{T} w_j^2​$

​	此处出现的$\gamma$和$\lambda$即为XGBoost模型内的$gamma$和$lambda$参数，增大它们的值可以使模型更加简单，避免过拟合。

​	将其带入先前得到的去除常数项的目标函数，得到

​						$Obj^{(t)} = \sum_{i=1}^{n} [g_if_t(x_i) + \frac{1}{2} h_if_t^2(x_i)] + \gamma T + \frac{1}{2} \lambda \sum_{j=1}^{T} w_j^2$

​							    $= \sum_{j=1}^{T} [(\sum_{i \in I_j} g_i) w_j + \frac{1}{2} (\sum_{i \in I_j} h_i + \lambda) w_j^2] + \gamma T​$

​	令$G_j = \sum{i \in I_j} g_i$ ，$H_j = \sum_{i \in I_j} h_i$，得

​					 			$Obj^{(t)} = \sum_{j=1}^{T} [G_j w_j + \frac{1}{2} (H_j + \lambda) w_j^2] + \gamma T$

​	由于各个叶节点的$w_j$值是相互独立的，于是上式可以看作是一个二次式最佳权重和最佳目标函数为

​											$w_j^* = - \frac{G_j}{H_j + \lambda}$

​									$Obj^* = -\frac{1}{2} \sum_{j=1}^{T} \frac{G_j^2}{H_j + \lambda} + \gamma T$

**树模型的切分**

​	若对树的节点进行切分，则切分后目标函数的改变量为

​								$Gain = \frac{1}{2} [ \frac{G_L^2}{H_L + \lambda} +  \frac{G_R^2}{H_R + \lambda} + \frac{(G_L+G_R)^2}{H_L+H_R+\lambda} ] - \gamma​$

​	其中$ \frac{G_L^2}{H_L + \lambda}$为进行切分后左边子节点的评分，$\frac{G_R^2}{H_R + \lambda}$为进行切分后右边子节点的评分，$\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}$为未进行切分的评分，$\gamma$为添加叶节点后的模型复杂度损失。如果$Gain$为负值，说明进行切分后，模型表现不如以前，则不进行切分；若$Gain$为正值，说明进行切分后，模型表现有所提升。



​	XGBoost和 Gradient Boost 相比 XGBoost 进行了一些优化：

1. 可以进行正则化防止过拟合

2. 考虑了数据为稀疏数据的情况

3. 特征列排序后以块的形式存储在内存中，在迭代中可以重复使用

4. 当数据集比较大时可以考虑怎样充分利用资源，提高算法效率

5. 提供了缺失值的自动处理，允许缺失值的存在

   XGBoost 结合了集成方法的优点以及其自身的特性，快速且有效，在 Kaggle 竞赛中被广泛采用。



### 基准模型
​	LightGBM

​	原本我想使用决策树模型（Decision Tree）作为基准模型训练，但是传统决策树模型不能接受含有缺失值的训练数据进行训练。故使用LightGBM作为基准模型，使用和本次项目使用的XGBoost模型相同的参数，使用同样的训练集进行训练，再和XGBoost模型进行对比。

​	由于LightGBM自身的特性，其训练速度比XGBoost快，但精度不如XGBoost高，因此理论上XGBoost表现是优于LightGBM的。

​	基准值为Kaggle前10%，即Private Leaderboard RMSPE得分为0.11773。




## III. 方法
### 数据预处理

~~~python
# 导入包
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb
% matplotlib inline

# 导入数据集
train = pd.read_csv('train.csv', low_memory=False)
test = pd.read_csv('test.csv', low_memory=False)
store = pd.read_csv('store.csv')
~~~

​	填补缺失值

~~~python
# 补全缺失值
test.fillna(1, inplace=True)
store.fillna(0, inplace=True)
~~~

​	利用Pandas库的merge方法将store.csv中商店的补充信息整合到train.csv和test.csv中

~~~python
# 合并数据集
train = pd.merge(train, store, on='Store')
test = pd.merge(test, store, on='Store')
~~~

​	删除Customer列

~~~python
# 删除Customers列
train.drop(['Customers'], axis=1, inplace=True)
~~~

​	转换Date列为datetime格式

~~~python
# 转换'Date'列为datetime格式
train['Date'] = pd.to_datetime(train['Date'], format='%Y-%m-%d')
test['Date'] = pd.to_datetime(test['Date'], format='%Y-%m-%d')
~~~

​	只保留商店开门且销售额不为0的数据，经处理后Open列无用，将其去除

~~~python
# 去除异常值
train = train.loc[((train['Open'] == 1) & (train['Sales'] > 0))]
train.drop(['Open'], axis=1, inplace=True)
test.drop(['Open'], axis=1, inplace=True)
~~~

**特征工程**

​	由于为时序类预测问题，将train.csv按时间排序，将Date列拆分成Day, Month, Year, WeekOfYear四个列以便模型理解

~~~python
# 对日期进行排序
train = train.sort_values(by='Date')

# 提取日期信息
train['Day'] = train['Date'].dt.day
train['Month'] = train['Date'].dt.month
train['Year'] = train['Date'].dt.year
train['WeekOfYear'] = train['Date'].dt.weekofyear

test['Day'] = test['Date'].dt.day
test['Month'] = test['Date'].dt.month
test['Year'] = test['Date'].dt.year
test['WeekOfYear'] = test['Date'].dt.weekofyear
~~~

​	由于单纯调整参数很难降低RMSPE得分，考虑为特征工程不足的问题，参考Kernel Rossmann Sales Top 1%（https://www.kaggle.com/xwxw2929/rossmann-sales-top1/notebook）的方法进行特征工程。此过程中对CompetitionOpenSinceYear, CompetitionOpenSinceMonth, Promo2SinceYear, Promo2SinceMonth和PromoInterval中的信息加以利用。特别是PromoInterval，其中的月份信息仍为字符串形式，模型无法理解，几乎相当于无用特征，需要通过创造特征加以利用。

~~~python
# 参考Kernel Rossmann Sales Top 1%: https://www.kaggle.com/xwxw2929/rossmann-sales-top1/notebook
def create_features(data):
    data['CompetitionOpen'] = 12 * (data['Year'] - data['CompetitionOpenSinceYear']) + (data['Month'] - data['CompetitionOpenSinceMonth'])
    data['PromoOpen'] = 12 * (data['Year'] - data['Promo2SinceYear']) + (data['WeekOfYear'] - data['Promo2SinceWeek']) / 4.0
    data['CompetitionOpen'] = data['CompetitionOpen'].apply(lambda x: x if x > 0 else 0)        
    data['PromoOpen'] = data['PromoOpen'].apply(lambda x: x if x > 0 else 0)
  
    month2str = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'}
    data['monthStr'] = data['Month'].map(month2str)
    data.loc[data['PromoInterval'] == 0, 'PromoInterval'] = ''
    data['IsPromoMonth'] = 0
    for interval in data['PromoInterval'].unique():
        if interval != '':
            for month in interval.split(','):
                data.loc[(data.monthStr == month) & (data.PromoInterval == interval), 'IsPromoMonth'] = 1
  
    return data

train = create_features(train)
test = create_features(test)
~~~

​	对销售额进行log1p变换，使销售额接近正态分布	

~~~python
# 对销售额进行log1p变换，使销售额接近正态分布
train['Sales'] = train['Sales'].apply(lambda x: np.log1p(x))
~~~

​	对StateHoliday, StoreType,和Assortment列进行独热编码

~~~python
train = pd.get_dummies(data=train,
                       columns=['StateHoliday','StoreType','Assortment'],
                       prefix=['StateHoliday','Store','Assort'])

test = pd.get_dummies(data=test,
                      columns=['StateHoliday','StoreType','Assortment'],
                      prefix=['StateHoliday','Store','Assort'])
~~~



### 执行过程
​	定义评价指标

~~~python
# 定义RMSPE
def rmspe(y, y_hat):
    return np.sqrt(np.mean((y_hat/y-1) ** 2))

def rmspe_xg(y_hat, y):
    y = np.expm1(y.get_label())
    y_hat = np.expm1(y_hat)
    return "rmspe", rmspe(y,y_hat)
~~~

​	取最后6周的数据作为验证集，剩下的数据作为训练集，将train.csv中Sales列单独抽取出来作为需要预测的列，再舍弃无用列

~~~python
# 切割训练集和验证集
# 取最后6周数据作为验证集
val_set = train[train['Date'] >= '2015-06-19']
train_set = train[train['Date'] < '2015-06-19']

y_train = train_set['Sales']
y_val = val_set['Sales']

# 舍弃无用列
X_train = train_set.drop(['Date', 'Sales', 'PromoInterval', 'monthStr'], axis=1)
X_val = val_set.drop(['Date', 'Sales', 'PromoInterval', 'monthStr'], axis=1)
test.drop(['Date', 'PromoInterval', 'monthStr'], axis=1, inplace=True)
~~~

​	将训练集和验证集转化为DMatrix以供XGBoost使用

~~~python
# 转化为DMatrix
dtrain = xgb.DMatrix(X_train, y_train)
dval = xgb.DMatrix(X_val, y_val)

# 设置watchlist
watchlist = [(dtrain, 'train'), (dval, 'eval')]
~~~

​	设定模型参数

~~~python
# 将参数设为随机搜索后的最优值
params = {
        'objective': 'reg:linear',
        'booster': 'gbtree',
        'max_depth': 10,
        'eta': 0.01,
        'subsample': 0.9,
        'colsample_bytree': 0.5,
        'min_child_weight': 1.0,
        'gamma': 2.0,
        'lambda': 1.0,
        'tree_method': 'gpu_hist',
        'silent': 1
        }
~~~

​	设置迭代次数并训练模型

~~~python
# 设置迭代次数
num_boost_round = 10000

# 训练XGBoost模型
xgb_model = xgb.train(params, dtrain, num_boost_round, evals=watchlist, \
                  early_stopping_rounds=100, feval=rmspe_xg, verbose_eval=True)
~~~



### 完善
​	此项目使用了随机搜索（Randomized Search）进行参数调整，但由于进行测试的参数是在参数空间内随机选取的，因此无法准确重现，而且选取的数量有限，不保证能选取到参数空间内真正最优的参数。

​	进行随机搜索

~~~python
# 创建RandomizedSearchCV的scorer
scorer = make_scorer(rmspe)

# 随机搜索参数设置
param = {
        'max_depth': [10],
        'subsample': [0.9],
        'eta': [0.01],
        'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [0.5, 1.0, 2.0, 3.0, 4.0, 5.0],
        'gamma': [1.0, 2.0, 3.0, 4.0, 5.0],
        'lambda': [1.0, 2.0, 3.0, 4.0, 5.0]
        }

# 创建XGBoost回归模型
model = xgb.XGBRegressor(silent=1, objective='reg:linear', booster='gbtree', \
                         tree_method='gpu_hist', num_round=200, seed=0)

# 第一次随机搜索
randomized_search = RandomizedSearchCV(model, param, n_iter=30, scoring=scorer, \
                                       random_state=0)
~~~

​	经过第一次随机搜索后，得到的最佳参数为：

```python
{'colsample_bytree': 0.5,
 'eta': 0.01,
 'gamma': 2.0,
 'lambda': 1.0,
 'max_depth': 10,
 'min_child_weight': 1.0,
 'subsample': 0.9}
```




## IV. 结果
### 模型的评价与验证

​	利用XGBoost自带方法plot_importance绘制特征重要性图

![图5: 特征重要性图](/Users/xuzisheng/Desktop/School_stuff/Udacity/机器学习工程师/forecast_rossmann_sales/second_submission/feature_importance.png)

​										图5：特征重要性图

​	参考的Kernel Rossmann Sales Top 1%（https://www.kaggle.com/xwxw2929/rossmann-sales-top1/notebook）中最后有个权重调整的环节，其过程为根据验证集选取一个最佳权重，使验证集经过调整权重后RMSPE得分能进一步降低，将此权重乘以测试集的预测数据，得分也能进一步降低。

~~~python
# 权重调整，参考Kernel Rossmann Sales Top 1%: https://www.kaggle.com/xwxw2929/rossmann-sales-top1/notebook
def weight_correction():
    weights = [(0.990+(i/1000)) for i in range(20)]
    errors = []
    for w in weights:
        error = rmspe(np.expm1(y_val), np.expm1(y_pred*w))
        errors.append(error)
    # plotting
    plt.plot(weights, errors)
    # print minimum error
    best_weight = weights[errors.index(min(errors))]
    print('Best weight is {}.'.format(best_weight))
~~~

~~~
weight_correction()
~~~

 	作图，得到最佳权重为0.996

![图6: 权重调整](/Users/xuzisheng/Desktop/School_stuff/Udacity/机器学习工程师/forecast_rossmann_sales/second_submission/weight_correction.png)

​										图6：权重调整

~~~python
# 将ID列单独取出并删除测试集里的ID
test_id = test['Id']
test.drop(['Id'], axis=1, inplace=True)

# 添加两列新列，防止mismatch的发生导致无法预测
test.insert(19, 'StateHoliday_b', 0)
test.insert(20, 'StateHoliday_c', 0)

# 对测试集进行预测
test_pred = xgb_model.predict(xgb.DMatrix(test))

# 将ID列放回提交的文件
submit_file = pd.concat([test_id, pd.Series(np.expm1(test_pred*0.996))], axis=1)
# 重命名列
submit_file.rename(columns={0:'Sales'}, inplace = True)

# 保存为csv文件提交至Kaggle
submit_file.to_csv('second_submission_10000.csv', index=False)
~~~

​	将最终预测结果上传到Kaggle进行打分，得到的结果与基准模型进行比较，得

![图7: 模型与基准模型得分比较](/Users/xuzisheng/Desktop/School_stuff/Udacity/机器学习工程师/forecast_rossmann_sales/second_submission/Kaggle得分.png)

​								图7：模型与基准模型得分比较

​	最终XGBoost模型的private得分为0.11681，public得分为0.11400；作为基准的LightGBM模型的private得分为0.13438，public得分为0.13058。

​	XGBoost模型超过了基准值0.11773，而LightGBM模型由于精度不及XGBoost，未超过基准值。

## V. 项目结论

### 结果可视化
![](/Users/xuzisheng/Desktop/School_stuff/Udacity/机器学习工程师/forecast_rossmann_sales/second_submission/store_520.png)

![](/Users/xuzisheng/Desktop/School_stuff/Udacity/机器学习工程师/forecast_rossmann_sales/second_submission/store_954.png)

![](/Users/xuzisheng/Desktop/School_stuff/Udacity/机器学习工程师/forecast_rossmann_sales/second_submission/store_1039.png)

​								图8：随机抽取商店的预测值与真实值对比

​	520号商店相比来说拟合得最好，934号商店拟合得较好，1039号商店拟合得最差，与真实值相差较大。



### 需要作出的改进

​	特征工程：Rossmann比赛第一名Gert的特征工程做得十分详尽，此项目中特征工程仅仅是对日期列进行拆分，以及进行独热编码的简单处理，相信通过细致处理特征工程，模型的表现能进一步得到提升。

​	参数优化：由于此项目使用随机搜索进行参数优化，有可能真正的最优参数不会被搜索到，提升抽取参数组合的数量能解决这一问题，但耗时会明显增长。此项目恰恰是因为想要快速搜寻最优参数而使用随机搜索而不是网格搜索。

​	模型：此项目使用仅仅XGBoost一个模型，如果能像集成方法一样，使用多个模型融合，模型的表现可能会得以提升。

----------
**References**

[^1]: 信息来自Kaggle竞赛页面https://www.kaggle.com/c/rossmann-store-sales
[^2]: https://www.kaggle.com/c/rossmann-store-sales/data
[^3]: https://www.kaggle.com/c/rossmann-store-sales#evaluation

Pandas官方文档：http://pandas.pydata.org/pandas-docs/version/0.24.0rc1/

XGBoost官方文档：https://xgboost.readthedocs.io/en/latest/index.html

Seaborn官方文档：http://seaborn.pydata.org/index.html

竞赛kernel-Rossmann Sales Top 1%：https://www.kaggle.com/xwxw2929/rossmann-sales-top1

陈天奇XGBoost介绍slides：https://homes.cs.washington.edu/~tqchen/pdf/BoostedTree.pdf
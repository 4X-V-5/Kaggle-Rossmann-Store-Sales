# Machine Learning Engineer Nanodegree
## Capstone Proposal
许梓晟  
January 22nd, 2019

## Proposal
### Domain Background
​	   机器学习是计算机系统用来逐步优化其在特定任务上表现的算法和统计模型的科学研究。[^1] 分为监督学习、无监督学习、半监督学习和强化学习几个类别，并衍生出基于多层神经网络的深度学习这一学科。相对于理论还不完善的深度学习，机器学习经过数十年发展，理论已经趋于完善，并逐步应用于业界。具体例子有Google利用机器学习对其数据中心的资源进行合理分配以降低能耗、利用机器学习进行天气预测、利用机器学习进行商业预测等等。此项目的目的为利用机器学习模型预测Rossmann商店的销售额。商店销售额受许多因素影响，单靠人工预测销售额耗时长的，并且由于人的大脑的特性，预测销售额并不是一个直观的过程，可能难以总结出规律。机器学习模型的引入使预测的整个过程更加直观、快捷，并且机器学习模型可能能够学习到人难以通过直觉观察到的潜在规律。

### Problem Statement
​	此项目属于机器学习中有监督学习中的回归问题，也属于时序类预测问题。主要的问题有如何利用store.csv文件，数据清理及一些异常值处理，特征工程以及寻找模型的最优参数。应用机器学习模型的主要困难在于数据的处理、如何选择合适模型以及模型的调整问题。通常我们的原始数据集是不可以直接用来训练模型的，因为其常常含有缺失值、异常值或者值的类型不对等等问题。再者，模型的选择问题——受目前的研究进展的影响，我们的模型并不是一个通用的模型，不同的模型所解决的问题是不一样的，基本类型有回归、分类和标注模型，而解决同一种问题的模型由于它们自身特性对数据集又有不同的表现。最后我们需要根据所选择的模型的原理，针对我们的数据集进行优化，最优化模型在我们所需要解决的问题的表现。

### Datasets and Inputs
此项目所对应的数据集可以在 Kaggle Rossmann Sales 竞赛页面下载[^2] ，Kaggle提供的数据文件有3个，分别为：

> train.csv - 包括销售额的历史数据
>
> test.csv - 不包括销售额的历史数据
>
> sample_submission.csv - 包含正确提交格式的提交文件样本
>
> store.csv - 商店的补充信息

**数据集包含的特征**

大部分特征都可以顾名思义，以下是需要加以说明的特征的简介

> Id - 在测试集中表示 (商店, 日期) 的编号
>
> Store - 每一个商店的唯一编号
>
> Sales - 给定日期的销售额 (需要预测的项目)
>
> Customers - 给定日期的客户数
>
> Open - 表示商店是否营业：0 = 休息，1 = 营业
>
> StateHoliday - 表示州假日，除了少数特例外，正常情况下所有商店在州假日休息。所有学校在公众假日和周末休息。a = 公众假日，b = 复活节假日，c = 圣诞节，0 = 无
>
> SchoolHoliday - 表示 (商店, 日期) 是否受到公立学校休息的影响
>
> StoreType - 区分4种不同商店：a, b, c, d
>
> Assortment - 描述分类级别：a = 基本，b = 额外，c = 扩展
>
> CompetitonDistance - 距离最近竞争对手商店的距离，以米为单位
>
> CompetitionOpenSince[Month/Year] - 提供竞争对手商店开始营业的大致年份和月份
>
> Promo - 表示那一天商店是否在促销
>
> Promo2 - Promo2是一些商店的连续促销：0 = 商店不参加，1 = 商店参加
>
> Promo2Since[Year/Week] - 描述商店参加Promo2的年份和周数
>
> PromoInterval - 描述Promo2启动的连续间隔，代表商店开始新促销的月份。 例如。 “2月，5月，8月，11月”意味着每一轮促销在该商店的任何一年的2月，5月，8月，11月开始

​	利用pandas的merge方法对数据集进行合并。

~~~python
train['Date'] = pd.to_datetime(train['Date'], format='%Y-%m-%d')
test['Date'] = pd.to_datetime(test['Date'], format='%Y-%m-%d')
~~~

​	将store.csv的商店补充信息添加到train.csv和test.csv中后，查看数据。

~~~python
train.info()
~~~



![image-20190123095930955](/Users/xuzisheng/Library/Application Support/typora-user-images/image-20190123095930955.png)

~~~python
test.info()
~~~



​	train.csv共有1017209条数据，可以看到其中CompetitionDistance, CompetitionSinceMonth, CompetitionSinceYear, Promo2SinceWeek, Promo2SinceYear, PromoInterval有缺失值，并且Date需要转换为datetime格式。

​	![image-20190123100301173](/Users/xuzisheng/Library/Application Support/typora-user-images/image-20190123100301173.png)

​	test.csv中共有41088条数据，其中Open, CompetitionDistance, CompetitionSinceMonth, CompetitionSinceYear, Promo2SinceWeek, Promo2SinceYear, PromoInterval有缺失值，Date列也需要转换为datetime格式。

​	由于test.csv是不包含Customer列的，因此train.csv中的Customer列对预测销售额不能发挥作用，予以删除。

~~~python
train.drop(['Customers'], axis=1, inplace=True)
~~~

​	train.csv中最早日期为2013-01-01，最晚日期为2015-07-31；test.csv中最早日期为2015-08-01，最晚日期为2015-09-17。

​	由于包含销售额为0的数据时，RMSPE的分数会出现inf时情况，因此只保留商店开门且销售额大于0的数据。经过处理后，Open列的数值都是1，没有意义，予以删除。

~~~python
# 去除异常值
train = train.loc[((train['Open'] == 1) & (train['Sales'] > 0))]
train.drop(['Open'], axis=1, inplace=True)
test.drop(['Open'], axis=1, inplace=True)
~~~

### 可视化

​	直观上来说，可能影响销售额的因素有:

1. DayOfWeek: 如果商店不进行休息，那么周末的销售额可能会比工作日的销售额多，人们有更多的时间去商店

2. Date: 特定的时间可能受节日促销等的影响，销售额提升

3. Promo: 进行促销的日子应当能推动顾客的购买商品，从而提高销售额

4. CompetitonDistance: 附近有竞争对手，可能销售额会有所影响

   首先选取一个数据较多的商店——1023号商店进行分析，以Sales列对DayOfWeek列作图，得到下图

![图1:1023号商店一周七天中销售额的分布](/Users/xuzisheng/Desktop/School_stuff/Udacity/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%B7%A5%E7%A8%8B%E5%B8%88/forecast_rossmann_sales/second_submission/DayOfWeek.png)

​	*可以看到，除了星期日商家休息无销售额外，星期一销售额分布比较分散，其中位数比其余日子大，为7500左右，而其余日子对销售额没有太大影响，中位数都落在6000左右。*

​	以Sales列对Date列作图，得到下图

![图2:1023号商店所有日期中销售额的分布](/Users/xuzisheng/Desktop/School_stuff/Udacity/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%B7%A5%E7%A8%8B%E5%B8%88/forecast_rossmann_sales/second_submission/Dates.png)

​	*可以看到，有趣的是，2014年和2015年的1月份左右有几天销售额是明显比其他日子高的，可能是由于新年假期的缘故。*

​	商店是否进行促销也可能对商店营业额造成影响，以Sales列对Promo列作图，得到下图

![图3:商店是否促销和销售额的关系](/Users/xuzisheng/Desktop/School_stuff/Udacity/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%B7%A5%E7%A8%8B%E5%B8%88/forecast_rossmann_sales/second_submission/Promo.png)

​	*可以看到，有促销的商店销售额的中位数比没有促销的商店高一点。由于已经剔除了商店不营业，销售额为0的情况，中位数不会受极小值影响。*

​	选取a类商店进行分析，对CompetitionDistance列进行groupby分组并取均值，以Sales列对处理后数据作图，得到下图

![图4：a商店竞争对手距离和销售额的关系](/Users/xuzisheng/Desktop/School_stuff/Udacity/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%B7%A5%E7%A8%8B%E5%B8%88/forecast_rossmann_sales/second_submission/CompetitionDistance.png)

​	*可以看到，似乎竞争对手距离对销售额的影响并不明显*



### Solution Statement

​	由于使用传统的决策树模型进行预测面临着准确率不高的问题，需要使用一种前沿的模型提高准确率。我计划采用在 Leaderboard 上被广泛使用的 XGBoost 算法对销售额进行预测。XGBoost (eXtreme Gradient Boosting) 是一种集成学习方法，是 Gradient Boost 算法的高效实现。和 Gradient Boost 相比 XGBoost 进行了一些优化：

1. 可以进行正则化防止过拟合

2. 考虑了数据为稀疏数据的情况

3. 特征列排序后以块的形式存储在内存中，在迭代中可以重复使用

4. 当数据集比较大时可以考虑怎样充分利用资源，提高算法效率

5. 提供了缺失值的自动处理，允许缺失值的存在

   XGBoost 结合了集成方法的优点以及其自身的特性，快速且有效，在 Kaggle 竞赛中被广泛采用

### Benchmark Model
​	LightGBM

​	原本我想使用决策树模型（Decision Tree）作为基准模型训练，但是传统决策树模型不能接受含有缺失值的训练数据进行训练。故使用LightGBM作为基准模型，使用和本次项目使用的XGBoost模型相同的参数，使用同样的训练集进行训练，再和XGBoost模型进行对比。

​	由于LightGBM自身的特性，其训练速度比XGBoost快，但精度不如XGBoost高，因此理论上XGBoost表现是优于LightGBM的。

​	基准值为Kaggle前10%，即Private Leaderboard RMSPE得分为0.11773。

### Evaluation Metrics
​	对模型评估的标准由 Kaggle Rossmann Store Sales 竞赛给出的均方根百分比误差 (Root Mean Suqare Percentage Error, RMSPE) 衡量[^3]，公式为

​								RMSPE = $\sqrt{\frac{1}{n} \sum_{i=1}^{n} (\frac{y_i - \hat{y_i}}{y_i})^2}$ 

​	其中 $y_i$ 表示一间商店一天的实际销售额，$\hat{y_i}$ 表示一间商店一天的预测销售额。

​	从公式可以看出，预测值与真实值越接近，RMSPE的值越低，因此，此项目的目标也即为降低RMSPE得分。

### Project Design
1. 数据合并

   将 store.csv 中的信息合并到 train.csv 中

2. 数据可视化

   进行一些数据可视化，观察是否存在异常数据

3. 对异常数据进行处理

   此处主要是将转换数据列为正确的格式；前面提到 XGBoost 可以自动处理缺失值，因此打算将缺失值交由 XGBoost 处理

4. 特征工程

   上面的 Datasets and inputs 提到 StateHoliday, StoreType 等一些数据列的数据是 string 格式 'a', 'b', 'c' 等，可以进行独热编码处理，并且将Date列内的日期信息提取出来

5. 划分训练集和验证集

   由于最终模型打分提交到 Kaggle 进行打分，我打算将train.csv分为训练集和验证集。由于这是一个时序类预测问题，打算将train.csv按Date列排序，再取最后六周的数据作为验证集，其余的作为训练集使用。

6. 创建一个 XGBoost 模型，按照 Kaggle 提供的RMSPE 定义 Loss Function，作为模型的 Metric

7. 训练模型，进行调参

8. 提交到 Kaggle 打分



[^1]: 摘自维基百科“机器学习”词条：https://en.wikipedia.org/wiki/Machine_learning
[^2]: https://www.kaggle.com/c/rossmann-store-sales/data
[^3]: https://www.kaggle.com/c/rossmann-store-sales#evaluation
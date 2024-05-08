
import joblib
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
# from sklearn.datasets import load_boston
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
 
 
 
 # 加载数据
data = load_breast_cancer()
X = data.data
y = data.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
 
 
print(X)
 
 
 
 
 
 
 
# #  #加载数据
# # boston = load_boston()

# boston = pd.read_csv('boston_housing_prices.csv')
# data = boston.data
# target = boston.target
# X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)
 
# # 创建成lgb特征的数据集格式
# lgb_train = lgb.Dataset(X_train, y_train)
# lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
 
# # 将参数写成字典下形式
# params = {
#     'task': 'train',
#     'boosting_type': 'gbdt',  # 设置提升类型
#     'objective': 'regression',  # 目标函数
#     'metric': {'l2', 'auc'},  # 评估函数
#     'num_leaves': 31,  # 叶子节点数
#     'learning_rate': 0.05,  # 学习速率
#     'feature_fraction': 0.9,  # 建树的特征选择比例
#     'bagging_fraction': 0.8,  # 建树的样本采样比例
#     'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
#     'verbose': 1  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
# }
 
# # 训练 cv and train
# gbm = lgb.train(params, lgb_train, num_boost_round=20, valid_sets=lgb_eval, early_stopping_rounds=5)
 
 
# # 保存模型到文件
# #gbm.save_model('model.txt')
# joblib.dump(lgb, './model/lgb.pkl')
 
# # 预测数据集
# y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
 
# # 评估模型
# print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)

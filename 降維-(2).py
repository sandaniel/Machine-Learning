from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

import numpy as np
from sklearn import manifold
import pandas as pd
import matplotlib.pyplot as plt

# 讀入資料, pandas.DataFrame
df=pd.read_csv('iris.csv', 
	sep=',', 
	names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'speices'])

df_feature=df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
df_label=df[['speices']]


# 建立一個分類模型
clf=LogisticRegression()
clf.fit(df_feature, df_label)

# 選出n_features_to_select個特徵作為主要分類依據
selector=RFE(clf, n_features_to_select=2)
selector=selector.fit(df_feature, df_label)

print(selector.support_)
print(selector.ranking_)

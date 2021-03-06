import numpy as np
from sklearn import manifold
import pandas as pd
import matplotlib.pyplot as plt


#----------------------------------------------------
# 讀入資料, pandas.DataFrame
df=pd.read_csv('iris.csv', 
	sep=',', 
	names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'speices'])

df_feature=df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
df_label=df[['speices']]


#----------------------------------------------------
mds=manifold.MDS(n_components=2)
Xtrans=mds.fit_transform(df_feature, df_label)


#----------------------------------------------------
# 設定字型及大小
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['font.size'] = 14

# 設定圖標題
plt.title('IRIS dataset')

# 設定x軸及y軸標題
plt.xlabel('X1')
plt.ylabel('X2')

# 資料表內的grid
plt.grid(True)

# 設定x軸及y軸的尺規範圍
plt.axis([-5, 5, -5, 5])

# 繪製資料
plt.plot(Xtrans[:50, 0], Xtrans[:50, 1], 'ro')
plt.plot(Xtrans[50:100, 0], Xtrans[50:100, 1], 'bd')
plt.plot(Xtrans[100:, 0], Xtrans[100:, 1], 'g^')

# 設定資料說明
plt.legend(['setosa', 'versicolor', 'virginica'], numpoints=1, loc='upper left')

# 顯示圖表
plt.show()

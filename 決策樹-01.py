#---------------------------------------------
# 下載安裝Graphviz, http://www.graphviz.org
# 增加path, 指向安裝路徑下的\bin
#---------------------------------------------
import numpy as np
from sklearn import manifold
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import decomposition

from sklearn import tree
import pydotplus 


#----------------------------------------------------
# 讀入資料, pandas.DataFrame
df=pd.read_csv('iris.csv', 
	sep=',', 
	names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'speices'])

df_feature=df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
df_label=df[['speices']]


#----------------------------------------------------
clf = tree.DecisionTreeClassifier()
clf = clf.fit(df_feature, df_label)

print(df_feature[0:2])
print(clf.predict(df_feature[0:2]))


#----------------------------------------------------
dot_data = tree.export_graphviz(clf, out_file=None) 
graph = pydotplus.graph_from_dot_data(dot_data) 
graph.write_pdf('iris.pdf') 

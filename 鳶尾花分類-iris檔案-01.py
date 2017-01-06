import numpy as np
import matplotlib.pyplot as plt

# 讀入資料
#依序為花萼長度, 花萼長度, 花瓣寬度, 花瓣長度, 品種
#data=np.genfromtxt('iris.csv',  dtype=None, delimiter=',')

# 讀入資料
data=np.genfromtxt('iris.csv',  dtype=[('sepal_length', 'f'), ('sepal_width', 'f'), ('petal_length', 'f'), ('petal_width', 'f'), ('speices', 'U25')], delimiter=',', usecols=(0, 1, 2, 3, 4), unpack=True)

# 整理資料
petal_length=data['petal_length']
speices=data['speices']
not_setosa_petal_length=petal_length[speices!='setosa']


# 印出資料
print(len(not_setosa_petal_length))
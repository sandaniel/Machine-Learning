import numpy as np
import matplotlib.pyplot as plt

# 讀入資料
#依序為花萼長度, 花萼長度, 花瓣寬度, 花瓣長度, 品種
#data=np.genfromtxt('iris.csv',  dtype=None, delimiter=',')

# 讀入資料
data=np.genfromtxt('iris.csv',  dtype=[('sepal_length', 'f'), ('sepal_width', 'f'), ('petal_length', 'f'), ('petal_width', 'f'), ('speices', 'U25')], delimiter=',', usecols=(0, 1, 2, 3, 4), unpack=True)

# 整理資料
petal_length=data['petal_length']
petal_width=data['petal_width']
speices=data['speices']

not_setosa_petal_length=petal_length[speices!='setosa']
not_setosa_petal_width=petal_width[speices!='setosa']
not_setosa_speices=speices[speices!='setosa']

t=0;
success_rate=-100

for one in not_setosa_petal_width:
    a1 = not_setosa_speices[not_setosa_petal_width >= one]
    a2 = not_setosa_speices[not_setosa_petal_width < one]

    acc1=len(a1[a1 == 'virginica'])
    acc2=len(a2[a2=='versicolor'])
    acc=acc1+acc2

    if (acc/len(not_setosa_speices)) > success_rate:
        success_rate=acc/len(not_setosa_speices)
        t=one


# 印出資料
print('正確率:', success_rate)
print('閥值:', t)
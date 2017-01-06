import numpy as np
import matplotlib.pyplot as plt

# 讀入資料
#依序為花萼長度(sepal_length), 花萼寬度(sepal_width), 花瓣長度(petal_length), 花瓣寬度(petal_width), 品種(species)
data=np.genfromtxt('iris.csv',  dtype=None, delimiter=',')

# 整理資料


# 待繪製的資料
sepal_length_setosa=[]
sepal_width_setosa=[]
petal_length_setosa=[]
petal_width_setosa=[]

sepal_length_versicolor=[]
sepal_width_versicolor=[]
petal_length_versicolor=[]
petal_width_versicolor=[]

sepal_length_virginica=[]
sepal_width_virginica=[]
petal_length_virginica=[]
petal_width_virginica=[]

for (sepal_length, sepal_width, petal_length, petal_width, species) in data:
    s=species.decode('UTF-8')

    if s=='setosa':
        sepal_length_setosa.append(sepal_length)
        sepal_width_setosa.append(sepal_width)
        petal_length_setosa.append(petal_length)
        petal_width_setosa.append(petal_width)

    if s=='versicolor':
        sepal_length_versicolor.append(sepal_length)
        sepal_width_versicolor.append(sepal_width)
        petal_length_versicolor.append(petal_length)
        petal_width_versicolor.append(petal_width)

    if s=='virginica':
        sepal_length_virginica.append(sepal_length)
        sepal_width_virginica.append(sepal_width)
        petal_length_virginica.append(petal_length)
        petal_width_virginica.append(petal_width)

# 設定字型及大小
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['font.size'] = 14

# 設定圖標題
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(6, 6))

#(1)
axes[0, 0].set_xlabel('花萼寬')
axes[0, 0].set_ylabel('花萼高')
axes[0, 0].get_xaxis().set_ticks([])
axes[0, 0].get_yaxis().set_ticks([])
setosa=axes[0, 0].scatter(sepal_length_setosa, sepal_width_setosa, color='r')
versicolor=axes[0, 0].scatter(sepal_length_versicolor, sepal_width_versicolor, color='b')
virginica=axes[0, 0].scatter(sepal_length_virginica, sepal_width_virginica, color='g')
axes[0, 0].grid()

#(2)
axes[0, 1].set_xlabel('花瓣寬')
axes[0, 1].set_ylabel('花瓣高')
axes[0, 1].get_xaxis().set_ticks([])
axes[0, 1].get_yaxis().set_ticks([])
setosa=axes[0, 1].scatter(petal_length_setosa, petal_width_setosa, color='r')
versicolor=axes[0, 1].scatter(petal_length_versicolor, petal_width_versicolor, color='b')
virginica=axes[0, 1].scatter(petal_length_virginica, petal_width_virginica, color='g')
axes[0, 1].grid()

#(3)
axes[0, 2].set_xlabel('花萼寬')
axes[0, 2].set_ylabel('花瓣高')
axes[0, 2].get_xaxis().set_ticks([])
axes[0, 2].get_yaxis().set_ticks([])
setosa=axes[0, 2].scatter(sepal_length_setosa, petal_width_setosa, color='r')
versicolor=axes[0, 2].scatter(sepal_length_versicolor, petal_width_versicolor, color='b')
virginica=axes[0, 2].scatter(sepal_length_virginica, petal_width_virginica, color='g')
axes[0, 2].grid()

#(4)
axes[1, 0].set_xlabel('花瓣寬')
axes[1, 0].set_ylabel('花萼高')
axes[1, 0].get_xaxis().set_ticks([])
axes[1, 0].get_yaxis().set_ticks([])
setosa=axes[1, 0].scatter(petal_width_setosa, sepal_length_setosa, color='r')
versicolor=axes[1, 0].scatter(petal_width_versicolor, sepal_length_versicolor, color='b')
virginica=axes[1, 0].scatter(petal_width_virginica, sepal_length_virginica, color='g')
axes[1, 0].grid()

#(5)
axes[1, 1].set_xlabel('花瓣高')
axes[1, 1].set_ylabel('花萼寬')
axes[1, 1].get_xaxis().set_ticks([])
axes[1, 1].get_yaxis().set_ticks([])
setosa=axes[1, 1].scatter(petal_length_setosa, sepal_width_setosa, color='r')
versicolor=axes[1, 1].scatter(petal_length_versicolor, sepal_width_versicolor, color='b')
virginica=axes[1, 1].scatter(petal_length_virginica, sepal_width_virginica, color='g')
axes[1, 1].grid()

#(6)
axes[1, 2].set_xlabel('花萼高')
axes[1, 2].set_ylabel('花瓣寬')
axes[1, 2].get_xaxis().set_ticks([])
axes[1, 2].get_yaxis().set_ticks([])
setosa=axes[1, 2].scatter(sepal_width_setosa, petal_length_setosa, color='r')
versicolor=axes[1, 2].scatter(sepal_width_versicolor, petal_length_versicolor, color='b')
virginica=axes[1, 2].scatter(sepal_width_virginica, petal_length_virginica, color='g')
axes[1, 2].grid()


# 顯示圖表
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# 讀入資料
#依序為花萼長度, 花萼長度, 花瓣寬度, 花瓣長度, 品種
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
fig, ax = plt.subplots()
ax.set_title('鳶尾花特徵分佈圖')

# 設定x軸及y軸標題
plt.xlabel('花萼寬度')
plt.ylabel('花萼高度')


# 將測試資料畫在圖上
setosa=plt.scatter(sepal_length_setosa, sepal_width_setosa, color='r')
versicolor=plt.scatter(sepal_length_versicolor, sepal_width_versicolor, color='b')
virginica=plt.scatter(sepal_length_virginica, sepal_width_virginica, color='g')

# 產生左上角的標籤說明
plt.legend((setosa, versicolor, virginica),
           ('setosa', 'versicolor', 'virginica'),
           scatterpoints=1,
           loc='upper left',
           fontsize=14)

# 畫出格線
plt.grid()

# 顯示圖表
plt.show()
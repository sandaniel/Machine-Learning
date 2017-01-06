import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# 訓練資料
# 依序為花萼長度(sepal_length), 花萼寬度(sepal_width), 花瓣長度(petal_length), 花瓣寬度(petal_width), 品種(species)
data=np.genfromtxt('iris.csv',  dtype=None, delimiter=',')

X=[]
Y=[]

# 訓練資料範圍, 為了繪圖用
a_min=999
a_max=-999
b_min=999
b_max=-999

for (sepal_length, sepal_width, petal_length, petal_width, species) in data:
    s=species.decode('UTF-8')

    #a=sepal_length
    #b=sepal_width
    a=petal_length
    b=petal_width

    if a>a_max: a_max=a
    if a<a_min: a_min=a
    if b>b_max: b_max=b
    if b<b_min: b_min=b
	
    X.append([a, b])		
	
	# 品種共有: setosa, versicolor, virginica
    if s=='versicolor':
        Y.append(0)
    else:
        Y.append(1)
				
X=np.array(X)
Y=np.array(Y)

		

# 建立分類模型
clf = svm.SVC(kernel='rbf', gamma=1, C=1)
clf.fit(X, Y)


# 設定圖形尺寸 
plt.figure(figsize=(12, 9))
#plt.clf()


# 畫出 support vectors
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=30, facecolors='y', zorder=10)
plt.axis('tight')


# 由訓練資料的標籤畫出不同顏色區域
x_min = a_min - 1.5
x_max = a_max + 1.5
y_min = b_min - 1.5
y_max = b_max + 1.5

XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

Z = Z.reshape(XX.shape)
plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'], levels=[-.5, 0, 0.5])

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)


# 畫出 training vectors
label_0 = np.array(Y) == 0
X_0 = X[label_0]

label_1 = np.array(Y) == 1
X_1 = X[label_1]

plt.scatter(X_0[:,0], X_0[:,1], c='b', s=120)
plt.scatter(X_1[:,0], X_1[:,1], c='g', s=120)


# 繪圖
plt.show()
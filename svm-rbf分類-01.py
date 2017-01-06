# Code source: Gaël Varoquaux
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# 訓練資料
X = np.c_[(-0.5, -.7),
          (-1.5, -1),
          (-1.4, -.9),
          (-1.3, -1.2),
          (-1.1, -.2),
          (-1.2, -.4),
          (-.5, 1.2),
          (-1.5, 2.1),
          (1, 1),
          # --
          (1.3, .8),
          (1.2, .5),
          (.2, -2),
          (.5, -2.4),
          (.2, -2.3),
          (0, -2.7),
          (1.3, 2.1)].T
		  
# 訓練資料的標籤
Y = [0] * 8 + [1] * 8


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
x_min = -3
x_max = 3
y_min = -3
y_max = 3

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

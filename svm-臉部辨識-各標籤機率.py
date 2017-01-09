
from __future__ import print_function

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn import svm
import numpy as np
import random


#------------------------------------------------
#   |
#   |--- face_recognition.py
#   |
#   |--- <faceData>
#            |
#            |-- <某人>
#            |      |
#            |   (此人照片1, 2, ...) 250px * 250px
#            |
#            |-- <某人>
#            |      |
#            |   (此人照片1, 2, ...)
#------------------------------------------------

# 載入照片檔, 但只有至少有100張照片的個人才載入, 臉部裁切後大小調整為0.4倍)
lfw_people = fetch_lfw_people(data_home='faceData/', min_faces_per_person=100, resize=0.4)


# 載入的(訓練+測試)照片數目, 高度及寬度
n_samples, h, w = lfw_people.images.shape


# 載入的(訓練+測試)照片的特徵值向量, 每張照片約有2000個特徵值
X = lfw_people.data
n_features = X.shape[1]


# 載入的(訓練+測試)照片的標籤(資料夾名稱), 有2000個特徵值, 約有5個不同標籤)
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]


# 將載入的(訓練+測試)照片, 分成75%作為訓練, 25%作為測試
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


#====================================================================================================
# 計算主成分分析(PCA, Principal Component Analysis) - 載入的(訓練+測試)照片以隱藏標籤的作法, 進行
#                                                     資臉部特徵(eigenfaces)萃取.
#
# 任意一張人臉圖像都可以被認為是將個標準臉的組合。
# 例如，一張人臉圖像可能是一張平均臉 + 特徵臉1的10% + 特徵臉2的55% - 特徵臉3的3%
# 值得注意的是，不需要太多的特徵臉來獲得大多數臉的近似組合.
#====================================================================================================
n_components = 150  #PCA保留的特徵數

print('所有特徵數=', X_train.shape[0])
print('PCA選用', n_components , '個特徵')

# n_components, PCA保留的特徵數
# svd_solver, randomized=使用Halko et al, 2011,的方法在所有特徵中選擇保留之特徵
# whiten, 白化, 每張圖有相同的方差
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X_train)


# 特徵臉重建
eigenfaces = pca.components_.reshape((n_components, h, w))


# 將訓練資料轉出為 PCA 資料
X_train_pca = pca.transform(X_train)

# 將測試資料轉出為 PCA 資料
X_test_pca = pca.transform(X_test)



# 設定 grid search 的參數範圍
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
			  
# 進行 grid search			  
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced', probability=True), param_grid)


#==============================
# 由最佳C及gamma參數訓練模型
#==============================
clf = clf.fit(X_train_pca, y_train)


#==============================
# 進行測試
#==============================
y_pred = clf.predict(X_test_pca)

# precision: 準確率
# recall rate: 召回率 = 正確被檢測出的數量 / 所有應該被檢測出的數量
# f1-score: (準確率 + 召回率) / 2, 有時兩者有不同權重
# support: 各類別正確判斷的個數
print('測試結果:')
print(classification_report(y_test, y_pred, target_names=target_names))


# confusion_matrix: 應該被分類到i, 最後被判斷為分類j, 形成的(i, j)值矩陣
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))


# 印出測試圖片屬於各個標籤的可能性
for i in range(len(X_test_pca)):
    print('第', i+1, '張圖:')
    predict=clf.predict_proba(X_test_pca[i].reshape(1, -1))
    for j in range(len(predict[0])):
        print(target_names[j], ':', predict[0, j])

    print('-'*40)
	
print('='*60)


#============================================================
# 繪圖
#============================================================
# 設定字型及大小
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['font.size'] = 14

# 選12張圖
k = random.randint(0, 100) #由測試圖片數量調整
print('亂數:', k)

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):   
    print('圖片共', len(images), '張') 
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)		
        plt.imshow(images[i+k].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i+k], size=12)
        plt.xticks(())
        plt.yticks(())

def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return '判斷:%s\n正確:%s' % (pred_name, true_name)


# 畫出測試圖片及測試結果	
prediction_titles = [title(y_pred, y_test, target_names, i)
                     for i in range(y_pred.shape[0])]

plot_gallery(X_test, prediction_titles, h, w)


# 畫出特徵圖片
eigenface_titles = ["特徵臉, 編號:%d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)


# 顯示圖片
plt.show()

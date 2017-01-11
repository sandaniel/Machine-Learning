#===========================================================
# 先載入 mnist提供的手寫字圖形
# (4個壓縮檔, 解壓縮成4個檔案)
# 存在python程式所在目錄中的 <handwritten>子目錄中
#===========================================================
import os, struct
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros
import numpy as np

from numpy import *
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


def load_mnist(dataset='training', digits=np.arange(10), path='handwritten/'):
    if dataset == "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)

    images = zeros((N, rows, cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
        images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]

    return images, labels
	

#=============================
# 載入手寫字 0及1
#=============================
trainingImages, trainingLabels = load_mnist('training', digits=[0,1,2,3,4,5,6,7,8,9])

n_samples = len(trainingImages)
print('-'*40)
print('共有訓練資料:', n_samples)
print('共有訓練標籤:', len(trainingLabels))


#=============================
# 改變輸入資料之維度
#=============================
r=len(trainingImages[0])
c=len(trainingImages[0][0])

data = trainingImages.reshape((n_samples, r*c))
labels=trainingLabels.squeeze()

print('原圖尺寸=', r , '*', c)
print('向量大小=', r*c)


#=============================
# 挑出資料的部份特徵
#=============================
n_components=36

print('以PCA挑出特徵數=', n_components)
print('將原向量改為少數特徵值向量...')

pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(data)
data_pca = pca.transform(data)


#=============================
# 訓練一個分類器
#=============================
training_size=5000

print('進行分類器訓練...')

# 設定 grid search 的參數範圍
param_grid = {'C': [1, 10, 100, 1000],
              'gamma': [0.0001, 0.001, 0.01], }
			  
# 進行 grid search			  
classifier = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
#classifier = svm.SVC(gamma=0.001)
classifier.fit(data_pca[:training_size], labels[:training_size])
print('訓練總數=', training_size)
print('訓練模型=', classifier)


#=============================
# 進行預測
#=============================
testing_size=500
expected = labels[training_size:training_size+testing_size]
predicted = classifier.predict(data_pca[training_size:training_size+testing_size])


#===============================
# 比較實際值與預測值
#===============================
total=0
correct=0
for i in range(len(expected)):
    total=total+1
    if expected[i]==predicted[i]:
	    correct=correct+1
    else:
        print('正確:', expected[i], ', 誤判:', predicted[i])	

print('-'*40)		
print('測試總量=', total)
print('正確=', correct)
print('正確率=', correct/total)
print('-'*40)	
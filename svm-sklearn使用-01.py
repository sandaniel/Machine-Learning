import numpy as np
import random
from sklearn.svm import SVC

#----------------------------------
# 載入資料
#----------------------------------
def loadDataSet(fileName):
    dataMat=[]
    labelMat=[]
 
    fr=open(fileName)
 
    for line in fr.readlines():
        lineArr=line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
 
    return dataMat, labelMat


#訓練資料個數(總資料100個)
numOfTraining=80	
	
# 載入資料	
dataArr, labelArr=loadDataSet('testSet-NOT-linearSeparable.txt')

# 資料重新排序
for i in range(0, 500):
    n=random.randint(0, 99)
    m=random.randint(0, 99)
	
    t=dataArr[n]
    dataArr[n]=dataArr[m]
    dataArr[m]=t
	
    t=labelArr[n]
    labelArr[n]=labelArr[m]
    labelArr[m]=t
	
# 將list轉成ndarray, 方便切分為訓練及測試兩區段
dataND=np.array(dataArr)
labelND=np.array(labelArr)
labelND=labelND.astype(int)

# 訓練資料
X=dataND[0:numOfTraining,]
print('訓練資料')
print(X)
print('-'*60)

# 訓練資料的標籤
Y=labelND[0:numOfTraining]
print('訓練標籤')
print(Y)
print('-'*60)

# 建立模型
'''
kernel : string, optional (default=’rbf’)
Specifies the kernel type to be used in the algorithm. 
It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable.
If none is given, ‘rbf’ will be used. 
'''
clf = SVC()
clf.fit(X, Y) 

print('模型設定')
print(clf)
print('-'*60)

# 測試資料
Z = dataND[numOfTraining:100,]

# 進行預測
prediction = clf.predict(Z)
print('正確標籤: ', labelND[numOfTraining:100])
print('預測標籤: ', prediction)

# 計算正確率
compareArr=(labelND[numOfTraining:100]==prediction)
print('正確率:', len(prediction[compareArr])/len(prediction))
print('-'*60)
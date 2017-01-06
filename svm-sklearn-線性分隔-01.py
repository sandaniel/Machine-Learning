import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
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
        data1=float(lineArr[0])
        data2=float(lineArr[1])
        label=int(lineArr[2])
		
        dataMat.append([data1, data2])
        labelMat.append(label)
 
    return dataMat, labelMat


#訓練資料個數(總資料100個)
numOfTraining=100	
	
# 載入資料	
dataArr, labelArr=loadDataSet('testSet-linearSeparable.txt')
	
# 將list轉成ndarray, 方便切分為訓練及測試兩區段
dataND=np.array(dataArr)
dataND=dataND.astype(float)
labelND=np.array(labelArr)
labelND=labelND.astype(int)

# 訓練資料
X=dataND[0:numOfTraining,]

# 訓練資料的標籤
Y=labelND[0:numOfTraining]
	
# 建立分類模型
clf = svm.SVC(kernel='linear')
clf.fit(X, Y)

# 計算分隔 hyperplane
w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-2, 10)
yy = a * xx - (clf.intercept_[0]) / w[1]

# 計算穿過 support vectors 的邊界
b = clf.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])

b = clf.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])


#==========================================
# 繪圖
#==========================================
import matplotlib.pyplot as plt

# 設定字型及大小
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['font.size'] = 14

# 設定圖標題
plt.title('圖標題')

# 設定x軸及y軸標題
plt.xlabel('x軸')
plt.ylabel('y軸')

# 資料表內的grid
plt.grid(True)

# 設定x軸及y軸的尺規範圍
plt.axis([-2, 12, -10, 10])

# 繪製資料
plt.plot(dataND[labelND==1,0], dataND[labelND==1,1], 'ys')      #標籤為1
plt.plot(dataND[labelND==-1,0], dataND[labelND==-1,1], 'c^')    #標籤為-1

# 繪製 hyperplane 及穿過 support vectors 的邊界
plt.plot(xx, yy, 'k-')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')

# 繪製 support vectors
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100, facecolors='r')

# 顯示圖形
plt.show()
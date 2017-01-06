'''
Support Vector Machines are an optimization problem. 
They are attempting to find a hyperplane that divides the two classes with the largest margin. 
The support vectors are the points which fall within this margin. 
It's easiest to understand if you build it up from simple to more complex.


Hard Margin Linear SVM

In an a training set where the data is linearly separable, and you are using a hard margin (no slack allowed),
the support vectors are the points which lie along the supporting hyperplanes 
(the hyperplanes parallel to the dividing hyperplane at the edges of the margin)


Hard-Margin SVM

All of the support vectors lie exactly on the margin. 
Regardless of the number of dimensions or size of data set, 
the number of support vectors could be as little as 2.


Soft-Margin Linear SVM

But what if our dataset isn't linearly separable? We introduce soft margin SVM. 
We no longer require that our data points lie outside the margin, 
we allow some amount of them to stray over the line into the margin. 
We use the slack parameter C to control this. (nu in nu-SVM) 
This gives us a wider margin and greater error on the training dataset, 
but improves generalization and/or allows us to find a linear separation of data that is not linearly separable.


Soft-margin Linear SVM

Now, the number of support vectors depends on how much slack we allow and 
the distribution of the data. If we allow a large amount of slack, 
we will have a large number of support vectors. 
If we allow very little slack, we will have very few support vectors. 
The accuracy depends on finding the right level of slack for the data being analized. 
Some data it will not be possible to get a high level of accuracy, 
we must simply find the best fit we can.


Non-Linear SVM

This brings us to non-linear SVM. We are still trying to linearly divide the data, 
but we are now trying to do it in a higher dimensional space. 
This is done via a kernel function, which of course has its own set of parameters. 
When we translate this back to the original feature space, the result is non-linear:


enter image description here

Now, the number of support vectors still depends on how much slack we allow, 
but it also depends on the complexity of our model. 
Each twist and turn in the final model in our input space requires one or more support vectors to define. 
Ultimately, the output of an SVM is the support vectors and an alpha, 
which in essence is defining how much influence that specific support vector has on the final decision.

Here, accuracy depends on the trade-off between a high-complexity model 
which may over-fit the data and a large-margin which will incorrectly classify some of the training data
in the interest of better generalization. 
The number of support vectors can range from very few to every single data point 
if you completely over-fit your data. This tradeoff is controlled via C 
and through the choice of kernel and kernel parameters.

I assume when you said performance you were referring to accuracy, 
but I thought I would also speak to performance in terms of computational complexity. 
In order to test a data point using an SVM model, 
you need to compute the dot product of each support vector with the test point. 
Therefore the computational complexity of the model is linear in the number of support vectors. 
Fewer support vectors means faster classification of test points.
'''

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
dataArr, labelArr=loadDataSet('testSet-NOT-linearSeparable.txt')
	
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
'''
The plots below illustrate the effect the parameter C has on the separation line. 
A large value of C basically tells our model that we do not have that much faith
in our data’s distribution, and will only consider points close to line of separation.
A small value of C includes more/all the observations, allowing the margins
to be calculated using all the data in the area.
'''
clf = svm.SVC(C=0.05, kernel='linear')
clf.fit(X, Y)

print('訓練的模型:')
print(clf)

# 計算分隔 hyperplane
w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-2, 10)
yy = a * xx - (clf.intercept_[0]) / w[1]

# 計算與 hyperplane 平行的 margin, 選擇(1)或(2)
#==================
# (1)hard margin
#==================
'''
b = clf.support_vectors_[0]
print(b)
yy_down = a * xx + (b[1] - a * b[0])

b = clf.support_vectors_[-1]
print(b)
yy_up = a * xx + (b[1] - a * b[0])
'''

# 計算與 hyperplane 平行的 margin, 選擇(1)或(2)
#==================
# (2)soft margin
#==================
margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
yy_down = yy + a * margin
yy_up = yy - a * margin


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

# 繪製 hyperplane 及 margin
plt.plot(xx, yy, 'k-')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')

# 繪製 support vectors
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100, facecolors='r')

# 顯示圖形
plt.show()
import numpy as np
import pandas as pd
from sklearn.svm import SVC

#==============================
# 載入資料
#==============================
# 讀入資料, pandas.DataFrame
df=pd.read_csv('testSet-NOT-linearSeparable.txt', 
	sep='\t', 
	names=['feature01', 'feature02', 'label'])

# 訓練/測試資料個數(總資料100個)
numOfTraining=80	
numOfTesting=20

# 資料重新亂數排序
df=df.sample(frac=1)

# 所有資料及標籤
dfData=df[['feature01', 'feature02']]
dfLabel=df['label']

# 訓練資料及標籤
dfTrainingData=dfData.head(numOfTraining)
dfTrainingLabel=dfLabel.head(numOfTraining)

# 測試資料及標籤
dfTestingData=dfData.tail(numOfTesting)
dfTestingLabel=dfLabel.tail(numOfTesting)
	

#================
# 建立模型
#================
'''
kernel : string, optional (default=’rbf’)
Specifies the kernel type to be used in the algorithm. 
It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable.
If none is given, ‘rbf’ will be used. 
'''
clf = SVC()
clf.fit(dfTrainingData, dfTrainingLabel) 

print('-'*60)
print('模型設定')
print(clf)
print('-'*60)


#==================
# 進行預測
#==================
prediction = clf.predict(dfTestingData)

print('正確標籤: ', dfTestingLabel.as_matrix())
print('預測標籤: ', prediction)


# 計算正確率
correctCount=np.sum(dfTestingLabel.as_matrix()==prediction)
print(correctCount/len(prediction))

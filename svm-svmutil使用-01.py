from svmutil import *

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
        labelMat.append(int(lineArr[2]))
 
    return dataMat, labelMat


# 訓練資料數目(共100個)	
numOfTraining=80

# 載入資料	
dataArr, labelArr=loadDataSet('testSet-NOT-linearSeparable.txt')

# 訓練模型
model = svm_train(labelArr[:numOfTraining], dataArr[:numOfTraining], '-c 1')	

# 測試
p_label, p_acc, p_val=svm_predict(labelArr[numOfTraining:], dataArr[numOfTraining:], model)

# 印出結果
print('預測標籤')
p_label = list(map(int, p_label))
print(p_label)

print('實際標籤')
print(labelArr[numOfTraining:])


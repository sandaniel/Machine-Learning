from numpy import *
import numpy as np

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

	
#----------------------------------
# 從0~m中找一個非i的亂數
#---------------------------------- 
def selectJrand(i, m):
    j=i
    while(j==i):
        j=int(random.uniform(0, m))
 
    return j
	

#-------------------------------------
# 將超過H的值改為H; 小於L的值改為L
#------------------------------------- 	
def clipAlpha(aj, H, L):
    if aj>H:
        aj=H
 
    if L>aj:
        aj=L
 
    return aj

 
#==============================================================================
# SMO算法
# 1. 随机数初始化向量权重alphaa, 并计算偏移 b
# 2  初始化误差项Ei
# 3. 选取两个向量作为需要调整的点,
# 4. 令a2<new> = a2<old> + y2(E1-E2)/K
# 5. if a2<new> > V, let a2<new>=V
# 6. if a2<new> < U, let a2<new>=U
# 7. Let a1<new> = a1<old> + y1*y2*(a2<old> - a2<new>)
# 8. 以新的a1<new>及a2<new>修改Ei及b
# 9. 如達終止條件則停止, 否則再開始進行步驟3~8 
#
# 實現SVM的smo演算法
# smo: Sequential Minimal Optimization(1998)
#
# C:懲罰因數
# toler:容許誤差
# maxInter:最多遞迴次數
#
# 回傳值:
# alphas:向量權重
# b:偏移量
#------------------------------------------------------------------------------  
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix=mat(dataMatIn)
    labelMat=mat(classLabels).transpose()
	
    b = 0  #偏移量	
    m, n = shape(dataMatrix)
    alphas = mat(zeros((m,1))) #向量權重
    iter=0
 
    while(iter<maxIter):
        alphaPairsChanged=0
        for i in range(m):
            fXi=float(multiply(alphas, labelMat).T * (dataMatrix*dataMatrix[i, :].T)) + b
            Ei=fXi-float(labelMat[i])  #誤差項
 
            if((labelMat[i]*Ei < -toler) and (alphas[i]<C)) or ((labelMat[i]*Ei > toler) and (alphas[i]>0)):
                j=selectJrand(i, m)
                fXj=float(multiply(alphas, labelMat).T * (dataMatrix*dataMatrix[j,:].T)) + b
 
                Ej=fXj - float(labelMat[j]) #誤差項
                alphaIold=alphas[i].copy()
                alphaJold=alphas[j].copy()
 
                if(labelMat[i]!=labelMat[j]):
                    L=max(0, alphas[j]-alphas[i])
                    H=min(C, C+alphas[j]-alphas[i])
                else:
                    L=max(0, alphas[j] + alphas[i] - C)
                    H=min(C, alphas[j] + alphas[i])
 
                if L==H:
                    print("L==H")
                    continue
 
                eta=2.0*dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:] * dataMatrix[i,:].T - dataMatrix[j,:] * dataMatrix[j,:].T
                if eta>=0:
                    print("eta>=0")
                    continue
 
                alphas[j]-=labelMat[j]*(Ei-Ej)/eta
                alphas[j]=clipAlpha(alphas[j], H, L)
 
                if(abs(alphas[j] - alphaJold) < 0.00001):
                    print("j not moving engoth")
                    continue
 
                alphas[i]+=labelMat[j]*labelMat[i]*(alphaJold-alphas[j])
 
                b1=b - Ei - labelMat[i]*(alphas[i] - alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
 
                b2= b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[j,:].T - labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[j,:]*dataMatrix[j,:].T
 
                if(0<alphas[i]) and (C>alphas[i]):
                    b=b1
                elif(0<alphas[j]) and (C>alphas[j]):
                    b=b2
                else:
                    b=(b1+b2)/2.0
 
                alphaPairsChanged+=1
 
                #print("iter: %d i:%d, pairs changed %d" %(iter, i, alphaPairsChanged))
 
        if(alphaPairsChanged==0):
            iter+=1
        else:
            iter=0
 
        print("iteration number:%d"%iter)
 
    return b, alphas                   
#==============================================================================
 

#載入測試檔 
dataArr, labelArr=loadDataSet('testSet.txt')
dataND=np.array(dataArr)
labelND=np.array(labelArr)

#smo分類模型建立 
b, alphas=smoSimple(dataArr, labelArr, 0.05, 0.001, 40)

alphasND=np.array(alphas)
alphasND=alphasND.transpose()


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
plt.axis([-1, 10, -5, 5])

# 繪製資料
plt.plot(dataND[labelND==1,0], dataND[labelND==1,1], 'ys', dataND[labelND==-1,0], dataND[labelND==-1,1], 'co')
plt.plot(dataND[(alphasND!=0)[0],0], dataND[(alphasND!=0)[0],1], 'r+')


#=======================================================
# hyperplane 分隔超平面(在此是一條線)
# 方程式=(-b-m0x)/m1
#-------------------------------------------------------
w=mat([0,0])

for i in range(len(alphas)):
    w=w+mat(dataArr[i])*labelArr[i]*float(alphas[i])
  
w=array(w)
 
m0=float(w[0][0])
m1=float(w[0][1])
 
b=float(b)
 
k1=[0, 10]
k2=[(-1*b-m0*k1[0])/m1,  (-1*b-m0*k1[1])/m1]

print(k2)
 
plt.rc('lines', linewidth=2)
plt.plot(k1, k2, '-m')
#=======================================================

 
# 顯示圖表
plt.show()
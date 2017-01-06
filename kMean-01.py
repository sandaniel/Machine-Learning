'''
K-means is a clustering algorithm that tries to partition a set of points 
into K sets (clusters) such that the points in each cluster tend to be near each other. 
It is unsupervised because the points have no external classification.
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.linalg as linalg


#-----------------------------------------------------------------
#計算兩個向量的歐基米得距離
def dist_raw(v1, v2):
	delta=v1-v2
	return linalg.norm(delta.tolist())
#-----------------------------------------------------------------

#-----------------------------------------------------------------
#計算ndarray中的某元素值個數
def count_array(nd, v):
	cnt=0
	for i in range(len(nd)):
		if nd[i]==v:
			cnt=cnt+1
			
	return cnt
#-----------------------------------------------------------------

#-----------------------------------------------------------------
#計算ndarray中的加總
def sum_array(nd, label, v):
	d1=0
	d2=0
	for i in range(len(label)):
		if label[i]==v:
			d1=d1+nd[i][0]
			d2=d2+nd[i][1]			
			
	return np.array([d1, d2])
#-----------------------------------------------------------------

# 產生常態分配亂數成績
chi01=np.random.randn(100)*10+55
eng01=np.random.randn(100)*10+55

chi02=np.random.randn(100)*10+45
eng02=np.random.randn(100)*10+45

# 修正資料
chi01=chi01.clip(0, 100)
eng01=eng01.clip(0, 100)
chi02=chi02.clip(0, 100)
eng02=eng02.clip(0, 100)

print('chi01', chi01)
print('chi02', chi02)
print('eng01', eng01)
print('eng02', eng02)

data01=np.concatenate([chi01, chi02])
data02=np.concatenate([eng01, eng02])
label=np.zeros(200)

data=np.array([data01, data02])
data=data.reshape(200, 2)
np.random.shuffle(data)
print(data)


#分簇
group=2
data1_center=data[0]
data2_center=data[1]

label[0]=1
label[1]=2

for i in range(2, 200):
	d1=dist_raw(data1_center, data[i])
	d2=dist_raw(data2_center, data[i])
	
	print('與群集1的中心距離:' , d1)
	print('與群集2的中心距離:' , d2)
	
	if d1>d2:
		label[i]=2
		m=sum_array(data, label, 2)
		data2_center=m/count_array(label, 2)
	else:
		label[i]=1
		m=sum_array(data, label, 1)
		data1_center=m/count_array(label, 1)
		

for k in range(1, 200):
	for i in range(0, 200):
		d1=dist_raw(data1_center, data[i])
		d2=dist_raw(data2_center, data[i])
		
		if d1>d2:
			label[i]=2
			m=sum_array(data, label, 2)
			data2_center=m/count_array(label, 2)
		else:
			label[i]=1
			m=sum_array(data, label, 1)
			data1_center=m/count_array(label, 1)
			
			
print('群集1中心:', data1_center)
print('群集2中心:', data2_center)		
		
		
		
# 設定字型及大小
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['font.size'] = 14

# 設定圖標題
plt.title('國文-英文成績分佈(依入學方式分別)')

# 設定x軸及y軸標題
plt.xlabel('國文成績')
plt.ylabel('英文成績')

# 資料表內的grid
plt.grid(True)

# 設定x軸及y軸的尺規範圍
plt.axis([-5, 105, -5, 105])

# 繪製資料
data01=data[label==1]
data02=data[label==2]
print(data01)
print(data01.shape)

plt.plot(data01[:,0], data01[:,1], 'yv')
plt.plot(data02[:,0], data02[:,1], 'g^')

plt.plot(data1_center[0], data1_center[1], 'ro')
plt.plot(data2_center[0], data2_center[1], 'ro')

# 設定資料說明
plt.legend(['群組一', '群組二'], numpoints=1, loc='upper left')

# 顯示圖表
plt.show()
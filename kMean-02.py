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

#
total_size=500 #偶數
group_size=5
repeat_time=100

# 產生常態分配亂數成績
chi01=np.random.randn(int(total_size/2))*10+60
eng01=np.random.randn(int(total_size/2))*10+60

chi02=np.random.randn(int(total_size/2))*10+40
eng02=np.random.randn(int(total_size/2))*10+40

# 修正資料
chi01=chi01.clip(0, 100)
eng01=eng01.clip(0, 100)
chi02=chi02.clip(0, 100)
eng02=eng02.clip(0, 100)

data01=np.concatenate([chi01, chi02])
data02=np.concatenate([eng01, eng02])
label=np.zeros(total_size)

data=np.array([data01, data02])
data=data.reshape(total_size, 2)
np.random.shuffle(data)


#分簇
data_center=data[0:group_size]
for i in range(group_size):	
	label[i]=i

dist=np.zeros(group_size)

for i in range(group_size, total_size):
	min=999
	idx=-1
	
	for k in range(group_size):
		dist[k]=dist_raw(data_center[k], data[i])
		if dist[k]<min:
			lab=k
			min=dist[k]
	
	label[i]=lab
	m=sum_array(data, label, lab)
	data_center[lab]=m/count_array(label, lab)	
		

for t in range(1, repeat_time):
	for i in range(0, total_size):
		for k in range(group_size):
			dist[k]=dist_raw(data_center[k], data[i])
		
		
		min=999
		idx=-1
		for k in range(group_size):
			if dist[k]<min:
				lab=k
				min=dist[k]
		
		label[i]=lab
		m=sum_array(data, label, lab)
		data_center[lab]=m/count_array(label, lab)
			
			
for k in range(group_size):			
	print('群集', k, '中心:', data_center[k])
		
		
		
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
for i in range(group_size):
	print_data=data[label==i]	
	plt.scatter(print_data[:,0], print_data[:,1], c=np.random.rand(3,1))
	plt.plot(data_center[i][0], data_center[i][1], 'ro')



# 顯示圖表
plt.show()
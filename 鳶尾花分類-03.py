from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
import numpy as np
import random


#載入資料
data=load_iris()

features=data['data']
feature_names=data['feature_names']
target=data['target']


#整理資料
load_data=features[target!=0,:]
load_target=target[target!=0]
load_size=len(load_target)
print('共有資料:', len(load_target))


#原始資料切分為訓練資料及測試資料
total_size = len(load_target)
training_ratio=0.8
rnd_array=np.array(random.sample(range(load_size),load_size))


#訓練資料及標示
training_data=load_data[rnd_array<total_size*training_ratio, :]
training_target=load_target[rnd_array<total_size*training_ratio]
training_size=len(training_data)


#測試資料及標示
testing_data=load_data[rnd_array>=total_size*training_ratio, :]
testing_target=load_target[rnd_array>=total_size*training_ratio]
testing_size=len(testing_data)

print('訓練資料個數:', training_size)
print('測試資料個數:', testing_size)


#----------------
# 進行訓練
#----------------
acc_rate=-10
acc_label='**'
acc_index=0
thresh=0

for i in range(len(feature_names)):
	data=training_data[training_target==2,i].copy()
	data.sort()
	
	now_training_data=training_data[:,i]
	
	for k in data:
		a=training_target[now_training_data>=k]
		b=training_target[now_training_data<k]

		correct=np.sum(a==2) + np.sum(b==1)
		this_acc=correct/training_size
		
		if this_acc>acc_rate:
			acc_rate=this_acc
			acc_label=feature_names[i]
			acc_index=i
			thresh=k			

print('----訓練結果----')			
print('訓練正確率:', acc_rate)
print('採用特徵:', acc_label)
print('分類門檻:', thresh)


#----------------
# 進行測試
#----------------
now_testing_data=testing_data[:, acc_index]
a=testing_target[now_testing_data>=thresh]
b=testing_target[now_testing_data<thresh]

testing_correct=np.sum(a==2) + np.sum(b==1)
testing_acc_rate=testing_correct/testing_size
		

print('----訓練結果----')
print('測試正確個數:', testing_correct)			
print('測試正確率:', testing_acc_rate)


#繪圖-訓練資料

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['font.size']=14

axes[0].set_title('訓練資料分佈')
axes[0].set_xlabel(feature_names[2])
axes[0].set_ylabel(feature_names[3])

for t, marker, c in zip(range(3), '>ox', 'rgb'):	
	axes[0].scatter(training_data[training_target==t,2], training_data[training_target==t,3], marker=marker, c=c)

axes[0].grid(True)	



#繪圖-測試資料
axes[1].set_title('訓練資料分佈')
axes[1].set_xlabel(feature_names[2])
axes[1].set_ylabel(feature_names[3])

for t, marker, c in zip(range(3), '>ox', 'rgb'):	
	axes[1].scatter(testing_data[testing_target==t,2], testing_data[testing_target==t,3], marker=marker, c=c)

axes[1].grid(True)	

plt.show()
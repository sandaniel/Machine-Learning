from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
import numpy as np

data=load_iris()

features=data['data']
feature_names=data['feature_names']
target=data['target']

test_data=features[target!=0,:]
test_target=target[target!=0]

acc_rate=-10
acc_label='**'
thresh=0

for i in range(len(feature_names)):
	data=test_data[test_target==2,i].copy()
	data.sort()
	
	now_test_data=test_data[:,i]
	
	for k in data:
		a=test_target[now_test_data>=k]
		b=test_target[now_test_data<k]

		correct=np.sum(a==2) + np.sum(b==1)
		this_acc=correct/len(test_target)
		
		if this_acc>acc_rate:
			acc_rate=this_acc
			acc_label=feature_names[i]
			thresh=k
		
print('正確率:', acc_rate)
print('分類標籤:',acc_label)
print('門檻值:',thresh)

for t, marker, c in zip(range(3), '>ox', 'rgb'):	
	plt.scatter(features[target==t,2], features[target==t,3], marker=marker, c=c)
	
plt.grid(True)	
plt.show()
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
import numpy as np

data=load_iris()

features=data['data']
feature_names=data['feature_names']
target=data['target']

a=features[target==1,0]
print(a.max())

for t, marker, c in zip(range(3), '>ox', 'rgb'):	
	plt.scatter(features[target==t,2], features[target==t,3], marker=marker, c=c)
	
plt.grid(True)	
plt.show()
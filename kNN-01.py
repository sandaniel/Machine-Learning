'''
(file/a1.txt)
Python program XML skill
(file/a2.txt)
Python expressive sample than ever examples 
(file/a3.txt)
Programming Python examples and sample
'''

'''
K-nearest neighbors is a classification (or regression) algorithm that in order to determine
the classification of a point, combines the classification of the K nearest points. 
It is supervised because you are trying to classify a point based on the 
known classification of other points.
'''


import os.path
from sklearn.feature_extraction.text import CountVectorizer as cv
import scipy as sp
import nltk.stem
from sklearn.feature_extraction.text import TfidfVectorizer

#-----------------------------------------------------------------
#計算兩個向量的歐基米得距離
def dist_raw(v1, v2):
	v1_normalized=v1/sp.linalg.norm(v1.toarray())  #將向量長度歸1
	v2_normalized=v2/sp.linalg.norm(v2.toarray())
	
	delta=v1_normalized-v2_normalized
	return sp.linalg.norm(delta.toarray())
#-----------------------------------------------------------------


#----------------------------------------------------------------------------
#將同義字統一的向量計數器
english_stemmer=nltk.stem.SnowballStemmer('english')
class StemmedCountVectorizer(cv):
	def build_analyzer(self):
		analyzer=super(StemmedCountVectorizer, self).build_analyzer()
		return lambda doc:(english_stemmer.stem(w) for w in analyzer(doc))
#----------------------------------------------------------------------------


#----------------------------------------------------------------------------
#將同義字統一的向量計數器, TF-IDF, 詞頻-反轉文檔頻率
english_stemmer=nltk.stem.SnowballStemmer('english')
class StemmedTfidfVectorizer(TfidfVectorizer):
	def build_analyzer(self):
		analyzer=super(TfidfVectorizer, self).build_analyzer()
		return lambda doc:(english_stemmer.stem(w) for w in analyzer(doc))
#----------------------------------------------------------------------------


#讀入存在file資料夾中的檔案
posts=[open(os.path.join('./file/', f)).read() for f in os.listdir('./file/')]

for i in range(len(posts)):
	print('文件', i , ':', posts[i])


#出現在90%以上及小於20%以下文章中的字, 將不列入向量計算
#刪除英文停用字
#(1)
'''
vect=StemmedCountVectorizer(min_df=0.2, max_df=0.9, stop_words='english') 
'''
#(2)
vect=StemmedTfidfVectorizer(min_df=1, stop_words='english') 

X_train=vect.fit_transform(posts)
num_samples, num_features=X_train.shape

print('共有樣本:', num_samples, '個')
print('共有特徵:', num_features, '個')
print('特徵:', vect.get_feature_names())

for i in range(0, num_samples):
	print('第', i, '個特徵向量:', X_train.getrow(i).toarray())


#準備一個待分類文件內容
new_post='How to learn Python with sample'
new_post_vec=vect.transform([new_post])
print('待分類特徵向量: ', new_post_vec.toarray())


#計算待分類文件與樣本特徵向量的距離
min_dist=999
best_vect=-1
for i in range(0, num_samples):
	dist=dist_raw(new_post_vec, X_train.getrow(i))	
	print('與第', i, '個樣本距離:', dist)
	
	if dist<min_dist:
		min_dist=dist
		best_vect=i


#印出最近距離者
print('最近距離:', min_dist)
print('最接近樣本:', best_vect)
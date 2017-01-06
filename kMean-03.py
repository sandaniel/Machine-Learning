import sklearn.datasets
import os.path
from sklearn.feature_extraction.text import CountVectorizer as cv
import scipy as sp
import nltk.stem
from sklearn.feature_extraction.text import TfidfVectorizer

#----------------------------------------------------------------------------
#將同義字統一的向量計數器, TF-IDF, 詞頻-反轉文檔頻率
english_stemmer=nltk.stem.SnowballStemmer('english')
class StemmedTfidfVectorizer(TfidfVectorizer):
	def build_analyzer(self):
		analyzer=super(TfidfVectorizer, self).build_analyzer()
		return lambda doc:(english_stemmer.stem(w) for w in analyzer(doc))
#----------------------------------------------------------------------------


MLCOMP_DIR = r'e:\pythonTest\data'

data=sklearn.datasets.load_mlcomp('20news-18828', mlcomp_root=MLCOMP_DIR)

print('讀入檔名:', data.filenames)
print('target_names:', data.target_names)
print('target:', data.target)
print('DESCR:', data.DESCR)


train_data=sklearn.datasets.load_mlcomp('20news-18828', 'train', mlcomp_root=MLCOMP_DIR)
test_data=sklearn.datasets.load_mlcomp('20news-18828', 'test', mlcomp_root=MLCOMP_DIR)

print('訓練資料集大小:', len(train_data.filenames))
print('測試資料集大小:', len(test_data.filenames))


#讀入訓練資料內容
posts=[open(f, encoding='utf-8', errors='ignore').read() for f in train_data.filenames]


#產生vector
vectorizer=StemmedTfidfVectorizer(max_df=0.5, min_df=10, stop_words='english')
vectorized=vectorizer.fit_transform(posts)
num_samples, num_features=vectorized.shape

print('樣本數:', num_samples)
print('特徵數:', num_features)


#----------------------
#進行KMeans分群
#----------------------
num_clusters=50
from sklearn.cluster import KMeans
km=KMeans(n_clusters=num_clusters, init='random', n_init=1, verbose=1)
km.fit(vectorized)

print('各文件分類標箋:', km.labels_)



#----------------------
#測試資料分類
#----------------------
#讀入訓練資料內容
test_posts=[open(f, encoding='utf-8', errors='ignore').read() for f in test_data.filenames]
for i in range(len(test_posts)):
	test_posts_vec=vectorizer.transform([test_posts[i]])
	test_posts_label=km.predict(test_posts_vec)[0]
	print('文件',i,'分類:', test_posts_label)
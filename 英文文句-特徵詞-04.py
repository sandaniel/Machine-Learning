#下載測試資料: http://mlcomp.org/datasets/379 
#資料名稱:dataset-379-20news-18828-JJIWS.zip
#存在程式所在位置的子資料夾<data>中, 解壓縮它.

from sklearn.feature_extraction.text import CountVectorizer
import os
import scipy as sp
import numpy as np

def dist_norm(v1, v2):
    v1_normalized=v1/sp.linalg.norm(v1)
    v2_normalized=v2/sp.linalg.norm(v2)
    delta=v1_normalized - v2_normalized
    return sp.linalg.norm(delta)

# 讀入資料, text file
DIR="doc/"
posts=[open(os.path.join(DIR, f), 'r', encoding='utf8').read() for f in os.listdir(DIR)]

doc01=posts[0]
doc02=posts[1]

print('------原文------------------------------------------')
print(doc01.encode('raw_unicode_escape').decode('utf-8'))
print(doc02.encode('raw_unicode_escape').decode('utf-8'))

# 找出文章中的特徵文字
vectorizer=CountVectorizer(min_df=1)
content=[]
content.append(doc01)
content.append(doc02)

stat=vectorizer.fit_transform(content)

print('------特徵文字------------------------------------------')
print(vectorizer.get_feature_names())

print('------特徵文字統計------------------------------------------')
print(stat.toarray().transpose())

print('------印出兩者皆有的特徵文字統計------------------------------------------')
fea=stat.toarray().transpose()
fea_names=vectorizer.get_feature_names()
k=0
for i, j in fea:
    k=k+1
    if i>0 and j>0:
        print(i, j)
        print(fea_names[k])


print('-------正規化----------------------------')
print(stat.toarray())
v1=stat.toarray()[0]
v2=stat.toarray()[1]

print('-------正規化後的距離----------------------------')
print(dist_norm(v1, v2))
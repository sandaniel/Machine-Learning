from sklearn.feature_extraction.text import CountVectorizer

# 讀入資料, text file
f=open('doc01.txt', 'r', encoding='utf8')
doc01=f.read()

f=open('doc02.txt', 'r', encoding='utf8')
doc02=f.read()

print('------原文------------------------------------------')
print('<<文件1>>')
print(doc01.encode('raw_unicode_escape').decode('utf-8'))
print('<<文件2>>')
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
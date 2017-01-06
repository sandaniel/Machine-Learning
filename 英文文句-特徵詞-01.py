from sklearn.feature_extraction.text import CountVectorizer

# 讀入資料, text file
f=open('doc.txt', 'r', encoding='utf8')
doc=f.read()
print('------原文------------------------------------------')
print(doc.encode('raw_unicode_escape').decode('utf-8'))

# 找出文章中的特徵文字
vectorizer=CountVectorizer(min_df=1)
content=[]
content.append(doc)
stat=vectorizer.fit_transform(content)
print('------特徵文字------------------------------------------')
print(vectorizer.get_feature_names())

print('------特徵文字統計------------------------------------------')
print(stat.toarray()[0])
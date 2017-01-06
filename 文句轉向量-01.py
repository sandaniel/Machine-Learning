from sklearn.feature_extraction.text import CountVectorizer

'''
When using a float in the range [0.0, 1.0] they refer to the document frequency. 
That is the percentage of documents that contain the term.
When using an int it refers to absolute number of documents that hold this term.

Consider the example where you have 5 text files (or documents). 
If you set max_df = 0.6 then that would translate to 0.6*5=3 documents. 
If you set max_df = 2 then that would simply translate to 2 documents.
'''

vectorizer = CountVectorizer(min_df=1)

'''
以下的測試有4篇文件, 每篇文件應有一種分類(在此程式未標示).
'''

corpus = [
    'This is the first document.',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?',
]

analyze = vectorizer.build_analyzer()
X = vectorizer.fit_transform(corpus)

print(X)
print('-'*60)

print(X.toarray())
print('-'*60)

print(vectorizer.get_feature_names())
print('-'*60)

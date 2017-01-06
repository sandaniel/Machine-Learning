from sklearn.feature_extraction.text import TfidfVectorizer

'''
參數參考:

(1)
min_df及max_df:
When using a float in the range [0.0, 1.0] they refer to the document frequency. 
That is the percentage of documents that contain the term.
When using an int it refers to absolute number of documents that hold this term.
Consider the example where you have 5 text files (or documents). 
If you set max_df = 0.6 then that would translate to 0.6*5=3 documents. 
If you set max_df = 2 then that would simply translate to 2 documents.
  
(2)
smooth_idf=False, idf(d, t) = log [ n / df(d, t) ] + 1
smooth_idf=True, idf(d, t) = log [ (1 + n) / 1 + df(d, t) ] + 1

(3)
norm=None, 表示不執行vector正規化.
norm='l1', 表示會將vector的曼哈頓距離調整為1
norm='l2', 表示會將vector的歐基米得距離調整為1

(4)
use_idf=False, 僅使用tf, 不用idf
use_idf=True, 使用 tf及idf
'''


'''
*************************
***僅使用tf, 不用idf時***
*************************
例如:
[[1, 1],
 [0, 1]] 

(1)norm='l1',
  其tf矩陣= [[1/2, 1/2],
             [0/1, 1/1]]  
          = [[0.5, 0.5],
             [  0,   1]]
  第1個vector的曼哈頓距離=1/2 + 1/2 = 1
			 
(2)norm='l2', 
   sqrt(1*1+1*1)=1.414
   sqrt(0+1*1)=1
   因此, 以l2正規化, 
   其tf矩陣= [[1/1.414, 1/1.414],
              [    0/1,     1/1]] 
 
   第1個vector的歐基米得距離=(1/sqrt(2))**2 + (1/sqrt(2))**2 = 1/2 + 1/2 = 1
'''

'''
***********************
***同時使用tf及idf時***
***********************
例如:
[[1, 1],
 [0, 1]] 

其idf=[log(2/1)+1, log(2/2)+1] = [1.693, 1]

原vector=[[1*1.693, 1*1]
          [0*1.693, 1*1]]
        =[[1.693, 1]
          [    0, 1]]
		  
(1)norm='l1',
  其tf矩陣= [[1.693/2.693, 1/2.693],
             [        0/1, 1/1]]  
          = [[0.6287, 0.3713],
             [     0,      1]]			 
  第1個vector的曼哈頓距離=0.6287+0.3713 = 1
			 
(2)norm='l2', 
   sqrt(1.693**2+1**2)=1.9662
   sqrt(0+1*1)=1
   因此, 以l2正規化, 
   其tf矩陣= [[1.693/1.9662, 1/1.9662],
              [    0/1     , 1/1]] 
		   = [[0.861, 0.509],
              [    0,     1]] 	  
 
   第1個vector的歐基米得距離=sqrt(0.861**2 + 0.509**2)=1
'''


vectorizer = TfidfVectorizer(min_df=1, smooth_idf=False, norm='l1', use_idf=True)

corpus = ["This is very strange",
          "This is very nice"]

X = vectorizer.fit_transform(corpus)

print(X)
print('-'*60)

print(X.toarray())
print('-'*60)

print(vectorizer.get_feature_names())
print('-'*60)

idf = vectorizer.idf_
print(dict(zip(vectorizer.get_feature_names(), idf)))
print('-'*60)

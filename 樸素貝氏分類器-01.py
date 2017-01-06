from sklearn import datasets

#3種不同的Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

'''
GaussianNB, 其適用特徵是連續型資料, 是常態分配, 如身高, 體重, 成績.
'''

'''
MultinomialNB, 其適用特徵是離散型資料, 如出現次數; 但如tf-idf(小數型態值)的分類在實務運作時也可用.
The multinomial Naive Bayes classifier is suitable for classification with discrete features
(e.g., word counts for text classification). 
The multinomial distribution normally requires integer feature counts. 
However, in practice, fractional counts such as tf-idf may also work.
'''

'''
BernoulliNB, 其適用特徵是二元型式的離散型資料, 如真/假.
Like MultinomialNB, this classifier is suitable for discrete data. 
The difference is that while MultinomialNB works with occurrence counts, 
BernoulliNB is designed for binary/boolean features.
'''

#載入iris data
iris = datasets.load_iris()

#iris data
print(iris.data)
print('-'*60)

#iris target
print(iris.target)
print('-'*60)

#載入3種樸素貝氏演算法
gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()

#準備某個測試資料
testData=[[3, 5, 7, 5]]
print(testData)
print('-'*50)

#測試分類
y_pred = gnb.fit(iris.data, iris.target).predict(testData)
print('Gaussian 分類:', y_pred)

y_pred = mnb.fit(iris.data, iris.target).predict(testData)
print('Multinomial 分類:', y_pred)

y_pred = bnb.fit(iris.data, iris.target).predict(testData)
print('Bernoulli 分類:', y_pred)

print('-'*50)
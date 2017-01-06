'''
結巴中文: github.com/fxsjy/jieba
jieba.cut參數參考:
cut_all=True, 全模式
我来到北京清华大学 ==> 我/ 来到/ 北京/ 清华/ 清华大学/ 华大/ 大学
cut_all=False, 精確模式
我来到北京清华大学 ==> 我/ 来到/ 北京/ 清华大学
'''

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


#=================================================================
#對文章進行分詞
#=================================================================
import jieba
import os

#讀入待分詞的文檔(指定資料夾)
#originText = open('speech.txt', 'r', encoding='utf8').read()

DIR="corpus/"  #文章存放資料夾
posts=[open(os.path.join(DIR, f), 'r', encoding='utf-8-sig').read() for f in os.listdir(DIR)]


#用來儲放所有文章的list
corpus=[]

#每篇每篇文章分別斷詞
cnt=0
poun=('，','。','！','：','「','」','…','、','？','【','】','.',':','?',';','!','~','`','+','-','<','>','/','[',']','{','}',"'",'"', '\n')
for originText in posts:
	#刪除標點符號
	filteredText = "".join(c for c in originText if c not in (poun))
	

	#設定停用詞
	stopwords = {}.fromkeys([ line.rstrip() for line in open('stopwords.txt', encoding='utf8')])
	#stopwords = {}.fromkeys(['的', '是'])


	#以精確模式分詞
	words = jieba.cut(filteredText, cut_all=False) 


	#去除停用詞後, 以' '分隔各詞
	final = ''
	for seg in words:
		if seg not in stopwords:
			seg+=' '
			final += seg
	
	#將分詞後的字串加入list中
	corpus.append(final)
	
	cnt+=1
	print(cnt)
	
	print(final)
	print('*'*60)

#印出所有文章的list
#print(corpus)


#=================================================================
#corpus是一個list, 存放各個已被分詞的文章字串.
#將corups內的每一個字串轉成相對應的向量
#=================================================================
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(min_df=1, smooth_idf=False, norm='l2', use_idf=True)

#將corpus轉成向量
train_data = vectorizer.fit_transform(corpus)

print(train_data)
print('-'*60)

#轉成的向量
print(train_data.toarray())
print('-'*60)

#每個分量代表的單詞
print(vectorizer.get_feature_names())
print('-'*60)

print('共', len(corpus), '篇文章')
print('-'*60)

print('共', len(vectorizer.get_feature_names()), '個特徵詞')
print('-'*60)

#idf = vectorizer.idf_
#print(dict(zip(vectorizer.get_feature_names(), idf)))
#print('-'*60)



#=================================================================
#將測試資料與訓練資料比對, 找出最接近的文章, 顯示該文章的類別
#=================================================================
#3種不同的Naive Bayes
#from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import sys
#from sklearn.naive_bayes import BernoulliNB


#載入3種樸素貝氏演算法
#gnb = GaussianNB()
mnb = MultinomialNB()
#bnb = BernoulliNB()


#測試分類
testStr1 = u'「永不言敗，不斷創造新局」是強者陳焜耀的成功哲學，與之伴隨的，必然是一部為之讚嘆的傳奇。他的人生上半場扛起家族重擔，扭轉了百年企業崩盤的結局從此，羽絨霸主Yao Chen在國際間留名他，就是合隆毛廠總裁陳焜耀。在那段黑暗中追逐光亮的日子，他不得不靠著鎮靜劑與安眠藥艱難前行，他曾以為這種暗自吞下苦痛、手撫滄桑的日子不會有盡頭，' 
testStr2 = u'但這一切都被細心的大兒子看在眼裡。因著兒子，他參加了生平第一次的ING路跑；因著兒子結識超馬好手林義傑，暫時抽離身心俱疲的商界，轉戰另一個戰場，挑戰「四大極地超級馬拉松」（The 4 Deserts Race Series）。兩年的征途在那遙遠的地方，他留下了與兒子的足跡，還有，通透、真摯的人生回望：生命的轉彎來自翻過一頁，揮灑另一頁――2012年，'  
testStr3 = u'首次征戰撒哈拉沙漠，發現原來「極地超級馬拉松」是一種體力極限與環境考驗的生存活動，超馬初體驗如逃命般完賽！'  
testStr4 = u'入陣，只為此生坦蕩――從商場轉戰另一個人生里程，即便場域相異，所秉持的信念卻相同，每場仗都要無愧於心的完結，因此，2013年再次啟程，來到世界最荒涼的阿他加馬荒漠與一望無際的戈壁，這趟走進荒野的旅途，他在原始中醒悟，人生總有路需要獨行，所以與兒子的同行之路更顯珍貴。兒子們，是引領他在生命長夜裡的微隱星光。' 
testStr5 = u'以愛之名 再戰沙場――每個人的心中都藏著一塊缺口，即使歲月流轉也不曾消逝，在他心裡，小兒子始終是心中的缺憾，在最需要陪伴的光陰裡，他扛起家業卻成了缺席的父親，所幸透過超馬，2014年他終於有機會帶著小兒子一起看世界，曾經疏離的父子情逐漸在約旦瓦地倫山谷、智利聖彼得小鎮的賽事中微芒綻放。' 
testStr6 = u'領略人生最大寶藏――經歷了撒哈拉、阿他加馬、戈壁之後，2014年底，父子三人一同來到了世界的盡頭―南極。幻域般的雪白大陸危機四伏，小兒子的貼心、大兒子的智慧陪他熬過雪地衝擊，於此同時，他領悟到人生半百，走過風霜，看盡斑駁，心中最惦念的終究不是輝煌，而是與子相依的時光。最真誠的文字、最珍貴的極地照片，帶你走進強者的「戰鬥人生」！'

testStr=testStr1+testStr2+testStr3+testStr4+testStr5+testStr6

#刪除標點符號
testStr = "".join(c for c in testStr if c not in (poun))

#以精確模式分詞
testWords = jieba.cut(testStr, cut_all=False) 

#計算測試詞向量
testVector=np.zeros([1, len(vectorizer.get_feature_names())])
for testSeg in testWords:
	try:		
		if testSeg not in stopwords:
			k=vectorizer.get_feature_names().index(testSeg)
			testVector[0, k]+=1
	except:
		print('找不到:', testSeg.encode(sys.stdin.encoding, "replace").decode(sys.stdin.encoding))

		
#假設20篇文章的分類	
docClass=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
print('*'*60)

#各文章積分
cnt=0
best=-99
bestIndex=0
for v in train_data.toarray():
	cnt+=1
	thisLikely=np.sum(v*testVector)
	
	if thisLikely>best:
		best=thisLikely
		bestIndex=cnt
	
	print('第', cnt, '篇文章積分:', np.sum(v*testVector))

print('-'*60)
print('最接近文章', bestIndex, ' , 積分=', best);
	

#進行預測
y_pred = mnb.fit(train_data.toarray(), docClass).predict(testVector)

print('-'*50)
print('最接近文章:')
print(corpus[y_pred])
print('-'*50)

print(cnt, ' Multinomial 分類:', y_pred)
print('-'*50)
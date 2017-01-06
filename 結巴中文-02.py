'''
結巴中文: github.com/fxsjy/jieba
'''

import jieba

text = open('speech.txt', 'r', encoding='utf8').read()


#進行分詞
'''
cut_all=True, 全模式
我来到北京清华大学 ==> 我/ 来到/ 北京/ 清华/ 清华大学/ 华大/ 大学

cut_all=False, 精確模式
我来到北京清华大学 ==> 我/ 来到/ 北京/ 清华大学
'''

out = "".join(c for c in text if c not in ('，','。','！','：','「','」','…','、','？','【','】','.',':','?',';','!','~','`','+','-','<','>','/','[',']','{','}',"'",'"'))

words = jieba.cut(out, cut_all=False)# 精确模式

print(" ".join(words))  
print('-'*60)


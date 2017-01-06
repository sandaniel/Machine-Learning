import numpy as np

# 讀入資料
data=np.genfromtxt('iris.csv',  dtype=[('sepal_length', 'f'), ('sepal_width', 'f'), ('petal_length', 'f'), ('petal_width', 'f'), ('speices', 'U25')], delimiter=',', usecols=(0, 1, 2, 3, 4), unpack=True)

# 整理資料
petal_length=data['petal_length']
petal_width=data['petal_width']
speices=data['speices']

not_setosa_petal_length=petal_length[speices!='setosa']
not_setosa_petal_width=petal_width[speices!='setosa']
not_setosa_speices=speices[speices!='setosa']

# 訓練資料 + 測試資料共100筆
x = np.random.rand(100, 5)

# 前80筆作為訓練資料, 後20筆作為測試資料
indices = np.random.permutation(x.shape[0])
training_idx, test_idx = indices[:80], indices[80:]

# 進行訓練
t=0;
success_rate=-100

for idx in training_idx:
    one=not_setosa_petal_width[idx]
    a1 = not_setosa_speices[not_setosa_petal_width >= one]
    a2 = not_setosa_speices[not_setosa_petal_width < one]

    acc1=len(a1[a1 == 'virginica'])
    acc2=len(a2[a2=='versicolor'])
    acc=acc1+acc2

    if (acc/len(not_setosa_speices)) > success_rate:
        success_rate=acc/len(not_setosa_speices)
        t=one

# 印出資料
print('訓練正確率:', success_rate)
print('訓練閥值:', t)

# 進行測試
test_set_not_setosa_speices=not_setosa_speices[test_idx]
test_set_not_setosa_petal_width=not_setosa_petal_width[test_idx]

a3 = test_set_not_setosa_speices[test_set_not_setosa_petal_width >= t]
a4 = test_set_not_setosa_speices[test_set_not_setosa_petal_width < t]

acc3 = len(a3[a3 == 'virginica'])
acc4 = len(a4[a4 == 'versicolor'])

print('應該是 virginica:', a3)
print('應該是 versicolor:', a4)
print('測試正確率:', (acc3 + acc4)/len(test_set_not_setosa_speices))

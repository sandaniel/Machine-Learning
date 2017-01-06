import numpy as np

# 讀入資料
data = []
labels = []

with open('seeds.tsv'.format('seeds')) as ifile:
    for line in ifile:
        tokens = line.strip().split('\t')
        data.append([float(tk) for tk in tokens[:-1]])
        labels.append(tokens[-1])

data = np.array(data)
labels = np.array(labels)

# 訓練資料 + 測試資料共210筆
x = np.random.rand(210, 5)

# 前150筆作為訓練資料, 後60筆作為測試資料
indices = np.random.permutation(x.shape[0])
training_data, training_label = data[indices[:150]], labels[indices[:150]]
test_data, test_label=data[indices[150:]], labels[indices[150:]]

# 進行knn比對
correct=0
error=0

for d in range(len(test_data)):
    min=99999;
    lab=''
    for i in range(len(training_data)):
        m=((test_data[d]-training_data[i])**2).sum()
        if m<min:
            min=m
            lab=training_label[i]

    print('***', test_label[d], "***", lab, "****", min)
    if test_label[d] == lab:
        correct=correct+1
    else:
        error=error+1


# 印出資料
print('正確=', correct)
print('錯誤=', error)
print('正確率=', correct/(correct+error))
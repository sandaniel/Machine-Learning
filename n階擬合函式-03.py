import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

# 讀入資料
data=np.genfromtxt('web_traffic.tsv', dtype="i", delimiter='\t', usecols=(0,1), unpack=True)

# 整理資料
data=np.transpose(data)
data=data[data[:,1]>0]

# 待繪製的資料
inflection=int(3.5*7*24)

xa=data[:inflection,0]
ya=data[:inflection,1]

xb=data[inflection:,0]
yb=data[inflection:,1]

# 找出趨近測試資料的1階線性方程式
fp1, residuals, rank1, sv1, rcond1=sp.polyfit(xa, ya, 10, full=True)
fp2, residuals, rank2, sv2, rcond2=sp.polyfit(xb, yb, 10, full=True)

# 設定字型及大小
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['font.size'] = 14

# 設定圖標題
fig, ax = plt.subplots()
ax.set_title('網路流量')

# 設定x軸及y軸標題
plt.xlabel('日期')
plt.ylabel('每小時用量')

# 線性方程式
f1=sp.poly1d(fp1)
f2=sp.poly1d(fp2)


# 印出誤差
print(f1.order, '階-', '誤差:', sp.sum((f1(xa)-ya)**2))
print(f2.order, '階-', '誤差:', sp.sum((f2(xb)-yb)**2))
print('總合誤差=', sp.sum((f1(xa)-ya)**2) + sp.sum((f2(xb)-yb)**2))

#傳回1000個(從0到x元素個數), 間隔相同的數
fxa=sp.linspace(0, inflection, 600)
fxb=sp.linspace(inflection, 750, 600)

# 將產生的數一個個代入線性方程式, 並畫在圖上
#plt.plot(fx, f1(fx), linewidth=4)
plt.plot(fxa, f1(fxa), linewidth=4)
plt.plot(fxb, f2(fxb), linewidth=4)

# 產生左上角的標籤說明
plt.legend(["d=%i" % f1.order, "d=%i" % f2.order], loc="upper left")


# 將測試資料畫在圖上
plt.scatter(xa, ya)
plt.scatter(xb, yb)

# 畫出格線
plt.grid()

# 顯示圖表
plt.show()
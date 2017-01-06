import numpy as np
import matplotlib.pyplot as plt
import scipy as sp


# 讀入資料
data=np.genfromtxt('web_traffic.tsv', dtype="i", delimiter='\t', usecols=(0,1), unpack=True)

# 整理資料
data=np.transpose(data)
data=data[data[:,1]>0]

# 待繪製的資料
infection=int(3.5*7*24)
x=data[infection:,0]
y=data[infection:,1]

# 找出趨近測試資料的1階線性方程式
fp1, residuals, rank, sv, rcond=sp.polyfit(x, y, 1, full=True)
fp2, residuals, rank2, sv2, rcond2=sp.polyfit(x, y, 10, full=True)
fp3, residuals, rank3, sv3, rcond3=sp.polyfit(x, y, 50, full=True)

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
f3=sp.poly1d(fp3)

# 方程式求x根
print(fp1[-1:]+100000)
print(np.roots(fp1))

# 印出誤差
print(f1.order, '階-', '誤差:', sp.sum((f1(x)-y)**2))
print(f2.order, '階-', '誤差:', sp.sum((f2(x)-y)**2))
print(f3.order, '階-', '誤差:', sp.sum((f3(x)-y)**2))

#傳回1000個(從0到x元素個數), 間隔相同的數
fx=sp.linspace(0, x[-1], 1000)

# 將產生的數一個個代入線性方程式, 並畫在圖上
plt.plot(fx, f1(fx), linewidth=4)
plt.plot(fx, f2(fx), linewidth=4)
plt.plot(fx, f3(fx), linewidth=4)

# 產生左上角的標籤說明
plt.legend(["d=%i" % f1.order, "d=%i" % f2.order, "d=%i" % f3.order], loc="upper left")


# 將測試資料畫在圖上
plt.scatter(x, y)

# 畫出格線
plt.grid()

# 顯示圖表
plt.show()
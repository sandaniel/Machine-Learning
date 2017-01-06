import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import math

# 讀入資料
data=np.genfromtxt('web_traffic.tsv', delimiter='\t', unpack=True)

# 待繪製的資料
x=data[0,:]
y=data[1,:]

# 整理資料
x=x[~sp.isnan(y)]
y=y[~sp.isnan(y)]

# 找出一條趨近測試資料的n階線性方程式
fp2=sp.polyfit(x, y, 2)
fp10=sp.polyfit(x, y, 10)

# 設定字型及大小
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['font.size'] = 14

# 設定圖標題
plt.title('網路流量')

# 設定x軸及y軸標題
plt.xlabel('時間序')
plt.ylabel('每小時用量')

# 線性方程式
f2=sp.poly1d(fp2)
f10=sp.poly1d(fp10)

#傳回1000個(從0到x元素個數), 間隔相同的數
fx=sp.linspace(0, x[-1], 1000)

# 將產生的數一個個代入線性方程式, 並畫在圖上
plt.plot(fx, f2(fx), 'r-', linewidth=4)
plt.plot(fx, f10(fx), 'g-', linewidth=4)

# 計算誤差
print('2階=', math.log(sp.sum(((y-f2(x))**2))))
print('10階=', math.log(sp.sum(((y-f10(x))**2))))

# 產生左上角的標籤說明
plt.legend(['2階', '10階'], loc="upper left")

# 將測試資料畫在圖上
plt.axis([-100, 800, 0, 7500])
plt.plot(x, y, 'y+')

# 畫出格線
plt.grid()

# 顯示圖表
plt.show()

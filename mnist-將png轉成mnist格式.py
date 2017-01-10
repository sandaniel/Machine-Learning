#================================================================================
# (1) 先下載安裝 gzip 程式, 並且在命令提示字元中輸入 gzip 不會出現 bad command.
# (2) pip install pillow
# (3) 下載圖片檔: https://github.com/gskielian/JPG-PNG-to-MNIST-NN-Format
#================================================================================
import os
from PIL import Image
from array import *
from random import shuffle


# 載入圖片的位置
Names = [['./training-images','train'], ['./test-images','test']]

for name in Names:	
	data_image = array('B')
	data_label = array('B')

	FileList = []
	for dirname in os.listdir(name[0])[0:]: 
		path = os.path.join(name[0], dirname)
		for filename in os.listdir(path):
			if filename.endswith(".png"):
				FileList.append(os.path.join(name[0],dirname,filename))

	shuffle(FileList) 

	for filename in FileList:				
		label = int(filename.split('\\')[1])
		print('檔案:', filename, '  標籤:', label)
		
		Im = Image.open(filename)
		pixel = Im.load()
		width, height = Im.size

		#加入圖片內容
		for x in range(0, width):
			for y in range(0, height):
				data_image.append(pixel[y, x])
		
		#加入標籤
		data_label.append(label)

	
	# 共有圖片數
	hexval = "{0:#0{1}x}". format(len(FileList),6) # number of files in HEX

	
	# 檔案表頭
	header = array('B')
	header.extend([0,0,8,1,0,0])
	header.append(int('0x'+hexval[2:][:2],16))
	header.append(int('0x'+hexval[2:][2:],16))
	
	data_label = header + data_label

	# 圖檔尺寸最大為256*256	
	if max([width,height]) <= 256:
		header.extend([0,0,0,width,0,0,0,height])
	else:
		raise ValueError('Image exceeds maximum size: 256x256 pixels');

		
	# Changing MSB for image data (0x00000803)	
	header[3] = 3 
	
	data_image = header + data_image

	# 輸出檔名, 寫內容, 關檔
	output_file = open(name[1]+'-images-idx3-ubyte', 'wb')
	data_image.tofile(output_file)
	output_file.close()

	output_file = open(name[1]+'-labels-idx1-ubyte', 'wb')
	data_label.tofile(output_file)
	output_file.close()

	
# 將檔案壓縮
for name in Names:	
	os.system('gzip '+name[1]+'-images-idx3-ubyte')
	os.system('gzip '+name[1]+'-labels-idx1-ubyte')

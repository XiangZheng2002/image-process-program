import cv2
import numpy as np
from PIL import Image

data = np.loadtxt('无阴影hand.txt')
print(data)
for i in range(data.shape[0]):
	for j in range(data.shape[1]):
		if(data[i][j] == 255):
			data[i][j] = 0
		else:
			data[i][j] = 1
print(data)
print("data's shape")
print(data.shape)
# 计算RGB均值
arr = np.array(data)
count = np.sum(arr == 1)
print(count)
r_sum = 0
g_sum = 0
b_sum = 0
pic = cv2.imread("hand.jpg")
b, g, r = cv2.split(pic)
for i in range(data.shape[0]):
	for j in range(data.shape[1]):
		if(data[i][j] == 1):
			r_sum += r[i][j]
			g_sum += g[i][j]
			b_sum += b[i][j]
r_ave, g_ave, b_ave = r_sum/count, g_sum/count, b_sum/count
print("r_ave")
print(r_ave)
print("g_ave")
print(g_ave)
print("b_ave")
print(b_ave)
# 读取彩色图像
image = Image.open('k_means12.png')
resized_image = image.resize((681, 468))
resized_image.save("k_means13.png")
img = cv2.imread('k_means13.png')
print("img's size")
print(img.shape)
# 将图像拆分为3个颜色通道
b, g, r = cv2.split(img)

print("b's shape")
print(b.shape)
for layer in [b, g, r]:
	for i in range(layer.shape[0]):
		for j in range(layer.shape[1]):
			if(data[i][j] == 0):
				layer[i][j] = 0
			else:
				continue
kmeansRGB = cv2.merge([b, g, r])
cv2.imshow('KMEANS Image', kmeansRGB)
# 定义结构元素
kernel = np.ones((5,5),np.uint8)

"""对每个颜色通道进行腐蚀操作"""
erosion_b = cv2.erode(b, kernel, iterations = 1)
erosion_g = cv2.erode(g, kernel, iterations = 1)
erosion_r = cv2.erode(r, kernel, iterations = 1)
# 合并每个颜色通道
erosion_img = cv2.merge([erosion_b, erosion_g, erosion_r])
cv2.imshow('Original Image', img)
cv2.imshow('Erosion Image', erosion_img)

"""对每个颜色通道进行闭操作"""
closing_b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, kernel)
closing_g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, kernel)
closing_r = cv2.morphologyEx(r, cv2.MORPH_CLOSE, kernel)
closing_img = cv2.merge([closing_b, closing_g, closing_r])
cv2.imshow('closing_img Image', closing_img)

"""对每个颜色通道进行膨胀运算"""
dilation_b = cv2.dilate(closing_b, kernel, iterations=1)
dilation_g = cv2.dilate(closing_g, kernel, iterations=1)
dilation_r = cv2.dilate(closing_r, kernel, iterations=1)
dilation_img = cv2.merge([dilation_b, dilation_g, dilation_r])
cv2.imshow('dilation_img Image', dilation_img)

"""对每个颜色通道进行开操作"""
open_b = cv2.morphologyEx(closing_b, cv2.MORPH_OPEN, kernel)
open_g = cv2.morphologyEx(closing_b, cv2.MORPH_OPEN, kernel)
open_r = cv2.morphologyEx(closing_b, cv2.MORPH_OPEN, kernel)
# 合并每个颜色通道
open_img = cv2.merge([open_b, open_g, open_r])
cv2.imshow('Original Image', img)
cv2.imshow('open_img Image', open_img)

# 显示图像
cv2.waitKey(0)
cv2.destroyAllWindows()


import cv2
import numpy as np
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk


"""---functions---"""
# RGB空间转HSV空间
def rgbtoHSV(rgb):
    r, g, b = rgb[:, :, 0] / 255.0, rgb[:, :, 1] / 255.0, rgb[:, :, 2] / 255.0
    print(rgb.shape)
    print(r)
    c_max = np.zeros((r.shape[0], r.shape[1]))
    c_min = np.zeros((r.shape[0], r.shape[1]))
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            c_max[i][j] = max(r[i][j], g[i][j], b[i][j])
            c_min[i][j] = min(r[i][j], g[i][j], b[i][j])
    print(c_max)
    print(c_min)
    delta = c_max - c_min
    h, s, v = np.zeros((r.shape[0], r.shape[1])), np.zeros((r.shape[0], r.shape[1])), np.zeros((r.shape[0], r.shape[1]))
    for i in range(delta.shape[0]):
        for j in range(delta.shape[1]):
            if delta[i][j] == 0:
                h[i][j] = 0
            elif c_max[i][j] == r[i][j]:
                h[i][j] = 60 * ((g[i][j] - b[i][j]) / delta[i][j])
            elif c_max[i][j] == g[i][j]:
                h[i][j] = 60 * ((b[i][j] - r[i][j]) / delta[i][j] + 2)
            else:
                h[i][j] = 60 * ((r[i][j] - g[i][j]) / delta[i][j] + 4)
            if c_max[i][j] == 0:
                s[i][j] = 0
            else:
                s[i][j] = delta[i][j] / c_max[i][j]
    v = c_max
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            if (h[i][j] < 0):
                h[i][j] += 360
    h = h / 2
    s = s * 255
    v = v * 255
    return h, s, v

# 返回非手部区域矩阵
def detectHand(h, s, v):
    arr = []
    arrPosition = np.ones((h.shape[0], h.shape[1]), dtype=int)
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            if (h[i][j] > 135) or (s[i][j] < 20) or (v[i][j] < 110):
                arr.append((i, j))
                arrPosition[i][j] = 0
    return arr, arrPosition

# 绘制蒙版
def drawMask(arr, path):
    image = Image.open(path)
    for pixel in arr:
        # 纯黑
        image.putpixel((pixel[1], pixel[0]), (0, 0, 0, 128))
    image.show()
    image.save('modified.jpg')


# 图像预处理
def readImage(path):
    # cv读取图像
    img = cv2.imread(path)
    # 灰度化处理图像
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(grayImage, 120, 255, cv2.THRESH_BINARY)
    return ret, binary


# 边缘检测图像分割
# robert算子图像分割
def robert(ret, binary):
    kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
    kernely = np.array([[0, -1], [1, 0]], dtype=int)
    x = cv2.filter2D(binary, cv2.CV_16S, kernelx)
    y = cv2.filter2D(binary, cv2.CV_16S, kernely)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Roberts = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.plot(), plt.imshow(Roberts, 'gray')
    plt.title("ROBERT")
    plt.xticks([]), plt.yticks([])
    plt.show()
    return Roberts

# Prewitt算子图像分割
def prewitt(binary):
    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
    x = cv2.filter2D(binary, cv2.CV_16S, kernelx)
    y = cv2.filter2D(binary, cv2.CV_16S, kernely)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Prewitt = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return Prewitt

# Sobel算子图像分割
def sobel(binary):
    x = cv2.Sobel(binary, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(binary, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return Sobel

# 拉普拉斯算法
def laplace(binary):
    dst = cv2.Laplacian(binary, cv2.CV_16S, ksize = 3)
    Laplacian = cv2.convertScaleAbs(dst)
    return Laplacian

# Scharr算子图像分割
def scharr(binary):
    x = cv2.Scharr(binary, cv2.CV_32F, 1, 0)  # X方向
    y = cv2.Scharr(binary, cv2.CV_32F, 0, 1)  # Y方向
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Scharr = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return Scharr

# Canny算子图像分割
def canny(binary):
    gaussianBlur = cv2.GaussianBlur(binary, (3,3), 0) #高斯滤波
    Canny = cv2.Canny(gaussianBlur, 50, 150)
    return Canny


# LOG算子图像分割
def LOG(binary):
    gaussianBlur = cv2.GaussianBlur(binary, (3, 3), 0)  # 高斯滤波
    dst = cv2.Laplacian(gaussianBlur, cv2.CV_16S, ksize=3)
    LOG = cv2.convertScaleAbs(dst)
    return LOG

# 绘制效果图
def drawEdgeResult(images):
    # 效果图
    titles = ['Source Image', 'Binary Image', 'Roberts Image',
              'Prewitt Image', 'Sobel Image', 'Laplacian Image',
              'Scharr Image', 'Canny Image', 'LOG Image']
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    for i in np.arange(8):
        plt.subplot(3, 3, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()
    print("multiEdgeResult DISPLAY!")
    # plt.savefig("multiEdgeResult.png", dpi=None, bbox_inches='tight')
    print("multiEdgeResult SAVE DOWN!")


def findContour(edges):
    # 查找边缘
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 绘制所有边缘
    cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
    # 输出所有边缘点坐标
    arr = []
    for i in range(len(contours)):
        # print("Contour ", i, " points: ")
        for j in range(len(contours[i])):
            x, y = contours[i][j][0][0], contours[i][j][0][1]
            # print("x:", x, " y:", y)
            point_ = (x, y)
            arr.append(point_)
    return arr


def splitContour(img, contour):
    # 绘制掩码后的图像
    # 创建掩码图像
    contour = np.array(contour)
    print(contour)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray)
    hull = cv2.convexHull(contour)
    cv2.drawContours(mask, [hull], -1, 255, thickness=-1)
    # cv2.drawContours(mask, np.array(contour), -1, 255, thickness=-1)
    result = cv2.bitwise_and(img, img, mask=mask)
    img_cv = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.imshow("Image", img_cv)
    cv2.waitKey(0)
    cv2.imwrite("最优算子分割结果.png", img_cv)
    print("最优算子分割结果.png SAVE down!!!")


# 通过k-means聚类图像分割
def kMeans(path):
    img = cv2.imread(path, 0)
    print(img.shape)
    rows, cols = img.shape[:]
    data = img.reshape((rows * cols, 1))
    data = np.float32(data)
    criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    # 聚集成5类
    compactness, labels, centers = cv2.kmeans(data, 5, None, criteria, 10, flags)
    dst = labels.reshape((img.shape[0], img.shape[1]))
    dst = np.array(dst)
    print(dst)
    dst1 = np.where(dst==0, 1, 0)
    dst2 = np.where(dst==1, 2, 0)
    dst3 = np.where(dst==2, 3, 0)
    dst4 = np.where(dst==3, 4, 0)
    dst5 = np.where(dst==4, 5, 0)
    # dst6 = np.where(dst==5, 6, 0)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 显示图像
    # titles = ['原始图像', '聚类图像', '聚类图像1', '聚类图像2', '聚类图像3', '聚类图像4', '聚类图像5', '聚类图像6']
    # titles = ['原始图像', '聚类图像', '聚类图像1', '聚类图像2', '聚类图像3', '聚类图像4']
    titles = ['原始图像', '聚类图像', '聚类图像1', '聚类图像2', '聚类图像3', '聚类图像4', '聚类图像5']
    # images = [img, dst, dst1, dst2, dst3, dst4, dst5, dst6]
    # images = [img, dst, dst1, dst2, dst3, dst4]
    images = [img, dst, dst1, dst2, dst3, dst4, dst5]
    for i in range(7):
        plt.imshow(images[i])
        plt.savefig('layerPic%d.png' % (i))
    for i in range(7):
        plt.subplot(3, 3, i + 1), plt.imshow(images[i]),
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()
    return dst1, dst2, dst3, dst4, dst5


# k-means 图层保存
def layerProcess(dst1, dst2, dst3, dst4, dst5):
    layers = [dst1, dst2, dst3, dst4, dst5]
    count = 1
    for layer in layers:
        with open("dst%d.txt" % (count), "w") as file:
            layer = np.array(layer)
            np.savetxt(file, layer, fmt='%d')
        count += 1
    print("ALL LAYERS WRITTEN!!!")

# 对部分图层置零操作
def layerMakeZero(matrix1, coordinates):
    for x, y in coordinates:
        newX = int((y-72) * 468/343)
        newY = int((x-80) * 681/496)
        for i in range(-5, 5, 1):
            matrix1[newX+i][newY+i] = 0
    return matrix1

# GUI界面及ACTIONS
class ImageDisplay:

    def __init__(self, image_path):
        self.zeroPixelList = []
        self.zeroPixelList2 = []
        self.zeroPixelList3 = []
        # 创建主窗口
        self.root = tk.Tk()
        # 创建zeroPixelList

        # 这里时候读取pixelZERO.txt 文件，现在感觉不可以加，上一个图层到下一个图层应该清空
        # with open("pixelZERO.txt", "r") as file:
        #     content = file.read()
        # str1 = ""
        # for element in content:
        #     # print(element)
        #     if(element == " "):
        #         x = int(str1)
        #         str1 = ""
        #     elif(element == '\n'):
        #         y = int(str1)
        #         self.zeroPixelList.append((x, y))
        #         str1 = ""
        #     else:
        #         str1 = str1 + element
        # self.zeroPixelList = np.empty([], dtype=tuple)
        # 读取图像并将其显示在画布上
        self.image = Image.open(image_path)
        self.photo = ImageTk.PhotoImage(self.image)
        self.canvas = tk.Canvas(self.root, width=self.image.width, height=self.image.height)
        self.canvas.create_image(0, 0, anchor='nw', image=self.photo)
        self.canvas.pack()
        # 定义四边形框的顶点
        self.vertices = []
        # 绑定鼠标单击事件处理函数
        self.canvas.bind('<Button-1>', self.on_canvas_click)
        # 进入消息循环
        self.root.mainloop()


    def on_canvas_click(self, event):
        # 获取单击事件的坐标
        x = event.x
        y = event.y
        # 记录顶点
        self.vertices.append((x, y))
        # 画顶点
        self.canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill='red')
        # 如果顶点数为4，则绘制四边形并输出像素坐标
        if len(self.vertices) == 4:
            self.canvas.create_polygon(*[coord for vertex in self.vertices for coord in vertex], outline='red')
            print("print(*self.vertices)")
            print(*self.vertices)
            pixel_coords = self.get_pixels_in_quad(*self.vertices)

            # # print(*self.vertices)
            a = [((self.vertices[0][0]), (self.vertices[0][1])),
                 ((self.vertices[1][0]), (self.vertices[1][1])),
                 ((self.vertices[2][0]), (self.vertices[2][1])),
                 ((self.vertices[3][0]), (self.vertices[3][1])),
                 ]
            pixel_coords = self.get_pixels_in_quad(*a)
            print("THIS PROPRCESS HAS PROCESSED!")
            print('Pixels inside the quadrilateral1:', pixel_coords)
            # self.zeroPixelList = self.getZeroPixels(pixel_coords3)
            # with open("pixelZERO.txt", "a") as file:
            #     print(self.zeroPixelList)
            #     for tup in self.zeroPixelList:
            #         strTowrite = ' '.join(str(i) for i in tup)
            #         file.write(strTowrite + '\n')
            #     file.close()
            # self.deleteRepeat()
            # self.zeroPixelList = self.deleteRepeatFromFile()
            self.zeroPixelList = pixel_coords
            print("FINAL self.zeroPixelList!!!")
            print(self.zeroPixelList)
            self.confirmIt()

    def get_pixels_in_quad(self, p1, p2, p3, p4):
        # 获取四边形内的像素坐标
        pixels = []
        # 将四边形顶点按照左上、右上、右下、左下的顺序排列
        sorted_vertices = sorted([p1, p2, p3, p4], key=lambda x: (x[1], x[0]))
        p1, p2, p3, p4 = sorted_vertices[0], sorted_vertices[1], sorted_vertices[3], sorted_vertices[2]
        # 计算四边形的边界框
        min_x, max_x = min(p1[0], p2[0]), max(p3[0], p4[0])
        min_y, max_y = min(p1[1], p4[1]), max(p2[1], p3[1])
        # 遍历边界框内的所有像素点，判断它们是否在四边形内部
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                if self.point_in_quad(x, y, p1, p2, p3, p4):
                    pixels.append((x, y))
        return pixels

    def point_in_quad(self, x, y, p1, p2, p3, p4):
        # 判断点是否在四边形内部
        p = (x, y)
        triangle1 = [p1, p2, p]
        triangle2 = [p2, p3, p]
        triangle3 = [p3, p4, p]
        triangle4 = [p4, p1, p]
        return (self.triangle_area(*triangle1) > 0
                and self.triangle_area(*triangle2) > 0
                and self.triangle_area(*triangle3) > 0
                and self.triangle_area(*triangle4) > 0)

    def triangle_area(self, p1, p2, p3):
        # 计算三角形面积
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])

    def getZeroPixels(self, pixels):
        print("pixels in function getZeroPixels")
        print(pixels)
        for element in pixels:
            # self.zeroPixelList = np.append(self.zeroPixelList, [element])
            self.zeroPixelList.append(element)
            # print(len(self.zeroPixelList))
        # self.zeroPixelList = set(self.zeroPixelList)
        # DUBUG 打印self.zeroPixelList
        # print("self.zeroPixelList in function getZeroPixels")
        # print(self.zeroPixelList)
        return self.zeroPixelList

    def deleteRepeatFromFile(self):
        # with open("pixelZERO.txt", "r") as file:
        #     content = file.read()
        # arr = []
        # str1 = ""
        # for element in content:
        #     # print(element)
        #     if(element == " "):
        #         x = int(str1)
        #         str1 = ""
        #     elif(element == '\n'):
        #         y = int(str1)
        #         arr.append((x, y))
        #         str1 = ""
        #     else:
        #         str1 = str1 + element
        # print("arr in deleteRepeatFromFile!!!")
        # print(arr)
        self.zeroPixelList = self.deleteRepeat(self.zeroPixelList)
        # 这里没有进行重复删除
        # self.zeroPixelList = arr
        return self.zeroPixelList

    def deleteRepeat(self, arr):
        print("arr in deleteRepeat function")
        print(arr)
        unique = set(tuple(x) for x in arr)
        self.zeroPixelList = [tuple(x) for x in unique]
        return self.zeroPixelList

    def returnZeroList(self):
        return self.zeroPixelList

    def confirmIt(self):
        # 创建Canvas组件，并将图片放入其中
        canvas = tk.Canvas(self.root, width=self.image.width, height=self.image.height)
        canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        canvas.pack()
        # 遍历坐标数组，并在画布上绘制每个点
        for x, y in self.zeroPixelList:
            canvas.create_oval(x - 1, y - 1, x + 1, y + 1, fill='red')



# 给每个图层的部分地区取0
def eachLayerZerize(allDst, imagePaths):
    display = []
    zeroList = []
    tempDst = []
    dstPros = []
    for i in range(len(allDst)):
        temp = ImageDisplay(imagePaths[i])
        display.append(temp)
        zeroList.append(display[i].returnZeroList())
        print("点击图像 %s 获取坐标进程已进行!!!" % (imagePaths[i]))
        tempDst.append(np.array(allDst[i]).copy())
        print("shape of zeroList[i]")
        zeroList[i] = np.array(zeroList[i])
        print(zeroList[i].shape)
        dstPros.append(layerMakeZero(tempDst[i], zeroList[i]))
        plt.imshow(dstPros[i], cmap='gray_r')
        plt.show()
    return dstPros

# 将layer中非0元素置1
def zerizedPic(allLayers):
    newPic = np.array(allLayers).copy()
    sumUp = np.sum(newPic, axis=0)
    print("print(sumUp.shape) in zerizedPic()")
    print(sumUp.shape)
    # 将不为0的元素置为1，其他元素保持不变
    sumUp[sumUp != 0] = 1
    print("print(sumUp.shape)2222222222 in zerizedPic()")
    print(sumUp.shape)
    return sumUp

# 使用Image展示去除阴影部分后的图像
def proPhoto(sumUp, path):
    # 打开图像并转换为灰度
    img = Image.open(path).convert('L')
    # 将图像转换为矩阵
    img_arr = np.array(img)
    with open("sumUp.txt", "w") as file:
        sumUp1 = np.array(sumUp)
        np.savetxt(file, sumUp1, fmt='%d')
    for i in range(img_arr.shape[0]):
        for j in range(img_arr.shape[1]):
            if(sumUp[i][j] == 0):
                img_arr[i][j] = 255
    # 将矩阵转换为图像对象
    img2 = Image.fromarray(img_arr, mode='L')
    # # 显示图像
    # with open("无阴影hand.txt", "w") as file:
    #     img_arr = np.array(img_arr)
    #     np.savetxt(file, img_arr, fmt='%d')
    img2.show()


"""---main---"""
# 打开原始图片
image = Image.open("hand.jpg")
# 获取原始图片尺寸
width, height = image.size
new_width = 496
new_height = 343
resized_image = image.resize((new_width, new_height))
# 保存缩小后的图片
resized_image.save("resized_hand.jpg")
img = cv2.imread('hand.jpg')
# check if the image is readable:
if img is None:
    raise Exception("读入图片失败，请检查图片是否存在")
h, s, v = rgbtoHSV(img)
# 将HSV保存到txt文件中
hFile = open("H计算结果.txt", "w")
data_str = str(h)
hFile.write(data_str)
hFile.close()
sFile = open("S计算结果.txt", "w")
data_str = str(s)
sFile.write(data_str)
sFile.close()
vFile = open("V计算结果.txt", "w")
data_str = str(v)
vFile.write(data_str)
vFile.close()
# 初步返回不是手部的arr
arr, arrPosition = detectHand(h, s, v)
# 将arr转换成file储存
file = open("非手部矩阵.txt", "w")
data_str = str(arr)
file.write(data_str)
file.close()
# 给手部区域draw mask
drawMask(arr, 'hand.jpg')
img = cv2.imread('modified.jpg')
# calculate ret & binary
ret, binary = readImage('modified.jpg')
# 调用算子函数
Roberts = robert(ret, binary)
Prewitt = prewitt(binary)
Sobel = sobel(binary)
Laplacian = laplace(binary)
Scharr = scharr(binary)
Canny = canny(binary)
LOG = LOG(binary)
images = [img, Roberts, Prewitt, Sobel, Laplacian, Scharr, Canny, LOG]
# 绘制效果图
drawEdgeResult(images)
# 使用canny获取边缘坐标
contour = findContour(Canny)
# print(contour)
splitContour(img, contour)
dst1, dst2, dst3, dst4, dst5 = kMeans("hand.jpg")
layerProcess(dst1, dst2, dst3, dst4, dst5)
allDst = [dst2, dst3, dst4, dst5]
# 点击图像获取坐标'
image_PATHS = ['layerPic3.png', 'layerPic4.png', 'layerPic5.png', 'layerPic6.png']
allLayers = eachLayerZerize(allDst, image_PATHS)
print("所有图层已成功加工完成!!!")
zerizedResult = np.array(zerizedPic(allLayers))
print(zerizedResult.shape)
img = Image.open('hand.jpg').convert('L')
# 将图像转换为矩阵
img_arr = np.array(img)
# 将矩阵转换为图像对象
img2 = Image.fromarray(img_arr, mode='L')
# 显示图像
img2.show()
plt.savefig('hand2.png')
plt.imshow(zerizedResult, cmap='gray_r')
plt.show()
proPhoto(zerizedResult, 'hand.jpg')
img = Image.open('hand.jpg').convert('L')
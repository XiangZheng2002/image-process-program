import matplotlib.pyplot as plt
import numpy as np

# 定义0和1的矩阵
matrix = np.array([[0, 1, 1], [1, 0, 0], [1, 1, 1]])

# 绘制图像
plt.imshow(matrix, cmap='gray')
plt.show()
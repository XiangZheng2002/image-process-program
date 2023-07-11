import numpy as np

# 创建一个空的 NumPy 数组
arr = np.array([])

# 要添加的元组列表
tup_list = [(1, 2), (3, 4), (5, 6)]

# 使用 numpy.append() 函数将元组列表添加到数组中
arr = np.append(arr, [tup_list])

print(arr) # 输出 [(1, 2) (3, 4) (5, 6)]
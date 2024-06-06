import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# 设置全局字体样式和大小
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman']
matplotlib.rcParams['font.size'] = 15


def inverse_sigmoid(xi):
    """计算反 Sigmoid 函数"""
    return np.log(xi / (1 - xi))


# 定义 y 值范围   注意避免 y=0和y=1，因为 ln(0) 和 ln(1) 是无限的
r = 0.5
x = np.linspace(-r, 1 - r, 1000)

w_min = 0.5
w_max = 0.9

# 计算反 Sigmoid 函数的值
y = (inverse_sigmoid(0.96 * x + 0.5)) / 7.8
w         = (w_min + w_max)/2 + (w_max - w_min)*y
w_reverse = (w_min + w_max)/2 - (w_max - w_min)*y
# w_reverse = np.min(w) + np.max(w) - w


# 绘制反 Sigmoid 函数的曲线
plt.figure(figsize=(8, 6))
plt.plot(x, w,         label='Inertia Weight', color='b')
plt.plot(x, w_reverse, label='Reverse Weight', color='r')
# plt.title(f'Inverse Sigmoid Function \n min={np.min(y)}  max={np.max(y)}')
plt.ylabel('Weight value')
plt.xlabel('x')
# plt.grid(True)
# 显示更细的网格
plt.grid(True, which='both', linestyle='--', linewidth=0.8)  # 'both' 表示同时显示主要和次要网格线

plt.legend()
plt.show()



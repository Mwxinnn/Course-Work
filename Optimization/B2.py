"""
推导用于求解此问题的随机梯度下降 (SGD) 算法, 将动量和 Nesterov 加速分别应用于推导出的 SGD 算法, 编写这三种方法对应的程序.
使用 N = 100, 初值 x = (0, · · · , 0) 和相同的线搜索准则与迭代终止条件, 比较几种方法的性能, 例如同样精度目标所需的迭代步数的均值与方差, 或者相同步数下的精度差异.
"""

import numpy as np
from scipy.optimize._linesearch import line_search_wolfe1

MaxIterNum = 100
StartPoint = np.zeros(MaxIterNum)


# 定义目标函数
def func(x):
    ret = 0
    for i in range(MaxIterNum):
        e_i = np.zeros(MaxIterNum)
        e_i[i] = i + 1
        vec = x - e_i
        ret += np.sum(vec * vec)
    return 1 / MaxIterNum * ret


# 定义目标函数的梯度算子
def gradient(x):
    return 2 * x - 2 / MaxIterNum * b


# 经分析，最优点的位置(1, 2, ... ,100)
b = np.arange(1, MaxIterNum + 1)
opt_point = np.arange(1, MaxIterNum + 1) / MaxIterNum


def sgd(x_o, func, gradient):
    counter = 0
    flag = False
    while not flag:
        current_gradient = gradient(x_o)
        alpha_k = line_search_wolfe1(f=func, fprime=gradient, xk=x_o, pk=-current_gradient)[0]
        x_n = x_o - alpha_k * current_gradient
        counter += 1
        if abs(func(x_n) - func(x_o)) < 1e-5:
            flag = True
        x_o = x_n
    return x_o, counter


def msgd(x_o, func, gradient, gamma=0.9):
    counter = 0
    p_x_o = x_o

    flag = False
    while not flag:
        current_gradient = gradient(x_o)
        alpha_k = line_search_wolfe1(f=func, fprime=gradient, xk=x_o, pk=-current_gradient)[0]
        x_n = x_o + gamma * (x_o - p_x_o) - alpha_k * current_gradient
        counter += 1
        if abs(func(x_n) - func(x_o)) < 1e-5:
            flag = True
        p_x_o = x_o
        x_o = x_n
    return x_o, counter


def nmsgd(x_o, func, gradient, gamma=0.9):
    counter = 0
    p_x_o = x_o

    flag = False
    while not flag:
        current_gradient = gradient(x_o)
        alpha_k = line_search_wolfe1(f=func, fprime=gradient, xk=x_o, pk=-current_gradient)[0]
        prev_gradient = gradient(x_o + gamma * (x_o - p_x_o))
        x_n = x_o + gamma * (x_o - p_x_o) - alpha_k * prev_gradient
        counter += 1
        if abs(func(x_n) - func(x_o)) < 1e-5:
            flag = True
        p_x_o = x_o
        x_o = x_n

    return x_o, counter


if __name__ == '__main__':
    print('最优函数值%.08f，最优点为' % (func(opt_point)), opt_point)
    final_result, counter = sgd(StartPoint, func, gradient)
    print('sgd方法共迭代%d次，最终函数值%.08f，距离最优点的距离%.08f' % (
        counter, func(final_result), np.linalg.norm(final_result - opt_point, ord=2)))
    final_result, counter = msgd(StartPoint, func, gradient, gamma=0.9)
    print('msgd方法共迭代%d次，最终函数值%.08f，距离最优点的距离%.08f' % (
        counter, func(final_result), np.linalg.norm(final_result - opt_point, ord=2)))
    final_result, counter = nmsgd(StartPoint, func, gradient, gamma=0.9)
    print('nmsgd方法共迭代%d次，最终函数值%.08f，距离最优点的距离%.08f' % (
        counter, func(final_result), np.linalg.norm(final_result - opt_point, ord=2)))

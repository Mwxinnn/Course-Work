"""
问题描述：
(A1) 平面上的三点 A1, A2, A3 的坐标分别为 (0, 0), (0, 3), (4, 0). 选取一种优化算法, 找
出平面上与这三点的距离之和最小的点的坐标.
解题思路：定义距离为欧式距离平方和，使用最速下降法，并通过wolfe准则搜索步长实现。
"""

A1 = (0, 0)
A2 = (0, 3)
A3 = (4, 0)
# 设置迭代初始点
x0 = (0, 1)
alpha0 = 1

# 计算x,y两点之间的距离的函数
DistanceCalculator = lambda x, y: (x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2
# 计算某点x到A1,A2,A3三个点的距离平方和的函数
TotalDistanceCalculator = lambda x: DistanceCalculator(A1, x) + DistanceCalculator(A2, x) + DistanceCalculator(A3, x)
# 计算以TotalDistanceCalculator为目标函数时，点x的梯度方向的函数
GradientCalculator = lambda x: (6 * x[0] - 8, 6 * x[1] - 6)

# 迭代x的函数
updator = lambda x, a: (x[0] - a * GradientCalculator(x)[0], x[1] - a * GradientCalculator(x)[1])

matmal = lambda x, y: x[0] * y[0] + x[1] * y[1]  # tuple的向量内积


# 更新alpha的Wolfe准则
def WolfeAlpha(x, alpha, c1, c2):
    assert gradient_current_iter
    # 计算验证wolfe准则所需要的量
    x_next_iter = updator(x, alpha)
    distance_next_iter = TotalDistanceCalculator(x_next_iter)
    gradient_next_iter = GradientCalculator(x_next_iter)
    # 验证是否满足wolfe准则
    condition1 = distance_next_iter <= distance_current_iter - c1 * alpha * matmal(gradient_current_iter,
                                                                                   gradient_current_iter)
    condition2 = matmal(gradient_current_iter, gradient_next_iter) <= c2 * matmal(gradient_current_iter,
                                                                                  gradient_current_iter)
    flag = condition1 and condition2
    while not flag:  # 不满足则更新alpha
        while not condition1 and not flag:
            alpha *= 0.9
            flag, alpha = WolfeAlpha(x, alpha, c1, c2)
        while not condition2 and not flag:
            alpha *= 1.1
            flag, alpha = WolfeAlpha(x, alpha, c1, c2)
    return flag, alpha

if __name__ == '__main__':
    x_o = x0
    minDistance = TotalDistanceCalculator(x0)
    counter = 0
    iters = 10000
    while iters - counter > 0:  # 最多迭代iters次
        distance_current_iter = TotalDistanceCalculator(x_o)
        gradient_current_iter = GradientCalculator(x_o)
        _, alpha = WolfeAlpha(x_o, alpha0, c1=0.2, c2=0.7)

        x_n = updator(x_o, alpha)
        currentDistance = TotalDistanceCalculator(x_n)
        if currentDistance <= minDistance:  # 当更新距离小于1e-7时，提前结束迭代
            if minDistance - currentDistance <= 1e-7:
                break
            x_o = x_n
            minDistance = currentDistance
        counter += 1

    print("total iter:", counter)
    print("final result:", x_n)
    print("current distance:", minDistance)

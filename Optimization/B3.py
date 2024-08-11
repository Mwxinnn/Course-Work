"""
尝试应用内点法求解线性规划问题:
min x1+2*x2+3*x3-6*x4
s.t.x1 - x2 - 1 <= 0 and x2 - x1 - 1 <= 0
    x2 - x3 - 1 <= 0 and x3 - x2 - 1 <= 0
    x3 - x4 - 1 <= 0 and x4 - x3 - 1 <= 0
"""
import numpy as np
from scipy.optimize import linprog

# 定义问题需要的变量
x1 = 2
x2 = 2
x3 = 3
x4 = 4  # 原问题变量
y1 = y2 = y3 = y4 = y5 = y6 = 10  # 对偶变量
x = np.array([x1, x2, x3, x4])[:, np.newaxis]
y = np.array([y1, y2, y3, y4, y5, y6])[:, np.newaxis]

A = np.array([[1, -1, 0, 0],
              [-1, 1, 0, 0],
              [0, 1, -1, 0],
              [0, -1, 1, 0],
              [0, 0, 1, -1],
              [0, 0, -1, 1]])

b_ = np.array([1, 1, 1, 1, 1, 1])  # 约束条件Ax+b<=0
b = np.array([-1, -1, -1, -1, -1, -1])[:, np.newaxis]  # 约束条件Ax+b<=0

c = np.array([1, 2, 3, -6])[:, np.newaxis]

# primal: c@x; constrains:A@X+b<=0
# dual: b@y; constrains: A^T@y=c,y>=0
# 互补松弛条件:y^T@(AX+b)=0

# 调包实现：
result = linprog(c, A_ub=A, b_ub=b_, method='interior-point')
print("Optimal solution (x):", result.x)
print("Optimal function:", c.T @ result.x)

# 尝试自己实现：我在实现过程中遇到了问题，花了很多时间也没能得到解决。我可能在迭代的理解上有一定的偏差，希望能得到老师的帮助！
# 思路：利用互补松弛条件，希望最小化对偶间隙。通过Ax+b<=0得到对偶问题min(b^Ty),s.t.A^Ty=-c,y>=0。
# 计算对偶间隙
mu = 1 / 6 * y.T @ (A @ x + b)


# 更新方向的计算：解线性方程组A^T(y+dy)=-c;(y+dy)^T(A(x+dx)+b)=sigma*mu
# 问题：写成矩阵形式后系数矩阵不是方阵，不可逆，我选择了用伪逆的方法解决，这是我认为可能造成问题的部分。
def update_inner_point(x, y, mu, sigma=0.7):
    cdx = np.hstack((y.T @ A, (A @ x + b).T))
    cdy = np.hstack((np.zeros((A.T.shape[0], cdx.shape[1] - A.T.shape[1])), A.T))
    iter_matrix = np.vstack((cdx, cdy))
    res1 = sigma * mu - y.T @ A @ x - y.T @ b
    res2 = -c - A.T @ y
    res = np.vstack((res1, res2))
    delta = np.linalg.pinv(iter_matrix) @ res  # 求伪逆
    return delta[:4, :], delta[4:10, :]


# 选择合适的步长：条件为y+alpha*dy>=0
# 问题：迭代一轮之后，alpha=0
def current_alpha(y, dy):
    y_n = y + dy
    idx, idy = np.where(y_n == y_n.min())
    if dy[idx, idy] < 0:
        alpha = min(-y[idx, idy] / dy[idx, idy], 1)
    else:
        alpha = max(1e-3, -y[idx, idy] / dy[idx, idy])
    return alpha


iters = 10000
counter = 0
while iters - counter > 0:
    # 设置迭代停止的条件：对偶间隙小+对偶问题可行性
    if abs(mu) < 1e-7 and np.linalg.norm(c - A.T @ y, ord=2) < 1e-7 and y.all() > 0:
        break
    dx, dy = update_inner_point(x, y, mu)
    alpha = current_alpha(y, dy)
    x = x + alpha * dx
    y = y + alpha * dy
    mu = 1 / 6 * y.T @ (A @ x + b)
    counter += 1

print("lagrange multiplier:", y)
print("total iter:", counter)
print("final result is:", x)
print("constrain:", A @ x + b)
print("final value is:", c.T @ x)

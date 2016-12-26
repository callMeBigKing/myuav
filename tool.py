from main import *
import numpy as np
from functools import reduce
# 一个个角度相减得到最后的弧度教

def get_type2_promap(core,R):
    # 再已知圆点和半径的情况下求圆，再每个小格子中的概率分布
    pro_map = np.zeros((Nx, Ny), dtype=float)
    points = []  # 里面添加point
    ceil_num = np.ceil(R / Lx - 0.5)  # 半径跨越的网格个数 向上取整

    sigmas = []
    point = [core[0], core[1] + ceil_num]
    type = 1
    while True:
        # 判断是于横轴相交还是纵轴相交
        # 右边一个格子，右下角距离圆心的距离若大于半径则是与横轴相交，否则与纵轴相交
        points.append(point)

        # 通过右下角的点与圆点的距离从而判断新添加进来的点，是于纵轴相交还是与横轴相交
        length = ((point[0] - core[0] + 0.5) * Lx) ** 2 + ((point[1] - core[1] - 0.5) * Ly) ** 2
        if length < R ** 2:  # 与纵轴相交
            type = 1
        else:  # 与横轴相交
            type = 0

        if (type == 1):  # 与纵轴相交，用到的是对边
            # 添加角度
            sigma = np.arcsin(((point[0] - core[0] + 0.5) * Lx) / R)
            # 下一个点是右移一格
            point = [point[0] + 1, point[1]]
        else:
            sigma = np.arccos(((point[1] - core[1] - 0.5) * Lx) / R)
            # 下一个点是下移动一格
            point = [point[0], point[1] - 1]
        # 停止条件，如果当前点和圆点在同一高度，那么break出来
        sigmas.append(sigma)

        if (point[1] == core[1]):
            # sigmas.append(np.arccos(0.5*Ly/R))
            points.append(point)
            break;

    temp1 = np.append(sigmas, np.pi / 2)
    temp2 = np.insert(sigmas, 0, 0)
    sigmas = temp1 - temp2

    num = len(points)
    for i, point, sigma in zip(range(num), points, sigmas):
        if (i == 0):
            sigma = 2 * sigma
            pro_map[point[0], point[1]] = sigma
            pro_map[point[0], 2 * core[1] - point[1]] = sigma
        elif (i == num - 1):
            sigma = 2 * sigma
            pro_map[point[0], point[1]] = sigma
            pro_map[2 * core[0] - point[0], point[1]] = sigma
        else:
            pro_map[point[0], point[1]] = sigma
            pro_map[point[0], 2 * core[1] - point[1]] = sigma
            pro_map[2 * core[0] - point[0], point[1]] = sigma
            pro_map[2 * core[0] - point[0], 2 * core[1] - point[1]] = sigma
    return pro_map


def get_neighbor(site):
    # 返回摩尔邻居list
    neighbor_list = []  # 里面放tuple
    x = site[0]
    y = site[1]
    for neighbor_x in range(3):
        for neighbor_y in range(3):
            g_x = x + neighbor_x - 1
            g_y = y + neighbor_y - 1
            if (0 <= g_x < Nx and 0 <= g_y < Ny):
                neighbor_list.append((g_x, g_y))
    return neighbor_list
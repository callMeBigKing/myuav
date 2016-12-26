from scipy import integrate
import numpy as np
from queue import Queue
from copy import deepcopy
search_step=4
Nx=10
Ny=10

value_map=np.array([ list(map(lambda j:i+j,range(Nx))) for i in range(Nx) ])

def decision(site):
    # 决策函数，根据性能指标计算，返回可行路径中的一点
    # 暂时用前序遍历算法 处理一下
    path = []
    path.append(site)
    best_path = []  # 用来存储最佳路径
    best_value = 0  # 用来存储最佳value

    # 先满上
    while len(path) < search_step:
        top = path[-1]
        neighbor_list = get_neighbor(top)
        path.append(neighbor_list[0])

    while len(path) > 0:
        # 判断是否能够替换成最佳路径
        value = 0
        norepeat_num = len(set(path))  # 用set 的长度来判断path中不重复的元素个数
        if norepeat_num == search_step:
            value = get_path_value(path)
        if value > best_value:
            best_value=value
            best_path = deepcopy(path)
        # 重新添加一条路径，包括两部分，删除，和添加
        throw = path.pop()
        while len(path) > 0:
            top = path[-1]
            top_neighbor_list = get_neighbor(top)
            index = top_neighbor_list.index(throw)
            if index < len(top_neighbor_list) - 1:  # 不是最后一个
                path.append(top_neighbor_list[index + 1])
                while len(path) < search_step:
                    top = path[-1]
                    neighbor_list = get_neighbor(top)
                    path.append(neighbor_list[0])
                break  # 满了search_step就可以break 出来
            else:
                throw = path.pop()  # 再出一个
    return best_path


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

def get_path_value(path):
    value=0
    for site in path:
        value+=value_map[site[0],site[1]]
    return value

if __name__ == '__main__':
    position=(3,2)
    path=decision(position)
    print(path)
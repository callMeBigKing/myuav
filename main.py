import numpy as np

from scipy.integrate import dblquad


'''
 无人机类，
    属性
        当前位置信息
        通信距离
        三幅搜索图
            1.	目标概率图 （一个目标一个概率分布图）
            2.	信息素图
            3.	环境确定度
        访问图和访问结果图
    方法
        决策方法
        更新搜索图方法
'''

##目标信息
target_msg = np.array([
    [1, 2, 10, 0],
    [12, 24, np.nan, np.nan],
    [3, 5, 20, np.nan],
])  # 目标信息，分别是x,y （物理坐标）,v，phi(方位角)# 。
sigma_0 = 3  # δ_0 初始位置的方差
sigma_e = 3  # 布朗运动的方差
Nt = target_msg.shape[0]  # 目标数量
target_type = np.zeros(Nt)  # 存放目标类型
target_true_msg = np.zeros((Nt, 2))  # 目标真实位置，存储所在栅格坐标
target_path = [[] for i in range(Nt)]  # 目标运动路径，生成过程在init_target_position and target_type 中
##战场环境离散
Ny = 30  # y轴上格子的个数
Nx = 30  # x轴上格子的个数
Lx, Ly = 10, 10  # 每个格子的长和宽
connect_map = np.zeros((Nx, Ny))  # 连通图，设置为全局变量
t0 = 2  # 无人机机进入战场时间
delta_t = 2  # 探索间隔时间

# 无人机共同性能信息
Nv = 10  # 无人机架数
PD = 0.95  # 探测概率
PF = 0.02  # 虚警概率

# 搜索的步数
search_step = 5  # 这里的5包括当前点
E = np.ones((Nx, Ny))  # 单位矩阵用于计算

### 信息素的基本信息

init_phero = 2  # 初始信息素的值
plan_phero = 2  # 路径规划的信息素值
# 吸引信息素信息
G_a = 0.5  # 信息素传播因子
E_a = 0.5  # 信息素挥发因子
D_a = 1  # 信息素的释放量
# 挥发信息素信息
G_r = 0.5  # 信息素传播因子
E_r = 0.5  # 信息素挥发因子
D_r = 10  # 信息素的释放量

switch_time=5 # 信息素开关因子，间隔时间大于switch_time才能够释放信息素

# 决策的权值信息
w1 = 0.5  # 概率的权重
w2 = 0.5  # 信息素的权重


class UAV:
    # 无人机类
    def __init__(self, broadcastLen=20, ):
        # 这里的内存分配只是纯粹的未来方便记住数据结构
        self.site = np.ones((1, 2))
        # 位置用二元数来表示
        self.path = [] #
        # path记录无人机飞行路径
        self.broadcastLen = broadcastLen
        # 无人机的广播距离
        self.pro_maps = []
        # 概率图 设置数据结构为三维数组 每个目标的概率分布图

        self.enviro_map = np.zeros((Nx, Ny))
        # 环境确定度
        self.visit_map = np.zeros((Nx, Ny))

        # 访问(探测)结果图
        self.detect_flag = np.ones((1, Nt))
        # 探测标志位，需要搜索用1来表示，不需要搜索则用0来表示 初始化为1

        # 访问信息图 数据结构同概率分布图
        self.detect_maps = np.zeros((Nt, Nx, Ny))

        # 吸引信息素和抑制信息素图
        self.phero_att_map = np.zeros((Nx, Ny))
        self.phero_rej_map = np.zeros((Nx, Ny))

        self.last_vist_map = np.zeros((Nx, Ny))  # 上一次访问时间，根据访问矩阵来更新

        self.phero_switch = np.zeros((Nx, Ny))  # 信息素开关矩阵

        self.phero_plan_map = np.zeros((Nx, Ny))  # 路径规划信息素矩阵

        # path 中的目标概率图预测，四个维度，下标1 表示path步数，下标2 表示target_i 后面是map
        self.path_predict_map = np.zeros((search_step, Nt, Nx, Ny))

        self.plan_path=[] # 计划行走路径
    def add_site(self,site):
        self.site=site
        self.path.append(site)

    def decision(self):
        # 决策函数，根据性能指标计算，返回可行路径中的一点
        # 暂时用前序遍历算法 处理一下
        path = []
        path.append(self.site)
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
                value = self.get_path_value(path)
            if value > best_value:
                best_value = value
                best_path = path.copy()
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
        self.site = best_path[1]  # 更新当前位置
        self.path=best_path
        return best_path

    def cal_path_predict_map(self):
        # 多次调用函数 self.predic_target() 预测目标位置
        for i in range(search_step):
            self.path_predict_map[i] = self.pro_maps
            if i != search_step - 1:
                self.predic_target()
        # 连续predit后，恢复原状
        self.pro_maps = self.path_predict_map[0]

    def get_path_value(self, path):
        # 返回path的花费，path 为list中存tuple [(1,2),(2,3)]
        value = 0
        for i, site in zip(range(search_step), path):
            if i == 0:
                continue
            pro_value = 0
            for j in range(Nt):
                if self.detect_flag == 0:  # 不需要继续搜索的目标直接跳过
                    continue
                pro = self.path_predict_map[i, j, site[0], site[1]]
                pro_value += np.log(1 / (1 - pro))
            # pro_value 需要进行一个类似归一化的处理过程
            pro_value = pro_value * Nt / np.sum(self.detect_flag)
            phero_value = self.phero_att_map[site[0], site[1]] - self.phero_rej_map[site[0], site[1]] - \
                          self.phero_plan_map[site[0], site[1]]
            value = value + w1 * pro_value + w2 * phero_value
        return value

    def update_phero(self):
        # 根据访问信息更新信息素
        GP_a = self.get_G_alpha_map()
        GP_r = self.get_G_r_map()
        self.phero_att_map = (E - self.visit_map) * (1 - E_a) * \
                             ((1 - G_a) * (self.phero_att_map + D_a * self.phero_switch) + GP_a)
        self.phero_rej_map = (1 - E_r) * \
                             ((1 - G_r) * (self.phero_rej_map + D_r * self.visit_map) + GP_r)

    def get_G_alpha_map(self):
        # 计算吸引信息素扩散矩阵
        tmep_map = G_a * (self.phero_att_map + self.phero_switch * D_a)
        G_alpha_map = np.zeros((Nx, Ny))
        for i in range(Nx):
            for j in range(Ny):
                neighbor_list = get_neighbor((i, j))
                for site in neighbor_list:
                    G_alpha_map[i, j] += tmep_map[site[0], site[1]]
                G_alpha_map[i, j] /= len(neighbor_list)
        return G_alpha_map

    def get_G_r_map(self):
        # 计算吸引信息素扩散矩阵
        tmep_map = G_r * (self.phero_rej_map + self.visit_map * D_r)
        G_r_map = np.zeros((Nx, Ny))
        for i in range(Nx):
            for j in range(Ny):
                neighbor_list = get_neighbor((i, j))
                for site in neighbor_list:
                    G_r_map[i, j] += tmep_map[site[0], site[1]]
                G_r_map[i, j] /= len(neighbor_list)
        return G_r_map

    def update_search(self):
        # 搜索图探测更新
        # 方法根据文章中的矩阵表达式来更新

        # 先分子
        H = self.visit_map * PD + (E - self.visit_map) * PF
        for pro_map, detect_map, i in zip(self.pro_maps, self.detect_maps, range(Nt)):
            if self.detect_flag[i] == 0:  # 等于0的部分不需要更新
                continue
            numerator = (H * (2 * detect_map - E) + (E - detect_map)) * pro_map
            pdn = E * PF + (PD - PF) * pro_map
            denominator = pdn * (2 * detect_map - E) + (E - detect_map)
            self.pro_maps[i] = numerator / denominator
            # 注意这里直接改变pro_map 没用这样写不改变 pro_maps，np.array()类型传回的不是引用

    def predic_target(self):
        # 根据探测结果更新搜素图
        # 预测目标在delt_t 后的概率分布类似于初始化的过程
        for target_i in range(Nt):
            if self.detect_flag[target_i] == 0:  # 已经探测到的目标 不需要继续更新将detect_flag置0
                continue

            v = target_msg[target_i, 2]  # v 速度
            phi = target_msg[target_i, 3]  # phi 角度
            now_pro_map_ij = self.pro_maps[target_i]  # 目前的概率
            predict_pro_map = np.zeros((Nx, Ny))
            if target_type[target_i] == 0 or target_type[target_i] == 1:
                # 不知道速度布朗运动,一种正态分布
                sigma = delta_t * sigma_e
                for condition_i in range(Nx):  # 条件概率 条件x_i
                    for condition_j in range(Ny):  # 条件概率 条件y_i
                        core = [condition_i, condition_j]
                        condition_pro_map_ij = get_Gauss_map(core, sigma)  # 条件概率图
                        predict_pro_map += now_pro_map_ij[condition_i, condition_j] * condition_pro_map_ij  # 全概率公式
            elif target_type[target_i] == 2:
                # 类似于初始化的时候
                R = v * delta_t
                for condition_i in range(Nx):  # 条件概率 条件x_i
                    for condition_j in range(Ny):  # 条件概率 条件y_i
                        core = [condition_i, condition_j]
                        condition_pro_map_ij = get_type2_promap(core, R)  # 条件概率图
                        predict_pro_map = now_pro_map_ij[condition_i, condition_j] * condition_pro_map_ij
            elif target_type[target_i] == 3:
                # 从一个map 中概率从一个点转移到另外一个点
                for condition_i in range(Nx):  # 条件概率 条件x_i
                    for condition_j in range(Ny):  # 条件概率 条件y_i
                        u_x = condition_i * Lx + Lx / 2
                        u_y = condition_j * Ly + Ly / 2
                        new_u_x = u_x + v * delta_t * np.cos(phi)
                        new_u_y = u_y + v * delta_t * np.sin(phi)
                        new_u_x, new_u_y = target_move_border_process([u_x, u_y], [new_u_x, new_u_y], phi) # 出界处理，函数里面有判断是否出界
                        site_x=np.floor(new_u_x / Lx)
                        site_y=np.floor(new_u_y / Ly)
                        predict_pro_map[site_x,site_y]+=now_pro_map_ij[condition_i,condition_j]
            self.pro_maps[target_i] = predict_pro_map

    def update_switch_map(self,iter):
        # 更新信息素开关矩阵
        # iter 为当前迭代次数
        # 取最大的数，没找到函数自己写
        up_map=self.visit_map*iter
        self.last_vist_map=(self.last_vist_map>up_map).astype(float)*self.last_vist_map+(self.last_vist_map<=up_map).astype(float)*up_map

        # 更新switch_map
        self.phero_switch=(np.ones((Nx,Ny))*(iter-switch_time)-self.last_vist_map).astype(float)



def get_type2_promap(core, R):
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
            if (0 <= point[0] < Nx and 0 <= point[1] < Ny):  # 每次添加都要做边界判断
                pro_map[point[0], point[1]] = sigma
            if (0 <= point[0] < Nx and 0 <= 2 * core[1] - point[1] < Ny):
                pro_map[point[0], 2 * core[1] - point[1]] = sigma
        elif (i == num - 1):
            sigma = 2 * sigma
            if (0 <= point[0] < Nx and 0 <= point[1] < Ny):  # 每次添加都要做边界判断
                pro_map[point[0], point[1]] = sigma
            if (0 <= 2 * core[0] - point[0] < Nx and 0 <= point[1] < Ny):
                pro_map[2 * core[0] - point[0], point[1]] = sigma
        else:
            if (0 <= point[0] < Nx and 0 <= point[1] < Ny):  # 每次添加都要做边界判断
                pro_map[point[0], point[1]] = sigma
            if (0 <= point[0] < Nx and 0 <= 2 * core[1] - point[1] < Ny):  # 每次添加都要做边界判断
                pro_map[point[0], 2 * core[1] - point[1]] = sigma
            if (0 <= 2 * core[0] - point[0] < Nx and 0 <= point[1] < Ny):  # 每次添加都要做边界判断
                pro_map[2 * core[0] - point[0], point[1]] = sigma
            if (0 <= 2 * core[0] - point[0] < Nx and 0 <= 2 * core[1] - point[1] < Ny):  # 每次添加都要做边界判断
                pro_map[2 * core[0] - point[0], 2 * core[1] - point[1]] = sigma
    pro_map = pro_map / np.sum(pro_map)  # 归一化
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


def get_Gauss_map(core, sigma):
    # 返回gauss
    pro_map = np.zeros((Nx, Ny))
    mu_x = core[0]  # mean_x
    mu_y = core[1]  # mean_y
    for i in range(Nx):
        x_i = i * Lx + Lx / 2  # 第i列格子的x轴坐标
        low_bound_x = x_i - Lx / 2
        hig_bound_x = x_i + Lx / 2
        for j in range(Ny):
            y_j = j * Ly + Ly / 2  # 第j行格子的y轴坐标
            low_bound_y = y_j - Ly / 2
            hig_bound_y = y_j + Ly / 2
            pro_map[i, j],err = dblquad(Gaussian, low_bound_x, hig_bound_x, lambda x: low_bound_y,
                                    lambda x: hig_bound_y, args=(mu_x, mu_y, sigma))
            # 上面会返回误差
    # 归一化
    pro_map = pro_map / np.sum(pro_map)
    return pro_map


# 初始化目标类型
def init_target_type():
    for i in range(Nt):
        if np.isnan(target_msg[i][0]) or np.isnan(target_msg[i][1]):
            target_type[i] = 0  # 均匀分布
        elif np.isnan(target_msg[i][2]):
            target_type[i] = 1  # 初始位置已知的布朗运动
        elif np.isnan(target_msg[i][3]):
            target_type[i] = 2  # 全概率公式积分
        else:
            target_type[i] = 3  # 高斯分布


def init_target_position():
    #     初始化目标的位置，type 0 随机生成，其他的高斯分布生成，越界了就再来一次
    for i in range(Nt):
        if (target_type[i] == 0):
            # 均匀分布随机
            point_x = np.random.randint(0, Nx);  # 左闭右开区间
            point_y = np.random.randint(0, Ny);  # 左闭右开区间
            target_true_msg[i] = [point_x, point_y]
        else:
            u_x = target_msg[i][0] * Lx + Lx / 2
            u_y = target_msg[i][1] * Ly + Ly / 2
            mean = [u_x, u_y]
            cov = [[sigma_0, 0], [0, sigma_0]]
            coordinate_x, coordinate_y = 1, 1
            while not (0 < coordinate_x < Nx * Lx and 0 < coordinate_y < Ny * Ly):
                coordinate_x, coordinate_y = np.random.multivariate_normal(mean, cov)
            target_true_msg[i][0] = np.floor(coordinate_x / Lx)  # 向下取整
            target_true_msg[i][1] = np.floor(coordinate_x / Lx)
        target_path[i].append(target_true_msg[i])


def target_move():
    # 目标移动 主要是处理边界，边界用无限拼接来处理，类似贪吃蛇。
    # 改动，改为最多只能够停留在边界不能够出去。卡死在那里
    for target, type, target_true in zip(target_msg, target_type, target_true_msg):
        u_x = target_true[0] * Lx + Lx / 2
        u_y = target_true[1] * Ly + Ly / 2
        v = target_msg[2]
        phi = target_msg[3]
        new_u_x, new_u_y = 0, 0
        if type == 0 or type == 1:
            # 正态分布
            mean = [u_x, u_y]
            cov = [[sigma_e * delta_t, 0], [0, sigma_e * delta_t]]
            while 0 < new_u_x < Nx * Lx and 0 < new_u_y < Ny * Ly:
                new_u_x, new_u_y = np.random.multivariate_normal(mean, cov)

        elif type == 2:
            phi = np.random.random() * 2 * np.pi  # 均匀分布随机一个角度phi
            new_u_x = u_x + v * delta_t * np.cos(phi)
            new_u_y = u_y + v * delta_t * np.sin(phi)

        elif type == 3:
            new_u_x = u_x + v * delta_t * np.cos(phi)
            new_u_y = u_y + v * delta_t * np.sin(phi)

        if type == 2 or type == 3:
            ## 边界处理过程
            new_u_x, new_u_y = target_move_border_process([u_x, u_y], [new_u_x, new_u_y], phi)

            # if not (0 < new_u_x < Nx * Lx and 0 < new_u_y < Ny * Ly):
            #     # 出去了那么就用sin cos 等方法来处理
            #     if np.pi / 2 < phi < 3 * np.pi / 2:
            #         add_y = u_x * np.tan(np.pi - phi)
            #         new_u_y = u_y + add_y
            #         new_u_x = 0
            #         if new_u_y < 0:
            #             add_x = u_y / np.tan(np.abs(np.pi - phi))
            #             new_u_y = 0
            #             new_u_x = u_x - add_x
            #         if new_u_y > Ny * Ly:
            #             add_x = (Ny * Ly - u_y) / np.tan(np.abs(np.pi - phi))
            #             new_u_y = Ny * Ly
            #             new_u_x = u_x - add_x
            #     else:
            #         add_y = (Nx * Lx - u_x) * np.tan(phi)
            #         new_u_y = u_y + add_y
            #         new_u_x = Nx * Lx
            #         if new_u_y < 0:
            #             add_x = np.abs(u_y / np.tan(phi))
            #             new_u_y = 0
            #             new_u_x = u_x + add_x
            #         if new_u_y > Ny * Ly:
            #             add_x = np.abs((Ny * Ly - u_y) / np.tan(phi))
            #             new_u_y = Ny * Ly
            #             new_u_x = u_x + add_x
        # new_u_x = (new_u_x / (Lx * Nx) - np.floor(new_u_x / (Lx * Nx))) * (Lx * Nx)  # 无限拼接的变换
        # new_u_y = (new_u_y / (Ly * Ny) - np.floor(new_u_y / (Ly * Ny))) * (Ly * Ny)  # 无限拼接的变换11
        target_true[0] = np.floor(new_u_x / Lx)  # 里面存的是所在栅格编号
        target_true[1] = np.floor(new_u_y / Ly)
        # 添加路径
        target_path.append(target_true)


def target_move_border_process(core, new_core, phi):
    # 目标运动的边界处理
    u_x = core[0]
    u_y = core[1]

    new_u_x = new_core[0]
    new_u_y = new_core[1]

    if not (0 < new_u_x < Nx * Lx and 0 < new_u_y < Ny * Ly):
        # 出去了那么就用sin cos 等方法来处理
        if np.pi / 2 < phi < 3 * np.pi / 2:
            add_y = u_x * np.tan(np.pi - phi)
            new_u_y = u_y + add_y
            new_u_x = 0
            if new_u_y < 0:
                add_x = u_y / np.tan(np.abs(np.pi - phi))
                new_u_y = 0
                new_u_x = u_x - add_x
            if new_u_y > Ny * Ly:
                add_x = (Ny * Ly - u_y) / np.tan(np.abs(np.pi - phi))
                new_u_y = Ny * Ly
                new_u_x = u_x - add_x
        else:
            add_y = (Nx * Lx - u_x) * np.tan(phi)
            new_u_y = u_y + add_y
            new_u_x = Nx * Lx
            if new_u_y < 0:
                add_x = np.abs(u_y / np.tan(phi))
                new_u_y = 0
                new_u_x = u_x + add_x
            if new_u_y > Ny * Ly:
                add_x = np.abs((Ny * Ly - u_y) / np.tan(phi))
                new_u_y = Ny * Ly
                new_u_x = u_x + add_x

    return new_u_x, new_u_y


def get_target_map():
    # 返回目标存在图 ,
    target_map = np.zeors((Nt, Nx, Ny))
    for target_site, t in zip(target_true_msg, range(Nt)):
        target_map[t, target_site[0], target_site[1]] = 1
    return target_map


# 计算连通图
def cal_connect_map(UAV_group):
    # 计算连通图
    # 分为三步 1.计算距离 2.计算邻接矩阵 3.计算联通矩阵
    shape = (Nv, Nv)

    # 计算距离
    dis = [list(map(lambda UAV_j: np.sum((np.array((Lx, Ly)) * (UAV_i.site - UAV_j.site)) ** 2), UAV_group)) for
           UAV_i in UAV_group]
    dis = np.sqrt(np.array(dis))
    ##
    border = np.zeros(shape)
    for i in range(Nv):
        for j in range(Nv):
            if dis[i, j] < UAV_group[i].broadcastLen:
                border[i, j] = 1
    ## 邻接矩阵的n阶和
    sum = np.zeros(shape)
    for i in range(Nv):
        sum = sum + np.mat(border) ** (i + 1)
    result = (sum > np.zeros(shape)).astype(float)
    return result


# 初始化目标概率图搜索图
def init_pro_map(UAV_group):
    # 数据结构定义为list里面放每个目标的概率分布图np.array()
    init_pro_map = []
    for i in range(Nt):
        mu_x = target_msg[i, 0]*Lx+Lx/2  # x* x点
        mu_y = target_msg[i, 1]*Ly+Ly/2  # y* y点
        v = target_msg[i, 2]  # v 速度
        phi = target_msg[i, 3]  # phi 角度
        targeti_pro_map = np.zeros((Nx, Ny), dtype=float)
        if target_type[i] == 0:
            # 均匀分布
            targeti_pro_map = targeti_pro_map + 1 / (Nx * Ny)
        elif target_type[i] == 1:
            # 布朗运动
            sigma = sigma_0 + t0 * sigma_e
            targeti_pro_map = get_Gauss_map([mu_x, mu_y], sigma)
        elif target_type[i] == 2:
            # 超级难得积分
            sigma = sigma_0
            targeti_pro_map_con=get_Gauss_map([mu_x, mu_y], sigma) # 条件概率
            temp_pro_map = np.zeros((Nx, Ny))
            for i in range(Nx):
                for j in range(Ny):
                    core = [i, j]
                    R = v * t0
                    pro_map_ij = get_type2_promap(core, R)  # 条件概率
                    temp_pro_map += targeti_pro_map_con[i, j] * pro_map_ij  # 全概率公式
            targeti_pro_map = temp_pro_map
        elif target_type[i] == 3:
            # 高斯分布积分
            sigma = sigma_0
            mu_x = mu_x + v * t0 * np.cos(phi)
            mu_y = mu_y + v * t0 * np.sin(phi)
            targeti_pro_map = get_Gauss_map([mu_x, mu_y], sigma)

        # 归一化
        targeti_pro_map = targeti_pro_map / np.sum(targeti_pro_map)

        init_pro_map.append(targeti_pro_map)


    for UAV in UAV_group:
        UAV.pro_maps = init_pro_map


# 初始化两个信息素图
def init_phero_map(UAV_group):
    for UAV in UAV_group:
        UAV.phero_rej_map = np.ones((Nx, Ny)) * init_phero
        UAV.phero_att_map = np.ones((Nx, Ny)) * init_phero


# 高斯分布的概率密度函数
def Gaussian(x, y, mu_x, mu_y, sigma):
    # x_,y_ 表示x*，y* (目标的)
    right = np.exp((-(x - mu_x) ** 2 - (y - mu_y) ** 2) / (2 * sigma))
    left = 1 / (2 * np.pi * sigma)
    return left * right


def init_UAV_position(UAV_group):
    # 随机分布在边界
    sites_x = np.random.randint(0, Nx, Nv)
    sites_y = np.random.randint(0, Ny, Nv)
    sites = np.transpose(np.array([sites_x, sites_y]))  # 转化成n*2 的矩阵
    bound = [0, Nx - 1, 0, Ny - 1]  # 四边的边界
    for UAV, site in zip(UAV_group, sites):
        bun_type = np.random.randint(0, 2) # 替换 x or y
        bun_index = np.random.randint(bun_type, bun_type + 2) # 替换 max or min
        site[bun_type] = bound[bun_index]  # 随机选出一边来用边界替代
        UAV.add_site(site)



# 成功探测到目标后 改变 detect_flag
def set_detect_flag(UAV_group, target_map, index_visit_map):
    add_detect_flag = np.ones(Nv) # 1 探测 0 不探测 注意初始化为1
    all_visit_map = np.zeros((Nx, Ny))

    for i in range(Nv):
        all_visit_map += index_visit_map[i]
    for i in range(Nt):
        temp = all_visit_map * target_map[i]
        flag = np.sum(temp)
        if flag > 0: # 表示探测到了
            add_detect_flag[i] = 0
    for UAV in UAV_group:
        UAV.detect_flag = ((UAV.detect_flag * add_detect_flag) > np.zeros(Nv)).astype(float)

def get_index_plan_map(UAV_group):
    # 带无人机下标的路径计划map 三维
    index_plan_map = np.zeros((Nv, Nx, Ny))
    for i in range(Nv):
        plan_path=UAV_group[i].plan_path
        for path_site in plan_path:
            index_plan_map[i, path_site[0], path_site[1]] = 1  # 无人机路径信息进行赋值
    return index_plan_map

def get_index_visit_map(UAV_group):
    index_visit_map = np.zeros((Nv, Nx, Ny))  # 三维的visit map 带有下标
    for i in range(Nv):
        index_visit_map[i, UAV_group[i].site[0], UAV_group[i].site[1]] = 1  # 对访问位置信息赋值
    return index_visit_map

if __name__ == '__main__':
    iter_num = 2000

    # 初始化目标
    init_target_type()
    init_target_position() # 里面有addpath

    # 无人机群初始化
    UAV_group = [UAV() for i in range(Nv)]  # 无人机群，无人机存储在这里面
    init_UAV_position(UAV_group)  # 初始化无人机位置 # 通过UAV.add_site(site) 来赋值位置和添加path

    # 初始化搜索图
    init_pro_map(UAV_group)  # 初始化概率分布图，每架无人机一样
    init_phero_map(UAV_group)  # 初始化信息素大小

    for iter in range(iter_num):
        # 　连通图
        connect_map = cal_connect_map(UAV_group)  # 连通图，每走一步就重新计算一下

        index_plan_map=get_index_plan_map(UAV_group) # 三维的路径规划图，第一个下标，表示无人机的下标

        index_visit_map=get_index_visit_map(UAV_group)# 三维的visit map 带有下标

        target_map=get_target_map() # 目标存在图 3维

        # 计算无人机的V和R,route_plan 阵
        for i in range(Nv):
            UAVi_visit_map = np.zeros((Nx, Ny))  # V矩阵

            UAVi_phero_plan_map = np.zeros((Nx, Ny))  # route plan 阵
            for j in range(Nv):
                if connect_map[j, i] == 1:
                    UAVi_visit_map += index_visit_map[i, :, :]
                    if i != j:  # 这里不包括自己的信息
                        UAVi_phero_plan_map += index_plan_map[i, :, :]

            UAV_group[i].phero_plan_map = D_r*UAVi_phero_plan_map # d_r 为释放信息素量
            UAV_group[i].visit_map = UAVi_visit_map
            # 这里是个三维的
            UAV_group[i].detect_maps = np.array([UAVi_visit_map * detect_map for detect_map in target_map])

        set_detect_flag(UAV_group, target_map, index_visit_map)  # 设置已探测到的目标flag

        for UAV in UAV_group:
            UAV.update_search()
            UAV.update_switch_map(iter)
            UAV.update_phero()
            UAV.predic_target()

            # 无人机下一步
            UAV.decision()
        # 目标移动
        target_move()

    # 这里应当添加画图
    # 最后结尾的时候最后一个决策也没有判断结果


    # for i in range(iter_num):
    #     # 先决策，走一步
    #     for UAV, i in zip(UAV_group, range(Nv)):
    #         # 循环的时候关键点在于计算 Vist_map and  Detect_map
    #         UAVi_path = UAV.decision()  #
    #         index_visit_map[i, UAV.site[0], UAV.site[1]] = 1  # 对访问位置信息赋值
    #         for path_site in UAVi_path:
    #             index_plan_map[i, path_site[0], path_site[1]] = 1  # 无人机路径信息进行赋值
    #     # 目标走一步
    #     target_move()
    #     target_map = get_target_map()  # 目标存在图
    #     # 计算无人机的V和R
    #     for i in range(Nv):
    #         UAVi_visit_map = np.zeros((Nx, Ny))  # V矩阵
    #         UAVi_detect_map = np.zeros((Nx, Ny))  # R阵
    #         UAVi_phero_plan_map = np.zeros((Nx, Ny))  # route plan 阵
    #
    #         for j in range(Nv):
    #             if connect_map[j, i] == 1:
    #                 UAVi_visit_map += index_visit_map[i, :, :]
    #                 if i != j:  # 这里不包括自己的信息
    #                     UAVi_phero_plan_map += index_plan_map[i, :, :]
    #
    #         UAV_group[i].phero_plan_map = UAVi_phero_plan_map
    #         UAV_group[i].visit_map = UAVi_visit_map
    #         # 这里是个三维的
    #         UAV_group[i].detect_maps = np.array([UAVi_visit_map * detect_map for detect_map in target_map])
    #
    #     set_detect_flag(UAV_group, target_map, index_visit_map)  # 设置已探测到的目标flag
    #
    #     UAV.update_search()
    #     UAV.update_phero()
    #     UAV.predic_target()


        # 　余下内容
        # 目标路径存储 ok
        # 无人机探测到目标后的处理过程 ok
        # 路径规划信息素 ok
        # decision 里面还有进行一些细节的处理 ok
        # value 的计算 ok

        # 目标运动边界处理 没有搞定
        # 1.
        # 目标本身运动     到了边界，若要继续出边界，就停下来。
        # 2.
        # 目标运动概率预测分布   做归一化处理

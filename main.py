import numpy as np
import math
from scipy.integrate import dblquad
from tool import get_type2_promap
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
    [1.1, 2.2, 10, 0],
    [1, 2, 10, 1],
    [3, 5, 20, np.nan],
])  # 目标信息，分别是x,y （物理坐标）,v，phi(方位角)# 。
sigma_0 = 3  # δ_0 初始位置的方差
sigma_e = 3  # 布朗运动的方差
Nt = target_msg.shape[0]  # 目标数量
target_type = np.zeors((1, Nt))  # 存放目标类型
target_true_msg = np.zero((Nt, 2))  # 模拟的真实栅格,
##战场环境离散
Ny = 100  # y轴上格子的个数
Nx = 100  # x轴上格子的个数
Lx, Ly = 10, 10  # 每个格子的长和宽
connect_map = np.zeros((Nx, Ny))  # 连通图，设置为全局变量
t0 = 3  # 无人机机进入战场时间
delta_t = 2  # 探索间隔时间

# 无人机共同性能信息
Nv = 10  # 无人机架数
PD = 0.95  # 探测概率
PF = 0.02  # 虚警概率

E = np.ones((Nx, Ny))  # 单位矩阵用于计算

### 信息素的基本信息

init_phero=2 # 初始信息素的值
# 吸引信息素信息
G_a=0.5 # 信息素传播因子
E_a=0.5 # 信息素挥发因子
D_a=1   # 信息素的释放量
# 挥发信息素信息
G_r=0.5 # 信息素传播因子
E_r=0.5 # 信息素挥发因子
D_r=10  # 信息素的释放量


class UAV:
    def __init__(self, broadcastLen=20, ):
        self.site = np.ones((1, 2))
        # 位置用二元数来表示
        self.path = list()
        self.path.append(self.site)
        # path记录无人机飞行路径
        self.broadcastLen = broadcastLen
        # 无人机的广播距离
        self.pro_maps = []
        # 概率图 设置数据结构为三维数组 每个目标的概率分布图
        self.phero_att_map = np.zeros((Nx, Ny))
        self.phero_rej_map = np.zeros((Nx, Ny))
        # 吸引信息素和抑制信息素图
        self.enviro_map = np.zeros((Nx, Ny))
        # 环境确定度
        self.visit_map = np.zeros((Nx, Ny))

        # 访问(探测)结果图
        self.detect_flag = np.ones((1, Nt))
        # 探测标志位，需要搜索用1来表示，不需要搜索则用0来表示

        # 访问信息图 数据结构同概率分布图
        self.detect_maps = np.zeros((Nt, Nx, Ny))
        self.last_vist_map=np.zeors((Nx,Ny)) # 上一次访问时间，根据访问矩阵来更新
        self.pero_switch=np.zeros((Nx,Ny)) # 信息素开关矩阵
    def decision(self):

    # 决策函数，根据性能指标计算，返回可行路径中的一点

    def update_phero(self):
        # 根据访问信息更新信息素

    def get_G_alpha_map(self):
        # 计算吸引信息素扩散矩阵
        G_alpha_map=np.zeros((Nx,Ny))
        for i in range(Nx):
            for j in range(Ny):
                neighbor_num=0
                for neighbor_x in range(3):
                    for neighbor_y in range(3):
                        g_x=i+neighbor_x-1
                        g_y=j+neighbor_y-1
                        if(0<=g_x<Nx and 0<=g_y<Ny):
                            neighbor_num+=1
                            G_alpha_map[i,j]+=self.phero_att_map[g_x,g_y]
                G_alpha_map[i,j]/=neighbor_num
        return G_alpha_map

    def get_G_r_map(self):
        # 计算抑制信息素扩散矩阵
    def update_search(self):
        # 搜索图探测更新
        # 方法根据文章中的矩阵表达式来更新

        # 先分子
        H = self.visit_map * PD + (E - self.visit_map) * PF
        for pro_map,detect_map,i in zip(self.pro_maps,self.detect_maps,range(Nt)):
            if self.detect_flag[i]==0:  # 等于0的部分不需要更新
                continue
            numerator = (H * (2 * detect_map - E) + (E - detect_map)) * pro_map
            pdn=E*PF+(PD-PF)*pro_map
            denominator=pdn*(2*detect_map-E)+(E-detect_map)
            self.pro_maps[i]=numerator/denominator
            # 注意这里直接改变pro_map 没用这样写不改变 pro_maps，np.array()类型传回的不是引用


    def predic_target(self):
        # 根据探测结果更新搜素图
        # 预测目标在delt_t 后的概率分布类似于初始化的过程
        for target_i in range(Nt):
            if self.detect_flag[target_i] == 0: # 已经探测到的目标 不需要继续更新将detect_flag置0
                continue

            v = target_msg[target_i, 2]  # v 速度
            phi = target_msg[target_i, 3]  # phi 角度
            now_pro_map_ij = self.pro_maps[target_i] # 目前的概率
            predict_pro_map=np.zeros((Nx,Ny))
            if target_type[target_i] == 0 or target_type[target_i] == 1:
                # 不知道速度布朗运动,一种正态分布
                sigma = delta_t * sigma_e
                for condition_i in range(Nx): #条件概率 条件x_i
                    for condition_j in range(Ny): #条件概率 条件y_i
                        condition_pro_map_ij=get_Gauss_map([condition_i,condition_j],sigma)# 条件概率图
                        predict_pro_map+=now_pro_map_ij[condition_i,condition_j]*condition_pro_map_ij # 全概率公式
            elif target_type[target_i] == 2:
                # 类似于初始化的时候
                for condition_i in range(Nx): #条件概率 条件x_i
                    for condition_j in range(Ny): #条件概率 条件y_i
                        R=v*delta_t
                        core=[condition_i,condition_j]
                        condition_pro_map_ij =get_type2_promap(core,R)   # 条件概率图
                        predict_pro_map=now_pro_map_ij[condition_i,condition_j]*condition_pro_map_ij
            elif target_type[target_i] == 3:
                # 一起位移而已，先放着
                for condition_i in range(Nx): #条件概率 条件x_i
                    for condition_j in range(Ny): #条件概率 条件y_i

            self.pro_maps[target_i] = predict_pro_map


def get_Gauss_map(core,sigma):
    # 返回gauss
    pro_map=np.zeros((Nx,Ny))
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
            pro_map[i, j] = dblquad(Gaussian, low_bound_x, hig_bound_x, lambda x: low_bound_y,
                                            lambda x: hig_bound_y, args=(mu_x, mu_y, sigma))
    pro_map = pro_map / np.sum(pro_map)
    return pro_map

def init_target_position():
    #     初始化目标的位置，type 0 随机生成，其他的高斯分布生成，越界了就再来一次
    for target, type,target_true in zip(target_msg, target_type,target_true_msg):
        if (type == 0):
            # 均匀分布随机
            point_x = np.random.randint(0, Nx);  # 左闭右开区间
            point_y = np.random.randint(0, Ny);  # 左闭右开区间
            target=[point_x,point_y]
        else:
            u_x= target[0]*Lx+Lx / 2
            u_y= target[1]*Ly+Ly / 2
            mean=[u_x,u_y]
            cov=[[sigma_0,0],[0,sigma_0]]
            coordinate_x,coordinate_y=-1,-1
            while(0<coordinate_x<Nx*Lx and 0<coordinate_y<Ny*Ly):
                coordinate_x,coordinate_y=np.random.multivariate_normal(mean,cov)
            target_true[0]=np.floor(coordinate_x/Lx) # 向下取整
            target_true[1]=np.floor(coordinate_x/Lx)

def target_move():
    # 目标移动 主要是处理边界，边界用无限拼接来处理，类似贪吃蛇。
    for target, type,target_true in zip(target_msg, target_type,target_true_msg):
        u_x = target_true[0] * Lx + Lx / 2
        u_y = target_true[1] * Ly + Ly / 2
        new_u_x,new_u_y=0,0
        if type==0 or type==1:
            # 正态分布
            mean = [u_x, u_y]
            cov = [[sigma_e*delta_t, 0], [0, sigma_e*delta_t]]
            new_u_x, new_u_y = np.random.multivariate_normal(mean, cov)
        elif type==2:
            phi=np.random.random()*np.pi  # 均匀分布随机一个角度phi
            new_u_x=u_x+target_msg[2]*np.cos(phi)
            new_u_y = u_y + target_msg[2] * np.sin(phi)
        elif type==3:
            phi=target_msg[3]
            new_u_x=u_x+target_msg[2]*np.cos(phi)
            new_u_y = u_y + target_msg[2] * np.sin(phi)
        new_u_x = (new_u_x / (Lx * Nx) - np.floor(new_u_x / (Lx * Nx))) * (Lx * Nx)  # 无限拼接的变换
        new_u_y = (new_u_y / (Ly * Ny) - np.floor(new_u_y / (Ly * Ny))) * (Ly * Ny)  # 无限拼接的变换

        target_true[0]=np.floor(new_u_x/Lx)*Lx+Lx/2
        target_true[1]=np.floor(new_u_y/Ly)*Ly+Ly/2


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
    ##
    sum = np.zeros(shape)
    for i in range(Nv):
        sum = sum + np.mat(border) ** (i + 1)
    result = (sum > np.zeros(shape)).astype(float)
    return result

# 初始化目标概率图搜索图
def init_pro_map():
    # 数据结构定义为list里面放每个目标的概率分布图np.array()
    init_pro_map = []
    for i in range(Nt):
        mu_x = target_msg[i, 0]  # x* x点
        mu_y = target_msg[i, 1]  # y* y点
        v = target_msg[i, 2]  # v 速度
        phi = target_msg[i, 3]  # phi 角度
        targeti_pro_map = np.zeros((Nx, Ny), type=float)
        if target_type[i] == 0:
            # 均匀分布
            targeti_pro_map = targeti_pro_map + 1 / (Nx * Ny)
        elif target_type[i] == 1:
            # 布朗运动
            sigma = sigma_0 + t0 * sigma_e
            targeti_pro_map = get_Gauss_map([mu_x, mu_y], sigma)
        elif target_type[i] == 2:
        # 超级难得积分
            sigma=sigma_0
        elif target_type[i] == 3:
            # 高斯分布积分
            sigma = sigma_0
            mu_x = mu_x + v * t0 * np.cos(phi)
            mu_y = mu_y + v * t0 * np.sin(phi)
            targeti_pro_map = get_Gauss_map([mu_x, mu_y], sigma)

        # 归一化
        targeti_pro_map = targeti_pro_map / np.sum(targeti_pro_map)

        if target_type==2:
            temp_pro_map=np.zeros((Nx,Ny))
            for i in range(Nx):
                for j in range(Ny):
                    core=[i,j]
                    R=v*t0
                    pro_map_ij=get_type2_promap(core,R) # 条件概率
                    temp_pro_map+=targeti_pro_map[i,j]*pro_map_ij # 全概率公式
            targeti_pro_map=temp_pro_map
        init_pro_map.append(targeti_pro_map)

# 初始化两个信息素图
def init_phero_map(UAV_group):
    for UAV in UAV_group:
        UAV.phero_rej_map=np.ones((Nx,Ny))*init_phero

# 高斯分布的概率密度函数
def Gaussian(x, y, mu_x, mu_y, sigma):
    # x_,y_ 表示x*，y* (目标的)
    right = np.exp((-(x - mu_x) ** 2 - (y - mu_y) ** 2) / (2 * sigma_e))
    left = 1 / (2 * np.pi * sigma)
    return left * right

if __name__ == '__main__':
    iter_num = 2000
    UAV_group = [UAV() for i in range(Nv)]
    # 无人机群，无人机存储在这里面
    init_pro_map()
    for i in range(iter_num):
        for UAV in UAV_group:
            UAV.decision()
            UAV.update_search()

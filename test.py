# import matplotlib.pyplot as plt
# import numpy as np
#
# x = np.arange(0, 100, 10)
# y = x * 2
#
# fig = plt.figure()
# ax = fig.gca()
# ax.set_xticks(np.arange(0, 300, 10))
# ax.set_yticks(np.arange(0, 300, 10))
# plt.scatter(x, y)
# plt.xlim(0, 300)
# plt.ylim(0, 300)
# plt.grid()
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
# plt.figure(1)
# plt.figure(2)
# ax1=plt.subplot(211)
# ax2=plt.subplot(212)
# x=np.linspace(0,3,100)
# for i in range(5):
#     plt.figure(1)
#     plt.plot(x,np.exp(i*x/3))
#     plt.grid()
#     fig = plt.figure(1)
#     ax = fig.gca()
#     ax.set_xticks(np.arange(0, 300, 10))
#     ax.set_yticks(np.arange(0, 300, 10))
#
#     fig=plt.sca(ax1)
#     plt.plot(x,np.sin(i*x))
#     plt.grid()
#
#
#     plt.sca(ax2)
#     plt.plot(x,np.cos(i*x))
#     plt.grid()
# plt.show()
Nx=10
Lx=1
Ny=10
Ly=1
def plot_path(data,i):
    # 给出无人机or目标的移动路径图，画出来
    fig=plt.figure(i)
    for path_i in data:
        x_point=[site[0] for site in path_i]
        y_point=[site[1] for site in path_i]
        plt.plot(x_point,y_point)
    # 画线
    plt.grid()
    ax = fig.gca()
    ax.set_xticks(np.arange(0, Nx*Lx, Lx))
    ax.set_yticks(np.arange(0, Ny*Ly, Ly))

def plot_detect(data):
    # 画折线图
    plt.figure(3)
    x_point=[find[0] for find in data ]
    y_point=[find[1] for find in data ]
    plt.plot(x_point, y_point)


detect=[[i,i+np.random.randint(0,3)] for i in range(10)]
plot_detect(detect)
path1=[[ (i,i+np.random.randint(-3,4)) for i in range(10)] for j in range(3) ]
plot_path(path1,1)
plot_path(path1,2)
plt.show()
import math
import matplotlib.pyplot as plt
from numpy.linalg import norm
import random

from control.pure_pursuit.script.pure_pursuit import *
from optimization.gradient_descent.script.gradient_descent import *
from ..script.potential_field import *


# 参考代码：https://blog.csdn.net/weixin_39549161/article/details/88712443

k = 0.1  # 前视距离系数
Lfc = 2.0  # 前视距离
Kp = 1.0  # 速度P控制器系数
dt = 0.1  # 时间间隔，单位：s
L = 2.9  # 车辆轴距，单位：m


class VehicleState:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v


def update(state, a, delta):
    state.x = state.x + state.v * math.cos(state.yaw) * dt
    state.y = state.y + state.v * math.sin(state.yaw) * dt
    state.yaw = state.yaw + state.v / L * math.tan(delta) * dt
    state.v = state.v + a * dt
    return state


def PControl(target, current):
    a = Kp * (target - current)
    return a


def pure_pursuit_control(state, cx, cy, pind):
    ind = calc_target_index(state, cx, cy)
    if pind >= ind:
        ind = pind
    if ind < len(cx):
        tx = cx[ind]
        ty = cy[ind]
    else:
        tx = cx[-1]
        ty = cy[-1]
        ind = len(cx) - 1
    alpha = math.atan2(ty - state.y, tx - state.x) - state.yaw
    if state.v < 0:  # back
        alpha = math.pi - alpha
    Lf = k * state.v + Lfc
    delta = math.atan2(2.0 * L * math.sin(alpha) / Lf, 1.0)
    return delta, ind


def calc_target_index(state, cx, cy):
    # 搜索最临近的路点
    dx = [state.x - icx for icx in cx]
    dy = [state.y - icy for icy in cy]
    d = [abs(math.sqrt(idx ** 2 + idy ** 2)) for (idx, idy) in zip(dx, dy)]
    ind = d.index(min(d))
    L = 0.0
    Lf = k * state.v + Lfc
    while Lf > L and (ind + 1) < len(cx):
        dx = cx[ind + 1] - cx[ind]
        dy = cx[ind + 1] - cx[ind]
        L += math.sqrt(dx ** 2 + dy ** 2)
        ind += 1
    return ind


def Coord2Vec(data: list):
    """将 list 中的 tuple 转成 numpy.array"""
    changed_data = []
    for item in data:
        changed_data.append(np.array([item]).T)
    if len(changed_data) == 1:
        return changed_data.pop()
    else:
        return changed_data


def GenerateRandomPositon(num, limit_x, limit_y):
    """生成随机坐标点

    Args:
        num: int类，设定生成随机位置点的数量
        limit_x: list类，区间限制范围[x0, x1]， 表示生成的随机点落在[x0, x1]范围内
        limit_y: list类，区间限制范围[y0, y1]， 表示生成的随机点落在[y0, y1]范围内

    Returns:
        randPosList: list类，生成的随机点坐标列表(random position list)
    """
    randPosList = []
    for i in range(num):
        x = random.randint(limit_x[0], limit_x[1])
        y = random.randint(limit_y[0], limit_y[1])
        randPosList.append((x, y))
    return randPosList


if __name__ == '__main__':
    #  设置目标路点
    cx = np.arange(0, 50, 1)
    cy = [math.sin(ix / 5.0) * ix / 2.0 for ix in cx]
    target_speed = 10.0 / 3.6  # [m/s]
    T = 100.0  # 最大模拟时间
    # 设置车辆的初始状态
    state = VehicleState(x=-0.0, y=-3.0, yaw=math.pi / 2, v=0.0)
    lastIndex = len(cx) - 1
    time = 0.0
    x = [state.x]
    y = [state.y]
    yaw = [state.yaw]
    v = [state.v]
    t = [0.0]
    target_ind = calc_target_index(state, cx, cy)

    while T >= time and lastIndex > target_ind:
        ai = PControl(target_speed, state.v)

        pind = target_ind
        ind = calc_target_index(state, cx, cy)
        if pind >= ind:
            ind = pind

        if ind < len(cx):
            tx = cx[ind]
            ty = cy[ind]

        else:
            tx = cx[-1]
            ty = cy[-1]
            ind = len(cx) - 1
        target_ind = ind

        # 设置一系列障碍物位置
        obsPosList = [(2, 0, 0), (6, 0, 0), (10, 5, 0), (30, -5, 0), (40, 20, 0), (45, 5, 0),
                      (15, 0, 0), (7, 3.5, 0), (20, -7.5, 0)]
        obsPosList = Coord2Vec(obsPosList)
        potentFiled = PotentialField()

        goalPos = np.array([[tx], [ty], [0]])
        roboPos = np.array([[state.x], [state.y], [0]])
        potentFiled(goalPos, obsPosList)

        # 通过人工势场修正待跟踪的目标点，新的目标点是沿 new_roboPos 方向，距离大小是
        # goalPos 与 roboPos 的距离
        # 梯度下降法
        new_roboPos = GradientDescent(roboPos, potentFiled.func, potentFiled.grad_func)

        goal2robo_dist = np.linalg.norm(goalPos - roboPos)
        modified_goal = roboPos + (new_roboPos - roboPos) * goal2robo_dist
        tx = modified_goal[0, 0]
        ty = modified_goal[1, 0]

        # 纯跟踪
        refPoint = [tx, ty]
        currPose = [state.x, state.y, state.yaw]
        currVel = state.v
        purePursuit = PurePursuit(k, L, math.pi / 4)
        di = purePursuit.controller(refPoint, currPose, currVel)

        state = update(state, ai, di)
        time = time + dt
        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        v.append(state.v)
        t.append(time)
        plt.cla()
        plt.plot(cx, cy, ".r", label="course")
        plt.plot(x, y, "-b", label="trajectory")
        plt.scatter(modified_goal[0, 0], modified_goal[1, 0], s=20, c=None,
                    edgecolors='k', marker='^', linewidths=2)
        plt.plot(cx[target_ind], cy[target_ind], "go", label="target")
        plt.axis("equal")
        plt.grid(True)
        plt.title("Speed[km/h]:" + str(state.v * 3.6)[:4])
        plt.quiver(roboPos[0], roboPos[1],
                   new_roboPos[0] - roboPos[0], new_roboPos[1] - roboPos[1],
                   color='r')
        for obsPos in obsPosList:
            plt.scatter(obsPos[0], obsPos[1], s=20, c='b', marker='o',
                        linewidths=2)
        plt.pause(0.02)
        plt.cla()

# 初始化绘图对象
fig, ax = plt.subplots(figsize=(14, 7))


def show(roboPos, goalPos, obsPosList):
    """显示运动轨迹"""
    # 设置XY轴的刻度
    ax.xaxis.set_ticks(np.arange(0, 100, 5))
    ax.yaxis.set_ticks(np.arange(0, 100, 5))

    # 设置XY轴的区间范围
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 50)

    ax.scatter(roboPos[0], roboPos[1], s=20, c=None, edgecolors='r', marker='*',
               linewidths=2)
    ax.scatter(goalPos[0], goalPos[1], s=20, c=None, edgecolors='k', marker='^',
               linewidths=2)

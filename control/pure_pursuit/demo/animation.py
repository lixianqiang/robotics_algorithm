import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pure_pursuit import PurePursuit

# 参考代码：https://blog.csdn.net/weixin_39549161/article/details/88712443
# 参考代码: https://zhuanlan.zhihu.com/p/258086374

class VehicleState:
    def __init__(self, x=0.0, y=2.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v

def CarStateUpdate(state, a, delta, dt):
    state.x = state.x + state.v * np.cos(state.yaw) * dt
    state.y = state.y + state.v * np.sin(state.yaw) * dt
    state.yaw = state.yaw + state.v / L * np.tan(delta) * dt
    state.v = state.v + a * dt
    return state

def PControl(target, current):
    Kp = 1.0  # 速度P控制器系数
    a = Kp * (target - current)
    return a

def init_func():
    global dt, refPoints, k, L, Lfc, target_speed, purePursuit, state, point_ani
    # 参考轨迹
    t = np.arange(0, 10*np.pi, dt)
    x = t.tolist()
    y = list(5 * np.sin(1 / 5 * t) + np.cos(t))
    refPoints = (x, y)
    plt.plot(refPoints[0], refPoints[1])

    k = 0.1  # 前视距离系数
    L = 1.0  # 车辆轴距，单位：m
    Lfc = 1.0  # 前视距离
    target_speed = 10.0 / 3.6  # [m/s]
    purePursuit = PurePursuit(k, L, np.pi / 4)

    state = VehicleState(x=-0.0, y=-2.0, yaw=np.pi/2, v=0.0)

    plt.title('Pure_Pursuit')
    point_ani, = plt.plot(state.x, state.y, "ro")
    return point_ani,

def update_points(time):
    global target_speed, state, purePursuit, xdata, ydata, Lfc, dt, \
        minIdx, refPoints, simTime, point_ani, dt


    currPose = [state.x, state.y, state.yaw]
    currVel = state.v

    allDist = [np.hypot(refPoint[0] - currPose[0], refPoint[1] - currPose[1])
               for refPoint in zip(refPoints[0], refPoints[1])]
    minIdx = allDist.index(min(allDist))

    # 根据前视距离来选定点
    L = 0
    Lf = k * state.v + Lfc
    while Lf > L and (minIdx + 1) < len(refPoints[0]):
        dx = refPoints[0][minIdx + 1] - refPoints[0][minIdx]
        dy = refPoints[1][minIdx + 1] - refPoints[1][minIdx]
        L += np.sqrt(dx ** 2 + dy ** 2)
        minIdx += 1

    if minIdx < len(refPoints[0]):
        refPoint = [refPoints[0][minIdx], refPoints[1][minIdx]]
    else:
        refPoint = [refPoints[0][-1], refPoints[1][-1]]

    ai = PControl(target_speed, state.v)

    di = purePursuit.controller(refPoint, currPose, currVel)

    state = CarStateUpdate(state, ai, di, dt)

    point_ani.set_data(state.x, state.y)

    point_ani.set_marker("o")
    point_ani.set_markersize(8)

    text_pt.set_position((state.x, state.y))
    text_pt.set_text("x=%.3f, y=%.3f" % (state.x, state.y))
    return point_ani, text_pt,

if __name__ == '__main__':

    purePursuit = None
    k = None
    L = None
    Lfc = None
    target_speed = None
    state = None
    point_ani = None
    refPoints = None

    simTime = 17
    dt = 0.1
    fig = plt.figure(tight_layout=True)
    plt.grid(ls="--")
    text_pt = plt.text(4, 0.8, '', fontsize=16)
    ani = animation.FuncAnimation(fig,
                                  update_points,
                                  np.arange(0, simTime, dt),
                                  init_func,
                                  interval=100,
                                  blit=True,
                                  repeat=False)

    plt.show()
    # 若需要生成gif动画取消下面注释
    # ani.save('pure_pursuit.gif', writer='imagemagick', fps=10)
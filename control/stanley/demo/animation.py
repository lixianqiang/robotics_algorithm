import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.misc import derivative
from stanley import Stanley

# 参考代码：https://blog.csdn.net/renyushuai900/article/details/98460758

class VehicleState:
    def __init__(self, x=0.0, y=2.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v

def CarStateUpdate(state, a, delta, dt):
    state.x = state.x + state.v * np.cos(state.yaw) * dt
    state.y = state.y + state.v * np.sin(state.yaw) * dt
    state.yaw = state.yaw + state.v * math.tan(delta) / L * dt
    state.v = state.v + a * dt
    return state

def PControl(target, current):
    Kp = 1.0  # 速度P控制器系数
    a = Kp * (target - current)
    return a

if __name__ == '__main__':
    dt = 0.1
    L = 1.0  # 车辆轴距，单位：m
    k = 0.5  # 增益参数
    target_speed = 10.0 / 3.6  # [m/s]
    state = VehicleState(x=-2.0, y=-0.0, yaw=np.pi/2, v=0.0)
    stanley = Stanley(k, L, np.pi / 4)

    # 参考轨迹1
    t = np.arange(0, 10*np.pi, dt)
    x = t.tolist()
    x = x[::]
    y = list(5 * np.sin(1 / 5 * t) + np.cos(t))

    # # 参考轨迹2
    # t = np.arange(0, 10, dt)
    # x = t.tolist()
    # x = x[::]
    # y = [0] * t

    # 参考轨迹点
    domains, ranges = t, np.array((x, y))
    polyFunc = interp1d(domains, ranges, kind=5, fill_value="extrapolate")
    refPoses = [[], [], []]
    for i in t:
        point = polyFunc(i)
        gradVal = derivative(polyFunc, i, dx=1e-6, n=1)
        yaw = math.atan2(gradVal[1], gradVal[0])
        refPoses[0].append(point[0])
        refPoses[1].append(point[1])
        refPoses[2].append(yaw)

    plt.scatter(refPoses[0], refPoses[1])

    traj_x = []
    traj_y = []
    for t in np.arange(0, 17, dt):
        frontWheel_x = state.x + L * np.cos(state.yaw)
        frontWheel_y = state.y + L * np.sin(state.yaw)
        frontWheelPoint = (frontWheel_x, frontWheel_y)

        allDist = []
        for refPose in zip(refPoses[0], refPoses[1]):
            dist = np.hypot(refPose[0] - frontWheelPoint[0], refPose[1] - frontWheelPoint[1])
            allDist.append(dist)
        minIdx = allDist.index(min(allDist))

        if minIdx < len(refPoses[0]):
            refPose = [refPoses[0][minIdx], refPoses[1][minIdx], refPoses[2][minIdx]]
        else:
            refPose = [refPoses[0][-1], refPoses[1][-1], refPoses[2][-1]]

        currPose = [state.x, state.y, state.yaw]
        currVel = state.v

        ai = PControl(target_speed, state.v)
        di = stanley.controller(refPose, currPose, currVel)
        state = CarStateUpdate(state, ai, di, dt)

        traj_x.append(state.x)
        traj_y.append(state.y)

        plt.cla()
        plt.plot(refPoses[0], refPoses[1], ".r", label="course")
        plt.plot(traj_x, traj_y, "-b", label="trajectory")
        plt.plot(refPose[0], refPose[1], "go", label="target")
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.001)
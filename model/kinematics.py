from common import *
import numpy as np
class DiffDrive:
    def __init__(self, base=0.3, wheelRadius=0.05, maxWheelVel=0.09, maxWheelAcc=0.2):
        """
        maxWheelVel: 单轮最大速度
        maxWheelAcc: 单轮最大加速度
        """
        self.base = base
        self.wheelRadius = wheelRadius
        self.maxWheelVel = maxWheelVel
        self.maxAcc = maxWheelAcc
        self.maxSteer = self._maxSteer()
        self.maxCrv = self._maxCrv()  # maxCrv: max curvature
        self.minCrv = self._minCrv()  # minCrv: min curvature

        self._pos = []
        self._yaw = []
        self._vl = []
        self._vr = []
        self._timeStamp = []
        self._hasInitStat = False

    def SetInitStat(self, initPos, initYaw, init_vl, init_vr, timeStamp):
        self._pos = [initPos]
        self._yaw = [initYaw]
        self._vl = [init_vl]
        if init_vl > self.maxWheelVel:
            self._vl = [np.sign(init_vl) * self.maxWheelVel]
        self._vr = [init_vr]
        if init_vr > self.maxWheelVel:
            self._vr = [np.sign(init_vr) * self.maxWheelVel]
        self._timeStamp = [timeStamp]
        self._hasInitStat = True

    def _maxSteer(self):
        vl, vr = self.maxWheelVel, -self.maxWheelVel
        R, L = self.wheelRadius, self.base
        # 以最大速度，左右轮反向旋转
        maxSteer = R * (vl - vr) / L
        return maxSteer

    def _minCrv(self):

        # w * R = v
        # k = 1/R = w / v
        k = self.maxSteer / self.maxWheelVel
        return k

    def _maxCrv(self):

        # w * R = v
        # k = 1/R = w / v
        k = self.maxSteer / (0.3 * self.maxWheelVel)
        return k

    def pos(self):
        if len(self._pos) == 0:
            return 0, 0
        x = self._pos[-1][0]
        y = self._pos[-1][1]
        return x, y

    def yaw(self):
        if len(self._yaw) == 0:
            return 0
        return self._yaw[-1]

    def vel(self, outputVector=False):
        if len(self._vl) == 0:
            if outputVector == True:
                return 0, 0
            return 0
        # 注意这里的速度表示的速度的模
        vel = (self._vl[-1] + self._vr[-1]) * self.wheelRadius / 2
        if outputVector == True:
            vel_x = vel * cos(self._yaw[-1])
            vel_y = vel * sin(self._yaw[-1])
            return vel_x, vel_y
        return vel

    def acc(self, outputVector=False):
        if len(self._vl) < 2:
            if outputVector == True:
                return 0, 0
            return 0

        vel1 = (self._vl[-1] + self._vr[-1]) * self.wheelRadius / 2
        vel0 = (self._vl[-2] + self._vr[-2]) * self.wheelRadius / 2

        # 注意这里的加速度表示的加速度的模
        dt = (self._timeStamp[-1] - self._timeStamp[-2])
        acc = (vel1 - vel0) / dt
        if outputVector == True:
            acc_x = acc * cos(self._yaw[-1])
            acc_y = acc * sin(self._yaw[-1])
            return acc_x, acc_y
        return acc

    def crv(self):
        """crv: curvature"""
        if len(self._pos) < 2:
            return 0
        x1, y1 = self._pos[-1]
        x0, y0 = self._pos[-2]
        ds = hypot(x1 - x0, y1 - y0)
        dyaw = self._yaw[-1] - self._yaw[-2]
        crv = dyaw / ds
        return crv

    def move(self, vel, steer, timeStamp):
        steer = WrapToPi(steer)
        L, R = self.base, self.wheelRadius
        vl = (2 * vel - steer * L) / (2 * R)
        vr = (2 * vel + steer * L) / (2 * R)
        pose = self.move2(vl, vr, timeStamp)
        return pose

    def move2(self, vl, vr, timeStamp):
        if not self._hasInitStat:
            raise "仍没设置初始状态，请检查"

        R = self.wheelRadius
        L = self.base
        theta = self._yaw[-1]
        dt = timeStamp - self._timeStamp[-1]

        dx = R / 2 * (vr + vl) * cos(theta) * dt
        dy = R / 2 * (vr + vl) * sin(theta) * dt
        dtheta = R / L * (vr - vl)

        x, y = self._pos[-1][0], self._pos[-1][1]
        x += dx
        y += dy
        theta = WrapToPi(theta + dtheta)

        self._pos.append((x, y))
        self._yaw.append(theta)
        self._vl.append(vl)
        self._vr.append(vr)
        self._timeStamp.append(timeStamp)

        while len(self._pos) > 3:
            self._pos.pop(0)
            self._yaw.pop(0)
            self._vl.pop(0)
            self._vr.pop(0)
            self._timeStamp.pop(0)
        return x, y, theta


class Unicycle:

    def __init__(self, initStatus):
        self.currPose = initStatus[0], initStatus[1]
        self.yaw = initStatus[2]

    def move(self, vel, steer):
        x, y, theta = self.currPose[0], self.currPose[1], self.yaw

        dx = vel * cos(theta)
        dy = vel * sin(theta)
        dtheta = steer

        x += dx
        y += dy
        theta += dtheta

        self.currPose[0], self.currPose[1], self.yaw = x, y, theta
        return x, y, theta

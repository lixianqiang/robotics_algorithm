from math import *
class DiffDrive:
    def __init__(self, initStatus, base=0.3, wheelRadius=0.05, maxAcc=0.3, maxVel=1):
        self._pos = [(initStatus[0], initStatus[1])]
        self._yaw = [initStatus[2]]
        self._vl = [initStatus[3]]
        self._vr = [initStatus[4]]
        self._timeStamp = [initStatus[5]]
        self._base = base
        self._wheelRadius = wheelRadius
        self._maxAcc = maxAcc
        self._maxVel = maxVel

    def pos(self):
        x = self._pos[-1][0]
        y = self._pos[-1][1]
        return x, y

    def yaw(self):
        return self._yaw[-1]

    def vel(self, outputVector=False):
        """
        注意这里的速度表示的速度的模
        """
        vel = (self._vl[-1] + self._vr[-1]) * self._wheelRadius / 2
        if outputVector == True:
            vel_x = vel * cos(self._yaw[-1])
            vel_y = vel * sin(self._yaw[-1])
            return vel_x, vel_y
        return vel

    def acc(self, outputVector=False):
        """
        注意这里的加速度表示的加速度的模
        """
        if len(self._vl) < 2:
            if outputVector == True:
                return 0, 0
            return 0

        vel1 = (self._vl[-1] + self._vr[-1]) * self._wheelRadius / 2
        vel0 = (self._vl[-2] + self._vr[-2]) * self._wheelRadius / 2

        dt = (self._timeStamp[-1] - self._timeStamp[-2])
        acc = (vel1 - vel0) / dt
        if outputVector == True:
            acc_x = acc * cos(self._yaw[-1])
            acc_y = acc * sin(self._yaw[-1])
            return acc_x, acc_y
        return acc

    def curvature(self):
        if len(self._pos) < 2:
            return 0
        x1, y1 = self._pos[-1]
        x0, y0 = self._pos[-2]
        ds = hypot(x1 - x0, y1 - y0)
        dyaw = self._yaw[-1] - self._yaw[-2]
        k = dyaw / ds
        return k

    def move(self, vel, steer, timeStamp):
        steer = NormalizeAngle(steer)
        L, R = self._base, self._wheelRadius
        vl = (2 * vel - steer * L) / (2 * R)
        vr = (2 * vel + steer * L) / (2 * R)
        pose = self.move2(vl, vr, timeStamp)
        return pose

    def move2(self, vl, vr, timeStamp):
        R = self._wheelRadius
        theta = self._yaw[-1]
        L = self._base
        steer = R / L * (vr - vl)
        dt = timeStamp - self._timeStamp[-1]
        dx = R / 2 * (vr + vl) * cos(theta + steer * dt) * dt
        dy = R / 2 * (vr + vl) * sin(theta + steer * dt) * dt

        dtheta = R / L * (vr - vl)

        x, y = self._pos[-1][0], self._pos[-1][1]
        x += dx
        y += dy
        theta = NormalizeAngle(theta + dtheta)

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

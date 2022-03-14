from math import *
class DiffDrive:

    def __init__(self, initStatus, base, wheelRadius):
        self.currPose = initStatus[0], initStatus[1]
        self.yaw = initStatus[2]
        self.base = base
        self.wheelRadius = wheelRadius

    def move(self, vel_leftWheel, vel_rightWheel):
        vl, vr = vel_leftWheel, vel_rightWheel
        L, R = self.base, self.wheelRadius
        x, y, theta = self.currPose[0], self.currPose[1], self.yaw

        dx = R / 2 * (vr + vl) * cos(theta)
        dy = R / 2 * (vr + vl) * sin(theta)
        dtheta = R / L * (vr - vl)

        x += dx
        y += dy
        theta += dtheta

        self.currPose[0], self.currPose[1], self.yaw = x, y, theta
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
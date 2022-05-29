"""
/******************************************************************************
License: GNU General Public License v2.0
File name: pure_pursuit.py
Author: LiXianQiang     Date:2021/05/20      Version: 0.0.1
Description: PurePursuit车辆横向控制算法实现
Class List:
    Class PurePursuit
        __init__(self, k_gain, bodyLen, max_steerAngle): 控制器初始化
        controller(self, tarPoint, currPose, currVel): 车辆横向控制器
Function List:
History:
    <author>            <date>              <version>           <desc>
    LiXianQiang         2021/05/20          0.0.1       PurePursuit车辆横向控制算法实现
******************************************************************************/
"""

import math
import numpy as np


class PurePursuit:
    """PurePursuit车辆横向控制算法

    属于后轮反馈控制算法

    参考链接: https://zhuanlan.zhihu.com/p/48117381
             https://www.bilibili.com/video/BV1RE411n7C8
    """

    def __init__(self, k_gain, bodyLen, max_steerAngle):
        """控制器初始化

        Args:
            k_gain: float类, 增益系数，越大跟踪效果越差，但是更平滑，反之
            bodyLen: float类, 车身长度(body length)
            max_steerAngle: float类, 最大转向角, 弧度制
        """
        self.k_gain = k_gain
        self.bodyLen = bodyLen
        self.max_steerAngle = max_steerAngle

    def controller(self, tarPoint, currPose, currVel):
        """车辆横向控制器

        关于算法的实现原理与介绍, 请查阅参考资料

        需要注意的是: 算法设计是基于车辆运动学模型,并未考虑车辆侧滑等因素
        因此算法仅适用于低速的运动场景(假设车辆不会发生侧滑) 

        Args:
            tarPoint: list类 目标点位置(target point), 一般由轨迹规划算法提供
                tarPoint[0]: float类 目标点的 x 轴坐标
                tarPoint[1]: float类 目标点的 y 轴坐标
            currPose: list类 当前位姿(current Pose)，即：后轮中心位置以及车辆航行角
                currPose[0]: float类 后轮中心的 x 轴
                currPose[1]: float类 后轮中心的 y 轴
                currPose[2]: float类 车辆航向角, 即：车身与全局坐标系之间的夹角
                    有效数值范围 [pi, -pi)
            currVel: float类 当前速度(current velocity)，即：后轮中心的速度
            
        Returns:
            diffAngle: float类, 前轮转向角输出值(diffient angle)
        """

        alpha = math.atan2(tarPoint[1] - currPose[1], tarPoint[0] - currPose[0]) \
                - currPose[2]

        # 前瞻距离 look-ahead distance
        # 计算方式一般有两种: 一种是固定值来给定;另一种是根据目标点到车辆当前位置的距离
        # 来间接给出(其距离大小可以通过(输入的)纵向速度来反映）
        lookahead_distance = self.k_gain * currVel

        # 防止因前瞻距离过小导致输出的跟踪信号(转向角)发生震荡,
        # 加入 k_soft 来起到转向平滑的作用
        k_soft = 1.0

        # 横向控制控制律
        delta = math.atan2(2 * self.bodyLen * math.sin(alpha),
                               k_soft + lookahead_distance)

        # 车辆转向的物理约束
        if abs(delta) > self.max_steerAngle:
            diffAngle = np.sign(delta) * self.max_steerAngle
        else:
            diffAngle = delta
        return diffAngle
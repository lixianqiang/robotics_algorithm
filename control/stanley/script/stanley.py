"""
/******************************************************************************
License: GNU General Public License v2.0
File name: stanley.py
Author: LiXianQiang     Date:2021/05/11      Version: 0.0.1
Description: Stanley车辆横向控制算法实现
Class List:
    Class Stanley
        __init__(self, k_gain, bodyLen, max_steerAngle): 控制器初始化
        CalculateFrontWheelPoseWith(self, rearWheelPose):
        controller(self, refPose, currPose, currVel, reverse): 车辆横向控制器
Function List:
History:
    <author>            <date>              <version>           <desc>
    LiXianQiang         2021/05/11          0.0.1       Stanley车辆横向控制算法实现
******************************************************************************/
"""
import math
import numpy as np


class Stanley:
    """Stanley车辆横向控制算法

    属于前轮反馈控制算法

    参考链接: https://blog.csdn.net/renyushuai900/article/details/98460758
            https://blog.csdn.net/caokaifa/article/details/91483376
            https://www.bilibili.com/video/BV1rE411J7Gz

    参考论文: <<Autonomous automobile trajectory tracking for off-road driving:
            controller design, experimental validation and racing>>
    """

    def __init__(self, k_gain, bodyLen, max_steerAngle):
        """控制器初始化

        Args:
            k_gain: float类, 增益系数
            bodyLen: float类, 车身长度
            max_steerAngle: float类, 最大转向角, 弧度制
        """
        self.k_gain = k_gain
        self.bodyLen = bodyLen
        self.max_steerAngle = max_steerAngle

    def CalculateFrontWheelPoseWith(self, rearWheelPose):
        """通过后轮位置计算前轮位置

        Args:
            rearWheelPose: list类 后轮位姿 （包括:后轮中心位置以及车辆航行角）
                rearWheelPose[0]: float类 后轮中心的 x 轴
                rearWheelPose[1]: float类 后轮中心的 y 轴
                rearWheelPose[2]: float类 车辆航向角（即：车身与惯性坐标系所成的夹角）
                    有效数值范围 [pi, -pi)
        Returns:
            frontWheelPose: list类 前轮位姿 （包括:前轮中心位置以及车辆航行角）
                frontWheelPose[0]: float类 前轮中心的 x 轴
                frontWheelPose[1]: float类 前轮中心的 y 轴
                frontWheelPose[2]: float类 车辆航向角（即：车身与惯性坐标系所成的夹角）
                    有效数值范围 [pi, -pi)
        """
        frontWheelPose = []
        frontWheelPose[0] = rearWheelPose[0] + self.bodyLen * np.cos(rearWheelPose[2])
        frontWheelPose[1] = rearWheelPose[1] + self.bodyLen * np.sin(rearWheelPose[2])
        frontWheelPose[2] = rearWheelPose[2]
        return frontWheelPose

    def controller(self, refPose, currPose, currVel, reverse=False):
        """车辆横向控制器

        关于算法的实现原理与介绍, 请查阅参考资料

        需要强调一点:
        根据参考论文的描述, 算法是基于运动学模型而设计, 因此没有考虑车辆惯性的影响:
        轮胎的侧偏以及转向伺服执行器的时延, 因此仅适用低速场景下. 但论文中通过对干扰项
        进行补偿的方式实现, 也给出了适用于动力学模型的Stanley车辆横向控制器

        Args:
            refPose: list类 参考点位姿 关于参考点的选择: 一般选择轨迹中距离前轮中心
                最近的一点，该点也叫最近路径点
                refPos[0]: float类 参考点的 x 轴
                refPos[1]: float类 参考点的 y 轴
                refPos[2]: float类 参考点的切线角（即：切线与惯性坐标系所成的夹角)
                    有效数值范围 [pi, -pi)
            currPose: list类 当前位姿（即:前轮中心位置以及车辆航行角）
                currPose[0]: float类 前轮中心的 x 轴
                currPose[1]: float类 前轮中心的 y 轴
                currPose[2]: float类 车辆航向角（即：车身与惯性坐标系所成的夹角）
                    有效数值范围 [pi, -pi)
            currVel: float类 当前速度（即：前轮中心的速度）简化情况下,
                可以将其视作车辆质心位置速度/后轮速度
            reverse: bool类 车辆的行驶方向, 默认缺省值 False
                True 为 倒车行驶
                Flase 为 正向行驶

        Returns:
            diffAngle: float类, 前轮转向角输出值
        """

        def AngleDiff(endAngle, startAngle):
            """计算两个角之间的最小转向夹角

            从起始角 startAngle 指向终止角 endAngle，即：endAngle - startAngle,
            旋转方向的正负方向判断: 逆时针方向为正, 顺时针方向为负

            Args:
                endAngle: 终止角, 有效数值范围 [pi,-pi)
                startAngle: 起始角, 有效数值范围 [pi,-pi)

            Returns:
                diffAngle: 输出的转向角 有效数值范围 [pi,-pi)
            """
            deltaAngle = endAngle - startAngle
            abs_deltaAngle = math.fabs(deltaAngle)
            abs_compleAngle = 2 * math.pi - abs_deltaAngle

            # 当旋转夹角在数值上大于补角时, 选择沿着补角进行旋转
            if abs_compleAngle < abs_deltaAngle:
                # 当 deltaAngle < 0 表示起始角一开始是以顺时针方向旋转, 并旋转到终止角,
                # 若沿着补角一侧进行旋转, 起始角则是以逆时针方向旋转, 并旋转到终止角
                # 当 deltaAngle > 0 表示起始角一开始是以逆时针方向旋转, 并旋转到终止角,
                # 若沿着补角一侧进行旋转, 起始角则是以顺时针方向旋转, 并旋转到终止角
                # 那么, (带方向)旋转夹角大小 = 正负方向 * 数值大小(补角的绝对值)
                diffAngle = -1 * np.sign(deltaAngle) * abs_compleAngle
            else:
                diffAngle = deltaAngle
            return diffAngle

        currPose = np.array([currPose]).T
        refPose = np.array([refPose]).T
        # 构建垂直于车辆的法向量
        yaw = currPose[2]
        # 模长为1的法向量
        normVec = np.array([[1 * math.cos(yaw + math.pi / 2)], [1 * math.sin(yaw + math.pi / 2)]])
        # 通过将误差向量投影到法向量，可以得到带符号的横向误差，
        LateralError = normVec.T @ (refPose[:2] - currPose[:2])

        # 根据论文描述, 当速度很低的时候, arctan(k*e/vel) 中的 k*e/vel 会变得很大,
        # 这时候它对横向(跟踪)误差会变得特别敏感, 因此在分母中加入 k_soft 使得
        # 小车在速度很低时也能正常运作
        k_soft = 1

        # 航向误差：参考点的切线角与车辆航向角所成夹角
        headingError = AngleDiff(refPose[2], currPose[2])

        # TODO 这里的currVel应该是前轮速度，这里暂时使用后轮速度代替
        # 横向控制控制律
        delta = headingError + math.atan2(self.k_gain * LateralError, k_soft + currVel)

        # 车辆转向的物理约束
        if abs(delta) > self.max_steerAngle:
            diffAngle = np.sign(delta) * self.max_steerAngle
        else:
            diffAngle = delta

        return diffAngle
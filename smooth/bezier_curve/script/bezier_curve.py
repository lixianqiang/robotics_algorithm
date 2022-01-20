"""
/******************************************************************************
License: GNU General Public License v2.0
File name: bezier_curve.py
Author: LiXianQiang     Date:2021/06/02      Version: 0.0.1
Description: 贝塞尔曲线算法实现
Class List:
    Class BezierCurve
        __init__(self, ctrlPntList): 曲线初始化
        BezierCurve(self, t): 贝赛尔曲线的一般化形式
        DeCasteljau(self, t): 德卡斯特里奥算法（贝塞尔曲线的递归形式）
        DeCasteljau2(self, t): 德卡斯特里奥算法（贝塞尔曲线的非递归形式）
Function List:
    Coord2Vec(data): 将list中的tuple转成numpy.array
History:
    <author>            <date>              <version>           <desc>
    LiXianQiang         2021/06/02          0.0.1               贝塞尔曲线算法实现
******************************************************************************/
"""

from math import factorial
import numpy as np


class BezierCurve:
    """贝赛尔曲线类

    参考资料：
        https://zh.wikipedia.org/wiki/%E8%B2%9D%E8%8C%B2%E6%9B%B2%E7%B7%9A
        https://zh.wikipedia.org/wiki/%E5%BE%B7%E5%8D%A1%E6%96%AF%E7%89%B9%E9%87%8C%E5%A5%A5%E7%AE%97%E6%B3%95
    """
    def __init__(self, ctrlPntList):
        """曲线初始化

        ctlPntList: list类, 控制点列表(control point list),
            其列表元素: numpy.array类，控制点坐标, n×1维, 如: 3×1维
        """
        self.ctrlPntList = ctrlPntList

    def BezierCurve(self, t):
        """贝赛尔曲线的一般化形式

        Args:
            t: float类，时间戳, 取值范围：[0,1]

        Returns:
            sum(B): numpy.array类，t时刻所对应的位置，n×1维, 如: 3×1维
        """
        C = [None] * len(self.ctrlPntList) # 二项式系数
        B = [None] * len(self.ctrlPntList) # 贝塞尔多项式

        n = len(self.ctrlPntList) - 1
        for i, P in enumerate(self.ctrlPntList):  # i = 0,...,n
            C[i] = factorial(n) / (factorial(i) * factorial(n-i))
            B[i] = P * C[i] * (1 - t) ** (n - i) * t ** i
        return sum(B)

    def DeCasteljau(self, t):
        """德卡斯特里奥算法（贝塞尔曲线的递归形式）

        Args:
            t: float类，时间戳, 取值范围：[0,1]

        Returns:
            recuFunc(t, self.ctrlPntList): 递归函数，
                其返回值：numpy.array类，t时刻所对应的位置，n×1维, 如: 3×1维
        """
        def recuFunc(t, j, n, P):
            """递归函数(recursive function)

            Args:
                t: float类，时间戳, 取值范围：[0,1]
                j: int类, 递归的控制点的次数
                n: int类, len(self.ctrlPntList) - 1
                P: list类，递归的控制点列表, 其列表元素: numpy类，控制点位置，n×1维,
                    如: 3×1维

            Returns:
                recuFunc(t, next_P): 递归函数，其返回值：numpy.array类，
                    t时刻所对应的 j 阶控制点
                next_P[0]: numpy.array类，t 时刻所对应的位置，n×1维, 如: 3×1维
            """
            j += 1
            next_P = [None] * ((n-j) + 1)
            for i in range((n-j) + 1):  # i = 0,1,...,n-j
                next_P[i] = P[i] * (1 - t) + P[i+1] * t
            if j == n:
                return next_P[0]
            else:
                return recuFunc(t, j, n, next_P)
        return recuFunc(t, 0, len(self.ctrlPntList)-1, self.ctrlPntList)

    def DeCasteljau2(self, t):
        """德卡斯特里奥算法（贝塞尔曲线的非递归形式）
        Args:
            t: float类，时间戳, 取值范围：[0,1]
        Returns:
            point: numpy.array类，t时刻所对应的控制点位置，n×1维
        """
        n = len(self.ctrlPntList) - 1
        # 实现与MatLab类似的元胞数组
        P = np.empty((n+1, n+1), dtype=object)
        for i, ctrlPnt in enumerate(self.ctrlPntList):  # i = 0,1,...,n
            P[i, 0] = ctrlPnt

        for j in range(1, n + 1):  # j = 1,2,...,n
            for i in range(0, n - j + 1):  # i = 0,1,...,n-j
                P[i, j] = P[i, j-1] * (1 - t) + P[i+1, j-1] * t
        point = P[0, n]
        return point


def Coord2Vec(data: list):
    """将list中的tuple转成numpy.array"""
    changed_data = []
    for item in data:
        changed_data.append(np.array([item]).T)
    if len(changed_data) == 1:
        return changed_data.pop()
    else:
        return changed_data
"""
/******************************************************************************
License: GNU General Public License v2.0
File name: cubic_spline.py
Author: LiXianQiang     Date:2021/06/28      Version: 0.0.1
Description: 三次样条算法实现
Class List:
    Class CubicSpline
        __init__(self, interpolatePoints, boundaryType, boundaryValue): 参数初始化
        CubicSpline(self, t): 三次样条算法
        derivative(self, t, order): 关于样条函数的各阶导函数
        _GenarateContrainBy(self, boundaryType): 根据边界类型生成不同的边界函数
        _GenarateSplineFunction(self, index, knotList, coordList,
                                boundaryCondition): 生成样条函数及其导函数
        _FirstBoundary(self): 计算样条函数的第一类边界条件所对应的系数
        _SecondBoundary(self): 计算样条函数的第二类边界条件所对应的系数
        _ThirdBoundary(self): 计算样条函数的第三类边界条件所对应的系数
Function List:
    Lists2IntrPnts(knotList, coordList): 将节点列表与插值点列表转换成字典形式的插值点
History:
    <author>            <date>              <version>           <desc>
    LiXianQiang         2021/06/28          0.0.1               三次样条算法实现
******************************************************************************/
"""


import numpy as np


class CubicSpline:
    """三次样条类

    Attributes:
        knotList: list类，节点，一般表示为时间
        coordList: numpy.array类，n×1维，一般表示为空间坐标
        boundaryValue: list类，边界点的导数值，其列表元素：list类/tuple类，
            首末边界的一阶导/二阶导，维度大小需与插值点保持一致。其列表元素个数：2个
        splineObj: tuple类，样条对象，其元组元素：函数句柄，关于样条函数的各阶导函数

    Methods:
        __init__(self, interpolatePoints, boundaryType, boundaryValue): 参数初始化
        CubicSpline(self, t): 三次样条算法
        derivative(self, t, order): 关于样条函数的各阶导函数
        _GenarateContrainBy(self, boundaryType): 根据边界类型生成不同的边界函数
        _GenarateSplineFunction(self, index, knotList, coordList,
                                boundaryCondition): 生成样条函数及其导函数
        _FirstBoundary(self): 计算样条函数的第一类边界条件所对应的系数
        _SecondBoundary(self): 计算样条函数的第二类边界条件所对应的系数
        _ThirdBoundary(self): 计算样条函数的第三类边界条件所对应的系数

    参考资料:
        <<数值分析>, 第二讲, p49-p54. 潘建瑜.
        https://blog.csdn.net/zb1165048017/article/details/48311603
    """
    def __init__(self, interpolatePoints, boundaryType='natural',
                 boundaryValue=None):
        """参数初始化

        Args:
            interpolatePoints: dict类，插值点列表(interpolation point list),
                 key值：浮点数的string形式，节点，一般表示为时间
                 value值: tuple类/list类，插值点坐标，一般表示为空间坐标
            boundaryType: string类，边界约束类型，可选参数：'first', 'second',
                'third', 'natural', 'period'，默认缺省值: ‘natural’，
                其中，{'natural', 'third', 'period'} 均不需要为 boundaryValue 赋值
            boundaryValue: list类，边界点的导数值，默认缺省值: None,
                其列表元素：list类/tuple类，首末边界的一阶导/二阶导，维度大小需与
                    插值点保持一致其列表元素个数：2个
        """
        self.knotList = []
        self.coordList = []
        # 从插值点中分离出节点与对应的插值点坐标
        for i, (knot, coord) in enumerate(interpolatePoints.items()):
            self.knotList.append(float(knot))
            self.coordList.append(coord)
        self.coordList = np.array(self.coordList)

        self.boundaryValue = boundaryValue
        boundaryCondition = self._GenarateContrainBy(boundaryType)

        self.splineObj = [None] * (len(self.knotList) - 1)

        for i in range(len(self.knotList)-1):
            self.splineObj[i] = self._GenarateSplineFunction(i,
                                                             self.knotList,
                                                             self.coordList,
                                                             boundaryCondition)

    def CubicSpline(self, t):
        """三次样条算法

        Args:
            t: float/int类，节点，一般表示时间

        Returns:
            funcVal: numpy.array类, 节点t所对应的坐标，n×1维, 如:3×1维
        """
        # TODO 筛选区间的方法可以进一步优化
        for i in range(len(self.knotList)-1):
            if self.knotList[i] <= t <= self.knotList[i+1]:
                funcVal = self.splineObj[i][0](t)
                return funcVal

    # TODO 以后实现了自动求导算法再替换掉
    def derivative(self, t, order=1):
        """关于样条函数的各阶导函数

        Args:
            t: float/int类，节点，一般表示时间
            order: int类，求导阶数，默认缺省值：1

        Returns:
            funcVal: numpy.array类, 对应阶次导函数在节点t时的导数，n×1维, 如:3×1维
        """
        # TODO 筛选区间的方法可以进一步优化
        for i in range(len(self.knotList)-1):
            if self.knotList[i] <= t <= self.knotList[i+1]:
                if order == 0:
                    splineFunc = self.splineObj[i][0]
                elif order == 1:
                    splineFunc = self.splineObj[i][1]
                elif order == 2:
                    splineFunc = self.splineObj[i][2]

                # TODO 实现高阶导函数的自动求导
                else:
                    pass
                break
        funcVal = splineFunc(t)
        return funcVal

    def _GenarateContrainBy(self, boundaryType):
        """根据边界类型生成不同的边界函数

        Args:
            boundaryType: string类，边界类型，可选参数：'first', 'second',
                'third', 'natural', 'period'

        Returns:
            boundaryContrain: 函数句柄，边界约束，其函数返回值: numpy.array类，
                不同边界条件所对应的系数
        """
        referDim = len(self.coordList[0])  # 参考维度 refer dimension
        if self.boundaryValue is None or len(self.boundaryValue) == 0:
            hasBoundVal = False
        else:
            hasBoundVal = True
            judgeResult = []
            for i in range(len(self.boundaryValue)):
                if isinstance(self.boundaryValue[i], (int, float)):
                    judgeResult.append(1 == referDim)
                elif isinstance(self.boundaryValue[i], (tuple, list)):
                    judgeResult.append(len(self.boundaryValue[i]) == referDim)
                else:
                    raise TypeError("输入类型既不是1维，也不是n维，不符合要求")
            hasSameDim = all(judgeResult)

        if boundaryType is "first":
            if not hasBoundVal:
                raise TypeError("第一类边界需要给定首末节点的一阶导数")
            if len(self.boundaryValue) != 2:
                raise ValueError("给定的边界值个数不正确")
            if not hasSameDim:
                raise SyntaxError("边界导数的维度需要与插值点坐标保持一致")

            boundaryContrain = self._FirstBoundary

        elif boundaryType in ['second', 'natural']:
            if boundaryType is 'second':
                if not hasBoundVal:
                    raise TypeError("第二类边界需要给定首末节点的二阶导数")
                if len(self.boundaryValue) != 2:
                    raise ValueError("给定的边界值个数不正确")
                if not hasSameDim:
                    raise SyntaxError("边界导数的维度需要与插值点坐标保持一致")

                boundaryContrain = self._SecondBoundary

            elif boundaryType is 'natural':
                if not hasBoundVal:
                    self.boundaryValue = [[0] * referDim for i in range(2)]
                    hasBoundVal = True
                    hasSameDim = all([len(self.boundaryValue[i]) == referDim
                                      for i in range(len(self.boundaryValue))])
                if len(self.boundaryValue) != 2:
                    raise ValueError("给定的边界值个数不正确")
                if not hasSameDim:
                    raise SyntaxError("边界导数的维度需要与插值点坐标保持一致")

                satisfyBoundValReq = all([self.boundaryValue[i] == [0] * referDim
                                          for i in range(2)])
                if not satisfyBoundValReq:
                    raise ValueError("给定的首末节点的二阶导数必须为0")
                boundaryContrain = self._SecondBoundary

        elif boundaryType in ['third', 'period']:
            if (self.coordList[0, :] != self.coordList[-1, :]).any():
                raise ValueError("首末节点不相等, 不满足边界条件")
            if hasBoundVal:
                print("提醒: 第三类边界条件/周期样条均不需要设置首末节点的边界导数")
            boundaryContrain = self._ThirdBoundary

        def BoundaryCondition():
            """边界约束"""
            self.boundaryValue = np.array(self.boundaryValue)
            coeffList = boundaryContrain()
            return coeffList
        return BoundaryCondition

    def _GenarateSplineFunction(self, index, knotList, coordList,
                                boundaryCondition):
        """生成样条函数及其导函数
        根据给定的边界条件函数 boundaryCondition 生成第 index 条的样条函数及其导函数

        Args:
            index: int类, 待生成的样条函数的索引
            knotList: list类, 节点列表，其列表元素：float类，节点
            coordList: numpy.array类, 插值点坐标，m×n维,
                m：插值点坐标的数目，
                n：插值点坐标对应的维度。
                如: 5×3维,有5个3维插值点
            boundaryCondition: 函数句柄，边界约束，其函数返回值: numpy.array类，
                不同边界条件所对应的系数

        Returns:
            splineObj: tuple类，样条对象，其元组元素：函数句柄，关于样条函数的各阶导函数，
                其函数返回值: numpy.array类，各阶导函数在t时刻的导数
        """
        x = knotList
        f = coordList
        M = boundaryCondition()
        n = len(self.knotList) - 1

        h = [None] * n
        for i in range(0, n-1+1):  # i = 0,1,...,n-1
            h[i] = x[i + 1] - x[i]

        i = index
        def SplineFunc(t):
            """样条函数"""
            funcVal = ((x[i+1] - t) ** 3 * M[i] / (6 * h[i]))  \
                      + ((t - x[i]) ** 3 * M[i+1] / (6 * h[i]))  \
                      + ((f[i+1] - f[i]) / h[i]
                         - (M[i+1] - M[i]) * h[i] / 6) * t \
                      + f[i] - M[i] * h[i] ** 2 / 6 \
                      - ((f[i+1] - f[i]) / h[i]
                         - (M[i+1] - M[i]) * h[i] / 6) * x[i]
            return funcVal

        def Deriv_SplineFunc(t):
            """样条函数的一阶导函数"""
            funcVal = (x[i+1] - t) ** 2 * M[i] / (2 * h[i])\
                      + (t - x[i]) ** 2 * M[i+1] / (2 * h[i]) \
                      + ((f[i+1] - f[i]) / h[i]
                         - (M[i + 1] - M[i]) * h[i] / 6)
            return funcVal

        def DDeriv_SplineFunc(t):
            """样条函数的二阶导函数"""
            funcVal = (x[i+1] - t) * M[i] / h[i] \
                      + (t - x[i]) * M[i+1] / h[i]
            return funcVal

        splineObj = (SplineFunc, Deriv_SplineFunc, DDeriv_SplineFunc)
        return splineObj

    def _FirstBoundary(self):
        """计算样条函数的第一类边界条件所对应的系数

        Returns：
            M：numpy.array类，第一类边界条件所对应的系数，具体含义请参考：<<数值分析>,
                第二讲, p49-p54. 潘建瑜.
        """
        x = self.knotList
        f = self.coordList
        bv = self.boundaryValue
        n = np.size(f, 0) - 1

        h = [None] * n
        for i in range(0, n-1+1):  # i = 0,...,n-1
            h[i] = x[i+1] - x[i]

        u = [0] * (n+1)
        z = [0] * (n+1)
        for i in range(1, n-1+1):  # i = 1,...n-1
            u[i] = h[i - 1] / (h[i-1] + h[i])
            z[i] = 1 - u[i]

        d = np.empty(f.shape, dtype=object)
        d[0] = 6 / h[0] * ((f[1] - f[0]) / h[0] - bv[0])
        for i in range(1, n-1+1):  # i = 1,...,n-1
            d[i] = 6 / (h[i-1] + h[i]) * ((f[i+1] - f[i]) / h[i]
                                          - (f[i] - f[i-1]) / h[i-1])
        d[n] = 6 / h[n-1] * (bv[1] - (f[n] - f[n-1]) / h[n-1])

        mat = np.zeros((n + 1, n + 1))
        mat[0, 0:1+1] = [2, 1]
        # 从 第1行 到 第n-1行
        for i in range(1, n-1+1):  # i = 1,...,n-1
            mat[i, i-1:i+1+1] = np.array([[u[i], 2, z[i]]])
        mat[n, n-1:n+1] = [1, 2]

        M = np.linalg.inv(mat) @ d
        return M

    def _SecondBoundary(self):
        """计算样条函数的第二类边界条件所对应的系数

        Returns：
            M：numpy.array类，第二类边界条件所对应的系数，具体含义请参考：<<数值分析>,
                第二讲, p49-p54. 潘建瑜.
        """
        x = self.knotList
        f = self.coordList
        bv = self.boundaryValue
        n = np.size(f, 0) - 1

        h = [None] * n
        for i in range(0, n-1+1):  # i = 0,...,n-1
            h[i] = x[i+1] - x[i]

        u = [0] * (n + 1)
        z = [0] * (n + 1)
        for i in range(1, n-1+1):  # i = 1,...n-1
            u[i] = h[i-1] / (h[i-1] + h[i])
            z[i] = 1 - u[i]

        d = np.empty(f.shape, dtype=object)
        d[0] = bv[0]
        for i in range(1, n-1+1):  # i = 1,...,n-1
            d[i] = 6 / (h[i-1] + h[i]) * ((f[i+1] - f[i]) / h[i]
                                          - (f[i] - f[i-1]) / h[i-1])
        d[n] = bv[1]

        mat = np.zeros((n + 1, n + 1))
        mat[0, 0] = 1
        # 从 第1行 到 第n-1行
        for i in range(1, n-1+1):  # i = 1,...,n-1
            mat[i, i-1:i+1+1] = np.array([[u[i], 2, z[i]]])
        mat[n, n] = 1

        M = np.linalg.inv(mat) @ d
        return M

    def _ThirdBoundary(self):
        """计算样条函数的第三类边界条件所对应的系数

        Returns：
            M：numpy.array类，第三类边界条件所对应的系数，具体含义请参考：<<数值分析>,
                第二讲, p49-p54. 潘建瑜.
        """
        x = self.knotList
        f = self.coordList
        n = len(f) - 1

        h = [None] * n
        for i in range(0, n-1+1):  # i = 0,...,n-1
            h[i] = x[i+1] - x[i]

        u = [0] * (n + 1)
        z = [0] * (n + 1)
        u[0] = h[0] / (h[0] + h[n-1])
        z[0] = 1 - u[0]
        for i in range(1, n-1+1):  # i = 1,...,n-1
            u[i] = h[i-1] / (h[i-1] + h[i])
            z[i] = 1 - u[i]
        u[n] = u[0]
        z[n] = z[0]

        d = [None] * (n + 1)
        d[0] = 6 / (h[0] + h[n-1]) * ((f[1] - f[0]) / h[0]
                                        - (f[n] - f[n-1]) / h[n-1])
        for i in range(1, n):  # i = 1,2,...,n-1
            d[i] = 6 / (h[i-1] + h[i]) * ((f[i+1] - f[i]) / h[i]
                                            - (f[i] - f[i-1]) / h[i-1])
        d[n] = d[0]

        mat = np.zeros((n + 1, n + 1))
        mat[0, 0:1+1], mat[0, n-1] = np.array([2, u[0]]), z[0]
        # 从 第1行 到 第n-1行
        for i in range(1, n):  # i = 1,...,n-1
            mat[i, i-1:i+1+1] = np.array([[u[i], 2, z[i]]])
        mat[n, 1], mat[n, n-1:n+1] = u[n], np.array([z[n], 2]),
        M = np.linalg.inv(mat) @ d
        return M


def Lists2IntrPnts(knotList, coordList):
    """将节点列表与插值点列表转换成字典形式的插值点

    Args:
        knotList: list类，节点列表，其列表元素：float类，节点
        coordList: list类，插值点坐标列表，其列表元素：tuple类/list类，插值点坐标

    Returns:
        intrPnts：dict类，插值点坐标对(interpolation points)，
            对应的key值：string形式的浮点数，节点，一般表示为时间。
            对应的value值: tuple类/list类，插值点坐标，一般表示为空间坐标
    """
    if len(knotList) != len(coordList):
        raise IndexError("列表元素的数目不匹配, knotList: {knots}个, "
                         "coordList: {coords}个".format(knots=len(knotList),
                                                        coords=len(coordList)))

    intrPnts = dict()
    for (knot, coord) in zip(knotList, coordList):
        if isinstance(coord, (int, float)):
            coord = [coord]
        intrPnts["{knot}".format(knot=knot)] = coord
    return intrPnts
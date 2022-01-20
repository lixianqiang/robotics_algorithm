"""
/******************************************************************************
License: GNU General Public License v2.0
File name: b_spline.py
Author: LiXianQiang     Date:2021/06/15      Version: 0.0.1
Description: B样条曲线算法实现
Class List:
    BSpline:
        __init__(self, knotList, ctrlPntList, degree, curveType): 参数初始化
        CoxdeBoor1(self, t): Cox-de Boor算法——递归形式
        CoxdeBoor2(self, t): Cox-de Boor算法——非递归形式
        _preprocess(self, spline, knotList, ctrlPntList, degree): 数据预处理
        _clamped_B_Spline(self, knotList, ctrlPntList, degree): clamped B样条
        _open_B_Spline(self, knotList, ctrlPntList, degree): open B样条
        _closed_B_Spline(self, knotList, ctrlPntList, degree): closed B样条
        StraightAt(self, knotIdxIntrvlList): 强制曲线段变成直线段
        PassThrough(self, ctrlPntIdxList): 强制B-样条曲线经过一个控制点
        TangentTo(self, ctrlPntIdxIntrvlList): 强制B-样条曲线与控制折线的一边相切

    bfloat:
        tofloat(self): bfloat转float
        __new__(cls, value): 生成bfloat对象
        __sub__(self, other): self - other
        __truediv__(self, other): self / other
         __rsub__(self, lhs): other - self
        __rtruediv__(self, lhs): other / self
Function List:
    Coord2Vec(data): 将list中的tuple转成numpy.array
    IsNonDecrease(seq): 判断seq是否为非递减序列
    IsNonUniform(seq): 判断seq是否为非均匀序列
History:
    <author>            <date>              <version>           <desc>
    LiXianQiang         2021/06/15          0.0.1               B样条曲线算法实现
******************************************************************************/
"""

import numpy as np


class BSpline:
    """B样条曲线类

    Attributes:
        knotList: list类，节点列表，其列表元素：float类，节点
        ctrlPntList: list类，控制点列表，其列表元素：numpy.array类，控制点坐标，
            n×1维, 如:3×1维
        degree: int类，曲线的次数
        validDomain: list类，有效定义域, 是 knotList 从第 j 个到第 m-j 个的切片
            其列表元素：float类，节点

    Methods:
        __init__(self, knotList, ctrlPntList, degree, curveType='open'):
            参数初始化
        CoxdeBoor1(self, t): Cox-de Boor算法——递归形式
        CoxdeBoor2(self, t): Cox-de Boor算法——非递归形式
        _preprocess(self, spline, knotList, ctrlPntList, degree): 数据预处理
        _clamped_B_Spline(self, knotList, ctrlPntList, degree): clamped B样条
        _open_B_Spline(self, knotList, ctrlPntList, degree): open B样条
        _closed_B_Spline(self, knotList, ctrlPntList, degree): closed B样条
        StraightAt(self, knotIdxIntrvlList): 强制曲线段变成直线段
        PassThrough(self, ctrlPntIdxList): 强制B-样条曲线经过一个控制点
        TangentTo(self, ctrlPntIdxIntrvlList): 强制B-样条曲线与控制折线的一边相切

    参考资料:
        https://blog.csdn.net/tuqu/article/details/5366701
    """
    def __init__(self, knotList, ctrlPntList, degree, curveType='clamped'):
        """参数初始化

        Args:
            knotList: list类，节点列表，其列表元素：float类，节点
            ctrlPntList: list类，控制点列表，其列表元素：numpy.array类，
                控制点坐标，n×1维, 如:3×1维
            degree: int类，待构造曲线的次数
            curveType: string类，待构造的曲线类型，默认缺省值:clamped
                可选参数：'clamped', 'open', 'closed'
        """
        m = len(knotList) - 1
        n = len(ctrlPntList) - 1
        j = degree
        if m != n + j + 1:
            raise ValueError("设节点数m+1个，控制点数n+1个，曲线次数j。"
                             "那么，输入参数需满足以下条件：m = n + j + 1，"
                             "而现在 m:{m}, n:{n}, j:{j}".format(m=m, n=n, j=j))
        elif not IsNonDecrease(knotList):
            raise Exception("节点列表 knotList 要求必须是非递减序列")
        else:
            curve = {"clamped": self._clamped_B_Spline,
                     "open": self._open_B_Spline,
                     "closed": self._closed_B_Spline}
            spline = curve.get(curveType)
            if spline is None:
                raise ValueError("曲线类型: '{type}' 不存在".format(type=curveType))
            # 数据预处理
            self._preprocess(spline, knotList, ctrlPntList, degree)

    def CoxdeBoor1(self, t):
        """Cox-de Boor算法——递归形式

        Args:
            t: float类，节点 取值范围: 在 self.validDomain 的范围内
            注意：不接受 Python 以外的 float 类型，如：numpy.float, numpy.float64等

        Returns:
            sum(...): numpy.array类，节点 t 对应的位置, n×1维, 如: 3×1维
            recuFunc(...): 函数句柄, 递归函数
        """
        k = 0  # 当前基函数的次数
        j = self.degree
        T = self.knotList
        P = self.ctrlPntList
        m = len(self.knotList) - 1
        B = [None] * m
        for i in range(m):  # i = 0,1,...,m-1
            if T[i] <= t < T[i + 1]:
                B[i] = 1
            else:
                B[i] = 0

        def recuFunc(t, k, m, j, T, B, P):
            """递归函数(recursive function)

            Args:
                t: float类，节点 取值范围: 在 self.validDomain 的范围内
                (不接受 Python 以外的 float 类型，如：numpy.float, numpy.float64,...)
                k: int类, 基函数的次数
                m: int类, 与节点数有关，m+1个节点
                j: int类, 待构造曲线的次数
                T: list类, 节点列表，其列表元素：float类，节点
                B: list类, 基函数列表，其列表元素：bfloat类, k次的基函数
                P: list类, 控制点列表,其列表元素: numpy.array类，
                    控制点坐标，n×1维, 如:3×1维

            Returns:
                sum(...): numpy.array类，节点 t 对应的位置, n×1维, 如: 3×1维,
                recuFunc(...): 函数句柄, 递归函数
            """
            k += 1
            next_B = [None] * ((m - 1 - k) + 1)
            for i in range((m - 1 - k) + 1):  # i = 0,1,...(m-1)-k
                next_B[i] = (t - T[i]) / (T[i + k] - T[i]) * B[i] \
                            + (T[i + k + 1] - t) / (T[i + k + 1] - T[i + 1]) \
                            * B[i + 1]
            if k == j:
                return sum([next_B[i] * P[i] for i in range(len(next_B))])
            else:
                return recuFunc(t, k, m, j, T, next_B, P)

        # 特殊地,要求输出的曲线为0次B样条曲线
        if k == j:
            return sum([B[i] * P[i] for i in range(len(B))])
        else:
            return recuFunc(t, k, m, j, T, B, P)

    def CoxdeBoor2(self, t):
        """Cox-de Boor算法——非递归形式

        Args:
            t: float类，节点 取值范围: 在 self.validDomain 的范围内
            注意：不接受 Python 以外的 float 类型，如：numpy.float, numpy.float64等

        Returns:
            sum(...): numpy.array类，节点 t 对应的位置, n×1维, 如: 3×1维
        """
        j = self.degree
        T = self.knotList
        m = len(self.knotList) - 1

        # 实现与MatLab类似的元胞数组
        B = np.empty(((m - 1) + 1, j + 1), dtype=object)
        for i in range((m - 1) + 1):  # i = 0,1,...,m-1
            if T[i] <= t < T[i + 1]:
                B[i, 0] = 1
            else:
                B[i, 0] = 0

        for k in range(1, j + 1):  # k = 1,2,...,j
            for i in range(0, (m - 1 - k) + 1):  # i = 0,1,...,(m-1)-k
                B[i, k] = (t - T[i]) / (T[i + k] - T[i]) * B[i, k - 1] \
                          + (T[i + k + 1] - t) / (T[i + k + 1] - T[i + 1]) \
                          * B[i + 1, k - 1]

        P = self.ctrlPntList
        n = len(self.ctrlPntList) - 1
        return sum([(P[i] * B[i, j]) for i in range(0, n + 1)])

    def _preprocess(self, spline, knotList, ctrlPntList, degree):
        """数据预处理

        1, 改变节点类型(从 float 转 bfloat)
        2, 根据曲线类型修改控制点,节点或阶数

        Args:
            spline: 函数句柄, 根据曲线类型修改控制点,节点或曲线次数
                其函数输入参数: knotList, ctrlPntList, degree
            knotList: list类, 节点列表, 其列表元素：float类，节点
            ctrlPntList: list类，控制点列表，其列表元素：numpy.array类，
                控制点坐标，n×1维, 如:3×1维
            degree: int类，待构造曲线的次数
        """
        prep_knotList = [None] * len(knotList)

        # 将列表元素从 float类 转成 bfloat类
        for i in range(len(knotList)):
            prep_knotList[i] = bfloat(knotList[i])

        self.knotList, self.ctrlPntList, self.degree \
            = spline(prep_knotList, ctrlPntList, degree)

    def _clamped_B_Spline(self, knotList, ctrlPntList, degree):
        """clamped B样条"""
        j = degree
        m = len(knotList) - 1
        T = knotList

        # 令首末节点 T[0], T[m] 的重复度为 j+1
        # 即：T[0] = T[1] =,...= T[j] 以及 T[m] = T[m-1] =,...,= T[m-j]
        for i in range(j):
            T[i + 1] = T[i]
            T[m - 1 - i] = T[m]

        # 有效定义域：[knotList[j], knotList[m-j])
        self.validDomain = knotList[j:m - j + 1]
        return knotList, ctrlPntList, degree

    def _open_B_Spline(self, knotList, ctrlPntList, degree):
        """open B样条"""
        j = degree
        m = len(knotList) - 1
        # 有效定义域：(knotList[j], knotList[m-j])
        self.validDomain = knotList[j:m - j + 1]
        return knotList, ctrlPntList, degree

    def _closed_B_Spline(self, knotList, ctrlPntList, degree):
        """closed B样条
        通过 Warpping 控制点的方式来构建闭曲线
        """
        j = degree
        P = ctrlPntList
        m = len(knotList) - 1
        n = len(ctrlPntList) - 1

        if IsNonUniform(knotList):
            print("警告：当给定的节点序列是非均匀的，"
                  "所构造处的“closed B样条曲线”是不能闭合的")

        # 将 n+1 个控制点 P0, P1, ...,Pn 的前 j 个控制点和后 j 个控制点分别依次相等
        # 即: P[0]=P[n-j+1], P[1]=P[n-j+2],...,P[j-2]=P[n-1], P[j-1]=P[n]
        for i in range(j):  # i = 0,1,...,j
            P[n - j + 1 + i] = P[i]

        # 有效定义域：[knotList[j], knotList[m-j])
        self.validDomain = knotList[j:m - j + 1]
        return knotList, ctrlPntList, degree

    def StraightAt(self, knotIdxIntrvlList):
        """强制曲线段变成直线段
        令节点区间内的曲线段变成直线

        Args：
            knotIdxIntrvlList: list类, 节点索引区间的列表(knot index interval‘s list)，
                其列表元素: tuple类, 由节点索引组成的区间 (knot index interval)
        """
        j = self.degree
        P = self.ctrlPntList
        for knotIdxIntrvl in knotIdxIntrvlList:
            i = knotIdxIntrvl[0]
            k = knotIdxIntrvl[-1]
            if not set(self.knotList[i:k+1]).issubset(set(self.validDomain)):
                raise ValueError("区间 {interval} 不在有效定义域的范围内"
                                 .format(interval=self.knotList[i:k+1]))

            dsList = []
            # 计算两两控制点之间的距离
            for n in range(i-j, k-1):
                ds = np.linalg.norm(P[n+1] - P[n])
                dsList.append(ds)

            scale = 0
            dS = sum(dsList)
            # 让 P[i]的前j+1个控制点(含P[i])共线
            for n, ds in enumerate(dsList):
                scale += ds / dS
                P[i-j+1+n] = P[i-j] + (P[k-1] - P[i-j]) * scale

    def PassThrough(self, ctrlPntIdxList):
        """强制B-样条曲线经过一个控制点
        生成的曲线将强制经过指定索引中所对应的控制点

        需要注意的是：索引列表不要求非递减序列，但是索引顺序会影响实际处理结果
        比如：[1,2,3] 与 [3,2,1] 处理完后显示的轨迹可能完全不同的

        Args:
            ctrlPntIdxList: list类，控制点索引列表(control point's index list)，
                其列表元素：int类, 控制点的索引值
                取值范围：在 self.ctrlPntList 的范围内
        """
        j = self.degree
        P = self.ctrlPntList
        for i in ctrlPntIdxList:
            if i > len(self.ctrlPntList) - 1:
                raise IndexError("索引 %d 不在索引范围" % i)
            # 让P[idx]的前j个控制点(含P[idx])重合
            for k in range(1, j):  # k = 1,2,...,j-1
                P[i-k] = P[i]

    def TangentTo(self, ctrlPntIdxIntrvList):
        """强制B-样条曲线与控制折线的一边相切
        指定首末节点索引，生成的曲线将相切于该首末节点的切线

        Args:
            ctrlPntIdxIntrvList: list类，控制点索引区间列表(list of control point's index interval)
                其列表元素: tuple类，由相应控制点的索引所组成的区间
                取值范围：在 self.ctrlPntList 的范围内，区间的前后索引之间需满足递增关系且索引差值大于1，即：[i,i+n],n=2,3,...
        """
        P = self.ctrlPntList
        for ctrlPntIntrvl in ctrlPntIdxIntrvList:
            i = ctrlPntIntrvl[0]
            k = ctrlPntIntrvl[-1]
            if k - i <= 1:
                raise Exception("[P[{i}], P[{k}]] 之间没有中间控制点"
                                .format(i=ctrlPntIntrvl[0],
                                        k=ctrlPntIntrvl[-1]))
            midPnt = P[i] + 0.5 * (P[k] - P[i])
            for n in range(i+1, k):
                P[n] = midPnt


class bfloat(float):

    def tofloat(self):
        """bfloat 转 float"""
        return self.__float__()

    def __new__(cls, value):
        """生成bfloat对象"""
        return super().__new__(cls, value)

    def __sub__(self, other):
        """self - other"""
        return bfloat(super().__sub__(other))

    def __truediv__(self, other):
        """self / other"""
        try:
            return bfloat(super().__truediv__(other))
        except ZeroDivisionError:
            return bfloat(0)

    def __rsub__(self, lhs):
        """other - self
        lhs: left hand side (左值)"""
        return bfloat(super().__rsub__(lhs))

    def __rtruediv__(self, lhs):
        """other / self
        lhs: left hand side (左值)"""
        try:
            return bfloat(super().__rtruediv__(lhs))
        except ZeroDivisionError:
            return bfloat(0)


def Coord2Vec(data: list):
    """将list中的tuple转成numpy.array"""
    changed_data = []
    for item in data:
        changed_data.append(np.array([item]).T)
    if len(changed_data) == 1:
        return changed_data.pop()
    else:
        return changed_data


def IsNonDecrease(seq: list):
    """判断seq是否为非递减序列
    非递减序列： t0 <= t1 <= t2 <= t3,...,tn-1 <= tn

    Args:
        seq: list类，待检查序列，列表元素: float类，

    Returns:
        True: seq 为非递减序列
        False: 反之

    参考资料:
        https://qastack.cn/programming/4983258/python-how-to-check-list-monotonicity
        https://www.zhihu.com/question/52742082
    """
    return all(x <= y for x, y in zip(seq, seq[1:]))


def IsNonUniform(seq: list):
    """判断seq是否为非均匀序列

    Args:
        seq: list类，待检查序列，列表元素: float类，

    Returns:
        True: seq 为非均匀序列
        False: 反之
    """
    error = 10e-10
    refVal = seq[1] - seq[0]
    for i in range(1, len(seq)):
        if (seq[i] - seq[i - 1]) - refVal > error:
            return True
    return False
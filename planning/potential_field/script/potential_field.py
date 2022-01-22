"""
/******************************************************************************
License: GNU General Public License v2.0
File name: potential_filed.py
Author: LiXianQiang     Date:2021/07/23      Version: 0.0.1
Description: 人工势场算法实现
Class List:
    QuadraticPotential(PotentialFunction):
        __init__(self, coff=10): 参数初始化
        func(self, currPos, goalPos): 计算二次势场函数在 currPos 处所对应的函数值
        grad_func(self, currPos, goalPos): 计算二次势场函数在 currPos 处所对应的梯度

    ConePotential(PotentialFunction):
        __init__(self, coff=10): 参数初始化
        func(self, currPos, centPos): 计算锥势场函数在 currPos 处所对应的函数值
        grad_func(self, currPos, centPos): 计算锥势场函数在 currPos 处所对应的梯度

    CylindricalPotential(PotentialFunction):
        __init__(self, distThrsh, coff=100): 参数初始化
        func(self, currPos, centPos): 计算柱形势场函数在 currPos 处所对应的函数值
        grad_func(self, currPos, centPos): 计算柱形势场函数在 currPos 处所对应的梯度

    ZeroPotential(PotentialFunction):
        func(self, currPos, centPos): 计算零势场函数在 currPos 处所对应的函数值
        grad_func(self, currPos, centPos): 计算零势场函数在 currPos 处所对应的梯度

    AttractivePotential:
        __init__(self): 参数初始化
        func(self, currPos, goalPos): 在 goalPos 的作用力下，引力场函数在 currPos
            处的所对应的函数值
        grad_func(self, currPos, goalPos): 在 goalPos 的作用力下，在 currPos
            处的所产生的梯度

    RepulsionPotential:
        __init__(self): 参数初始化
        func(self, currPos, obsPos): 在 obsPos 的作用下，斥力场函数在 currPos
            处所对应的函数值
        grad_func(self, currPos, obsPos): 在 obsPos 的作用下，斥力场函数在 currPos
            处所对应的梯度

    PotentialField:
        __init__(self): 初始化引力势场对象和斥力势场对象
        __call__(self, goalPos, obsPosList): 更新目标位置和障碍物位置
        func(self, currPos): 人工势场函数在 currPos 处所对应的函数值
        grad_func(self, currPos): 人工势场函数在 currPos 处所对应的梯度

Function List:
History:
    <author>            <date>              <version>           <desc>
    LiXianQiang         2021/07/23            0.0.1          人工势场算法实现
******************************************************************************/
"""
import numpy as np
from design_patterns import Context
from design_patterns import IStrategy as PotentialFunction


class QuadraticPotential(PotentialFunction):

    def __init__(self, coff=10):
        """参数初始化

        Args:
            _coff: float类，二次势场函数的系数，默认缺省值: 10
        """
        self._coff = coff

    def func(self, currPos, goalPos):
        """计算二次势场函数在 currPos 处所对应的函数值
        Args：
            currPos：numpy.array类，2×1矩阵 当前位置
        Returns:
            funVal： float类，二次势场函数在 currPos 所对应的函数值
        """
        funcVal = 0.5 * self._coff * \
                  np.linalg.norm(currPos - goalPos, ord=2) ** 2
        return funcVal

    def grad_func(self, currPos, goalPos):
        """计算二次势场函数在 currPos 处所对应的梯度
        Args：
            currPos：numpy.array类，2×1矩阵 当前位置
        Returns:
            gradVal： numpy.array类，2×1矩阵 二次势场函数在 currPos 所对应的梯度
        """
        gradVal = self._coff * (currPos - goalPos)
        return gradVal


class ConePotential(PotentialFunction):
    """锥势场类"""

    def __init__(self, coff=10):
        """参数初始化

        Args:
            _coff: float类，锥势场函数的系数，默认缺省值: 10
        """
        self._coff = coff

    def func(self, currPos, centPos):
        """计算锥势场函数在 currPos 处所对应的函数值
        Args：
            currPos：numpy.array类，2×1矩阵 当前位置
        Returns:
            funVal： float类，2×1矩阵 锥势场函数在 currPos 所对应的函数值
        """
        funcVal = self._coff * np.linalg.norm(currPos - centPos, ord=2)
        return funcVal

    def grad_func(self, currPos, centPos):
        """计算锥势场函数在 currPos 处所对应的梯度
        Args：
            currPos：numpy.array类，2×1矩阵 当前位置
        Returns:
            gradVal： numpy.array类，2×1矩阵 锥势场函数在 currPos 所对应的梯度
        """
        # 表示 currPos 到 centPos 之间的欧几里得距离(euclidean distance)
        euclDist = np.linalg.norm(currPos - centPos, ord=2)
        gradVal = self._coff / euclDist * (currPos - centPos)
        return gradVal


class CylindricalPotential(PotentialFunction):
    """柱形势场类"""

    def __init__(self, distThrsh, coff=100):
        """参数初始化

        Args:
            _distThrsh：flaot类，距离阈值(distance threshold)
            _coff: float类，柱形势场函数的系数，默认缺省值: 100
        """
        self._distThrsh = distThrsh
        self._coff = coff

    def func(self, currPos, centPos):
        """计算柱形势场函数在 currPos 处所对应的函数值
        Args：
            currPos：numpy.array类，2×1矩阵 当前位置
        Returns:
            funVal： float类，柱形势场函数在 currPos 所对应的函数值
        """
        # 表示 currPos 到 centPos 之间的欧几里得距离(euclidean distance)
        euclDist = np.linalg.norm(currPos - centPos, ord=2)
        funcVal = 0.5 * self._coff * ((1 / euclDist - 1 / self._distThrsh) ** 2)
        return funcVal

    def grad_func(self, currPos, centPos):
        """计算柱形势场函数在 currPos 处所对应的梯度
        Args：
            currPos：numpy.array类，2×1矩阵 当前位置
        Returns:
            gradVal： numpy.array类，2×1矩阵 柱形势场函数在 currPos 所对应的梯度
        """
        # 表示 currPos 到 centPos 之间的欧几里得距离(euclidean distance)
        euclDist = np.linalg.norm(currPos - centPos, ord=2)
        gradVal = self._coff * (1 / self._distThrsh - 1 / euclDist) \
                  * ((1 / euclDist) ** 2) * ((currPos - centPos) / euclDist)
        return gradVal


class ZeroPotential(PotentialFunction):
    """零势场类"""

    def func(self, currPos, centPos):
        """计算零势场函数在 currPos 处所对应的函数值
        Args：
            currPos：numpy.array类，2×1矩阵 当前位置
        Returns:
            funVal： float类，零势场函数在 currPos所对应的函数值
        """
        zeroVector = np.zeros(currPos.shape)
        funVal = np.linalg.norm(zeroVector, ord=2)
        return funVal

    def grad_func(self, currPos, centPos):
        """计算零势场函数在 currPos 处所对应的梯度
        Args：
            currPos：numpy.array类，2×1矩阵 当前位置
        Returns:
            gradVal： numpy.array类，2×1矩阵 柱形势场函数在 currPos 所对应的梯度
        """
        gradVal = np.zeros(currPos.shape)
        return gradVal


class AttractivePotential:
    """引力势场类

    Attributes:
        _d_goal: float类，切换引力势场函数的距离阈值，取值范围: (0, inf]，单位: m (米)
        context: Context类， 上下文管理器，用于不同势场函数的切换
        quadPot：PotentialFunction类，二次势场函数
        conePot：PotentialFunction类，锥势场函数
    Methods:
        __init__(self, attrCoff, d_goal): 参数初始化
        func(self, currPos, obsPos): 在 goalPos 的作用下，引力场函数在 currPos
            处所对应的函数值
        grad_func(self, currPos, obsPos): 在 obsPos 的作用下，引力场函数在 currPos
            处所对应的梯度
    """
    def __init__(self):
        """参数初始化"""
        self._d_goal = 10
        self.context = Context()
        self.quadPot = QuadraticPotential()
        self.conePot = ConePotential()

    def func(self, currPos, goalPos):
        """在 goalPos 的作用力下，引力场函数在 currPos 处的所对应的函数值

        不同形式的引力势场函数最终计算得到的引力大小是不一样。其中，常见的引力场函数有：
            线性场函数: 锥势场函数
            二次场函数: 二次势场函数
            联合势场函数: 由锥势场函数与二次势场函数组成的分段函数

        目前所采用的是联合势场函数，即：远距离时使用线性势场函数，近距离时使用二次势场函数

        Args:
            currPos: numpy.array类，2×1矩阵，当前位置 q = [x, y]'
            goalPos: numpy.array类，2×1矩阵，目标位置 q_goal = [x1, y1]'

        Returns:
            funVal: numpy.array类，引力势场函数在 currPos 所对应的函数值
        """

        # 不同的距离采用不同的势场函数
        if np.linalg.norm(currPos - goalPos, ord=2) < self._d_goal:
            self.context.ChangeStrategy(self.quadPot)
        else:
            self.context.ChangeStrategy(self.conePot)
        funVal = self.context.func(currPos, goalPos)
        return funVal

    def grad_func(self, currPos, goalPos):
        """在 goalPos 的作用力下，在 currPos 处的所产生的梯度

        距离较远时，为了防止速度增长过快，使用线性势场的负梯度作为引力的表征
        距离较近时，防止引力减少导致速度过低，使用二次势场的负梯度作为引力表征

        Args:
            currPos: numpy.array类，2×1矩阵，当前位置 q = [x, y]'
            goalPos: numpy.array类，2×1矩阵，目标位置 q_goal = [x1, y1]'

        Returns:
            attraction: numpy.array类，引力，由引力场函数的负梯度所表征
        """

        # 不同的距离采用不同的势场梯度函数
        if np.linalg.norm(currPos - goalPos, ord=2) < self._d_goal:
            self.context.ChangeStrategy(self.quadPot)
        else:
            self.context.ChangeStrategy(self.conePot)
        gradVal = self.context.grad_func(currPos, goalPos)
        return gradVal


class RepulsionPotential:
    """斥力势场类

    Attributes:
        _d_obs: float类，切换斥力势场函数的距离阈值，取值范围: (0, inf]，单位: m (米)
        context: Context类， 上下文管理器，用于不同势场函数的切换
        cylinPot：PotentialFunction类，柱形势场函数
        zeroPot：PotentialFunction类，零势场函数
    Methods:
        __init__(self, attrCoff, d_goal): 参数初始化
        func(self, currPos, obsPos): 在 obsPos 的作用下，斥力场函数在 currPos
            处所对应的函数值
        grad_func(self, currPos, obsPos): 在 obsPos 的作用下，斥力场函数在 currPos
            处所对应的梯度
    """

    def __init__(self):
        """参数初始化"""
        self._d_obs = 10
        self.context = Context()
        self.cylinPot = CylindricalPotential(self._d_obs)
        self.zeroPot = ZeroPotential()

    def func(self, currPos, obsPos):
        """在 obsPos 的作用下，斥力场函数在 currPos 处所对应的函数值

        一般情况下，距离较近时使用柱形势场函数，距离较远时，势场函数为零

        Args:
            currPos: numpy.array类，2×1矩阵，当前位置
            obsPos: numpy.array类，2×1矩阵，障碍物位置

        Returns:
            funVal: numpy.array类，2×1矩阵，斥力势场函数在 currPos 所对应的函数值
        """

        # 不同的距离采用不同的势场函数
        if np.linalg.norm(currPos - obsPos, ord=2) < self._d_obs:
            self.context.ChangeStrategy(self.cylinPot)
        else:
            self.context.ChangeStrategy(self.zeroPot)
        funVal = self.context.func(currPos, obsPos)
        return funVal

    def grad_func(self, currPos, obsPos):
        """在 obsPos 的作用下，斥力场函数在 currPos 处所对应的梯度

        一般情况下，距离较近时使用柱形势场函数，距离较远时，势场函数为零

        Args:
            currPos: numpy.array类，2×1矩阵，当前位置
            obsPos: numpy.array类，2×1矩阵，障碍物位置

        Returns:
            gradVal: numpy.array类，2×1矩阵，斥力势场函数在 currPos 处所对应的梯度
        """

        # 不同的距离采用不同的势场梯度函数
        if np.linalg.norm(currPos - obsPos, ord=2) < self._d_obs:
            self.context.ChangeStrategy(self.cylinPot)
        else:
            self.context.ChangeStrategy(self.zeroPot)
        gradVal = self.context.grad_func(currPos, obsPos)
        return gradVal


class PotentialField:
    """人工势场算法

        算法思路：该算法将目标点和障碍物分别对机器人产生引力和斥力，使机器人沿
        合力方向运动。其中，引力与斥力用势场函数的负梯度来表征，其合力即为负梯度之和；
        算法的本质：对势场函数使用梯度下降法，其输出值是一系列局部坐标信息,
        可以作为控制器的参考信号用于运动控制

        注意事项：算法更多适用于具有差分轮结构或独轮结构等可以化简为质心模型的轮式机器人

        参考资料: https://www.cs.cmu.edu/~motionplanning/lecture/Chap4-Potential-Field_howie.pdf
                https://cs.gmu.edu/~kosecka/cs685/cs685-potential-fields.pdf
                https://www.bilibili.com/video/BV1WA411p7xe?t=8
                https://zhuanlan.zhihu.com/p/144816424

        Attributes:
            attPot: AttractivePotential，引力势场对象
            repPot: RepulsionPotential，斥力势场对象
        Methods:
            __init__(self): 初始化引力势场对象和斥力势场对象
            __call__(self, goalPos, obsPosList): 更新目标位置和障碍物位置
            func(self, currPos): 人工势场函数在 currPos 处所对应的函数值
            grad_func(self, currPos): 人工势场函数在 currPos 处所对应的梯度
    """
    def __init__(self):
        """初始化引力势场对象和斥力势场对象"""
        self.attPot = AttractivePotential()
        self.repPot = RepulsionPotential()

    def __call__(self, goalPos=None, obsPosList=None):
        """更新目标位置和障碍物位置

        Args:
            goalPos: numpy.array类，2×1矩阵，目标位置,默认缺省值: None
            obsPosList: list类, 障碍物位置, 默认缺省值: None
        """
        self.goalPos = goalPos
        self.obsPosList = obsPosList

    def func(self, currPos):
        """人工势场函数在 currPos 处所对应的函数值
        计算当前位置在人工势场函数所对应的函数值

        Args:
            currPos: numpy.array类，2×1矩阵，当前位置
        Returns:
            valSum: float类，当前位置在人工势场函数所对应的函数值
        """
        if self.goalPos is None:
            self.goalPos = currPos
        if self.obsPosList is None:
            self.obsPosList = []

        valRep = 0
        # 累计各个障碍物对当前位置所对应的函数值之和
        for obsPos in self.obsPosList:
            valRep += self.repPot.func(currPos, obsPos)
        # 目标位置在当前位置所对应的函数值
        valAtt = self.attPot.func(currPos, self.goalPos)
        valSum = valRep + valAtt
        return valSum

    def grad_func(self, currPos):
        """人工势场函数在 currPos 处所对应的梯度

        计算当前位置在人工势场函数所对应的梯度

        Args:
            currPos: numpy.array类，当前位置
        Returns:
            valSum: numpy.array，当前位置在人工势场函数所对应的梯度
        """
        if self.goalPos is None:
            self.goalPos = currPos
        if self.obsPosList is None:
            self.obsPosList = []

        gradRep = 0
        # 累计各个障碍物对当前位置所对应的梯度和
        for obsPos in self.obsPosList:
            gradRep += self.repPot.grad_func(currPos, obsPos)
        # 目标位置在当前位置所对应的梯度
        gradAtt = self.attPot.grad_func(currPos, self.goalPos)
        gradSum = gradRep + gradAtt
        return gradSum
"""
/******************************************************************************
License: GNU General Public License v2.0
File name: gradient_descent.py
Author: LiXianQiang     Date:2021/07/15      Version: 0.0.1
Description: 梯度下降法算法实现
Class List:
Function List:
    BacktrackingLineSearch(x0, func, grad_func, stepLen): 回溯直线搜索（回溯法）
    GradientDescent(x0, func, grad_func): 梯度下降法

History:
    <author>            <date>              <version>           <desc>
    LiXianQiang         2021/07/15            0.0.1          梯度下降法算法实现
******************************************************************************/
"""
import numpy as np


def BacktrackingLineSearch(x0, desDir, func, grad_func, stepLen=1, scalFac=0.5,
                           termCondi="Armijo"):
    """回溯直线搜索（回溯法）

    核心思路：从起始步长 stepLen 开始，按缩放因子 scalFac 比例减少，直到满足停止条件,
    停止条件包括：Armijo条件，Wolfe条件，Goldstein条件

    参考资料：https://www.cnblogs.com/kemaswill/p/3416231.html
             https://zh.wikipedia.org/wiki/%E7%BA%BF%E6%90%9C%E7%B4%A2
             https://zhuanlan.zhihu.com/p/141837044
             https://www.cnblogs.com/yifdu25/p/8093725.html
            <<Numerical Optimization>>, Chapter 3, p35-p42. J. Nocedal, S.Wright.

    Args:
        x0: 可行解，类型与函数句柄func的形参一样
        desDir: 下降方向，类型与函数句柄func的形参一样
        func: 函数句柄, 原函数，其返回值是 float类
        grad_func: 函数句柄, 原函数所对应的梯度函数，其返回值是 numpy.array类，n×1维
        stepLen：float类，步长(step length)，默认缺省值：1
        scalFac: float类，步长缩放因子(scaling factor of step length)，
            取值范围：(0,1)，默认缺省值：0.5
        termCondi: string类，终止条件(termination condition)，默认缺省值：'Armijo'
            可选参数：'Armijo', 'Wolfe', 'Goldstein'

    Returns:
        stepLen: float类，移动的步长(step length)
    """

    def Armijo():
        """Armijo条件"""
        nonlocal x0, desDir, func, grad_func, stepLen
        # 取值范围 (0, 1)
        c1 = 10e-4  # 经验取值 c1=10e-4

        # 方向导数
        dirDev = (grad_func(x0).T @ desDir)[0, 0]

        # 当结果为 True 时，表示已经得到充分减少，当前的下降步长 stepLen 为最优步长
        result = func(x0 + stepLen * desDir)\
                 <= func(x0) + c1 * stepLen * dirDev
        return result

    def Wolfe():
        """Wolfe条件"""
        nonlocal x0, desDir, func, grad_func, stepLen
        # 取值范围 0 < c1 < c2 < 1
        c1 = 10e-4  # 经验取值 c1=10e-4;
        c2 = 0.99  # 经验取值 牛顿法或拟牛顿法: c2=0.9；非线性共轭梯度法: c2=0.1

        # 方向导数
        dirDev = (grad_func(x0).T @ desDir)[0, 0]

        # 当结果为 True 时，表示已经得到充分减少，当前的下降步长 stepLen 为最优步长
        armijo = func(x0 + stepLen * desDir) \
                 <= func(x0) + c1 * stepLen * dirDev
        curvature = (grad_func(x0 + stepLen * desDir).T @ desDir)[0, 0]\
                    >= c2 * dirDev
        result = armijo and curvature
        return result

    def Goldstein():
        """Goldstein条件"""
        nonlocal x0, desDir, func, grad_func, stepLen
        # 取值范围 (0, 0.5)
        c1 = 10e-4  # 经验取值 c1=10e-4;

        # 方向导数
        dirDev = (grad_func(x0).T @ desDir)[0, 0]

        # 当结果为 True 时，表示已经得到充分减少，当前的下降步长 stepLen 为最优步长
        result = func(x0) + (1 - c1) * stepLen * dirDev\
                 <= func(x0 + stepLen * desDir)\
                 <= func(x0) + c1 * stepLen * dirDev
        return result

    condition = {'Armijo': Armijo, 'Wolfe': Wolfe, 'Goldstein': Goldstein}
    beTerminated = condition.get(termCondi)

    while not beTerminated():
        # 按比例逐渐减少步长
        stepLen = scalFac * stepLen
        print("the stepLen is", stepLen)
    return stepLen


def GradientDescent(x0, func, grad_func):
    """梯度下降法

    Args:
        x0: 可行解，类型与函数句柄func形参一样
        func: 函数句柄, 原函数，其返回值是 float类
        grad_func: 函数句柄, 原函数所对应的梯度函数，其返回值是 numpy.array类，n×1维

    Returns:
        x1: 可行解，类型与函数句柄func形参一样
    """
    # 下降方向，这里选用负梯度方向
    desDir = -1 / np.linalg.norm(grad_func(x0), ord=2) * grad_func(x0)

    # 下降方向的步长计算
    iterStep = BacktrackingLineSearch(x0, desDir, func, grad_func)
    x1 = x0 + iterStep * -1 / np.linalg.norm(grad_func(x0), ord=2) * grad_func(x0)
    return x1
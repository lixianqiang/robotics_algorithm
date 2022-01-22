"""
/******************************************************************************
License: GNU General Public License v2.0
File name: simulated_annealing.py
Author: LiXianQiang     Date:2021/08/05      Version: 0.0.1
Description: 模拟退火算法实现
Class List:
    SimulatedAnnealing:
        __init__(self, optimizedObject, coolingSchedule, timeLimit, solutionType): 初始化
        solved(self): 求解器
Function List:
    SetTimer(timeDuring): 异步计时器
     SolutionGenerator(solved): 解生成器
History:
    <author>            <date>              <version>           <desc>
    LiXianQiang         2021/08/05            0.0.1          模拟退火算法实现
******************************************************************************/
"""

import asyncio
import time
import random
import numpy as np
class SimulatedAnnealing:
    """模拟退火算法

    参考资料: https://en.wikipedia.org/wiki/Simulated_annealing
    """
    def __init__(self, optimizedObject, coolingSchedule,
                 timeLimit=float('inf'), solutionType='minimize'):
        """初始化

        Args:
            optimizedObject: dict类，待优化对象，包括以下key值:
                'objectFunction': 函数句柄，目标函数
                'x0': 'objectFunction'的形参类型，objectFunction的初始解
                'constraints': 函数句柄， 解生成器（用于生成满足约束条件的解），
                'bounds': 函数句柄，边界条件

            coolingSchedule: dict类，退火方案，包括以下key值:
                'temperatureFunction': 函数句柄，温度函数，
                'initialTemperature': float类，初始温度
                'terminatorCondition': 函数句柄，终止条件
            timeLimit: float类，求解运行的时间限制，默认缺省值: float('inf')
            solutionType: string类，待求解类型，可选字段：'minimize','maximize'
        """
        self.objectiveFunction = optimizedObject['objectFunction']
        self.x0 = optimizedObject['x0']
        self.solutionGenrator = optimizedObject['constraints']

        self.temperatureFunction = coolingSchedule['temperatureFunction']
        self.initialTemperature = coolingSchedule['initialTemperature']
        self.terminatorCondition = coolingSchedule['terminatorCondition']

        # 异步定时器
        self.Timer = SetTimer(timeLimit)

        # Boltzmann 常数
        k = 1

        # Metropolis 准则
        objFunc = self.objectiveFunction
        def Metropolis4Minimize(x_new, x, T):
            dE = objFunc(x_new) - objFunc(x)
            if dE <= 0:
                prob = 1
            else:
                prob = np.exp(-dE / k * T)
            return prob

        def Metropolis4Maximize(x_new, x, T):
            dE = objFunc(x_new) - objFunc(x)
            if dE >= 0:
                prob = 1
            else:
                prob = np.exp(dE / k * T)
            return prob

        MetropolisGuidelines = {'minimize': Metropolis4Minimize,
                                'maximize': Metropolis4Maximize}
        self.AcceptProbabilityFunction = MetropolisGuidelines.get(solutionType)

    def solved(self):
        """求解器"""
        x = self.x0
        SolutionGenrator = self.solutionGenrator

        T = self.initialTemperature
        CoolDownOn = self.temperatureFunction
        isSafityTerminCond = self.terminatorCondition
        AccProbFunc = self.AcceptProbabilityFunction
        isTimeUp = self.Timer

        while(not isSafityTerminCond(T) and not isTimeUp()):
            x_new = SolutionGenrator(x)

            accProb = AccProbFunc(x_new, x, T)

            if random.random() < accProb:
                x = x_new
            else:
                x = x
            T = CoolDownOn(T)
        x_opt = x
        return x_opt


def SetTimer(timeDuring):
    """异步计时器

    Args:
        timeDuring: float类，设置定时时间间隔

    Returns:
        Timer: 函数句柄，计时器，其函数返回值：True: 到达指定时间
                                         False: 还没到达指定时间
    """
    # 当调用函数Timer()时，表示计时就开始，is_live会从False变成True
    # 当Timer()返回结果是True时（即：isTimeUp()返回True时），is_live会设置为Fasle
    is_live = False
    # 终止时间，一般由当前时间+dt所决定
    endTime = 0
    # 异步事件
    loop = asyncio.get_event_loop()

    async def isTimeUp():
        nonlocal is_live, endTime
        if time.time() >= endTime:
            is_live = False
            return True
        else:
            return False

    # 异步定时器
    def Timer():
        nonlocal is_live, endTime
        if is_live is False:
            is_live = True
            endTime = time.time() + timeDuring
        return loop.run_until_complete(isTimeUp())

    return Timer

# TODO 以后待优化
def SolutionGenerator(solved):
    """ 解生成器
    根据当前解随机选择离当前解较近的候选解
    randomly selects a solution close to the current one
    """

    def neighbourSolve():
        """the candidate generator"""
        pass
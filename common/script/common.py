"""
/******************************************************************************
License: GNU General Public License v2.0
File name: common.py
Author: LiXianQiang     Date:2021/02/16      Version: 0.0.1
Description: 常用算法实现
Class List:
Function List:
    AngleDiff(endAngle, startAngle): 计算两个角之间的最小转向夹角
    Quaternions2EulerAngle(w, x, y, z): 四元素转欧拉角
    Quaternion2RotationMatrc(w, x, y, z): 四元数转旋转矩阵
    IsNonDecrease(seq): 判断 seq 是否为非递减序列
    SetTimer(dt): 异步计时器实现
History:
    <author>            <date>              <version>           <desc>
    LiXianQiang         2021/02/16          0.0.1               常用算法实现
******************************************************************************/
"""

import numpy as np
import asyncio
import time

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
    abs_deltaAngle = np.fabs(deltaAngle)
    abs_compleAngle = 2 * np.pi - abs_deltaAngle

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


def Quaternions2EulerAngle(w, x, y, z):
    """四元素转欧拉角

    Args: 
        w,x,y,z: float类，分别对应四元数的实部与三个虚部

    Returns:
        roll,pitch,yaw:float类，分别对应:滚动角，俯仰角，偏航角
    """
    roll = np.atan2(2 * (w * x + y * z), 1 - 2 * (x ** 2 + y ** 2))
    pitch = np.asin(2 * (w * y - x * z))
    yaw = np.atan2(2 * (w * z + x * y), 1 - 2 * (y ** 2 + z ** 2))
    return roll, pitch, yaw


def Quaternion2RotationMatrc(w, x, y, z):
    """四元数转旋转矩阵

    Args: 
        w,x,y,z: float类，分别对应四元数的实部与三个虚部

    Returns:
        rotMat:numpy.array类，旋转矩阵，3*3矩阵
    """
    rotMat = np.array([[w ^ 2 + x ^ 2 - y ^ 2 - z ^ 2, 2 * x * y - 2 * w * z,
                        2 * x * z + 2 * w * y],
                       [2 * x * y + 2 * w * z, w ^ 2 - x ^ 2 + y ^ 2 - z ^ 2,
                        2 * y * z - 2 * w * x],
                       [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x,
                        w ^ 2 - x ^ 2 - y ^ 2 + z ^ 2]])
    return rotMat


def IsNonDecrease(seq):
    """判断 seq 是否为非递减序列
    Args:
        seq: list类，待检查序列，列表元素: float类，

    Returns:
        True: 判断 seq 为非递减序列
        False: 反之

    参考资料:
        https://qastack.cn/programming/4983258/python-how-to-check-list-monotonicity
        https://www.zhihu.com/question/52742082
    """
    return all(x <= y for x, y in zip(seq, seq[1:]))


def SetTimer(dt):
    """异步计时器实现

    Args:
        dt: float类，设置定时时间间隔

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

    def Timer():
        nonlocal is_live, endTime
        if is_live is False:
            is_live = True
            endTime = time.time() + dt
        return loop.run_until_complete(isTimeUp())

    return Timer

# TODO 未整理
def CalStepSize(pathLen, stepSize, searchMothed='dichotomy'):
    pow = 0
    while stepSize <= pathLen / (2 ** (pow + 1)):
        pow += 1
    # 二分法
    if searchMothed == 'dichotomy':
        stepSize = pathLen / (2 ** pow)
        return stepSize
    # 最小公倍数
    elif searchMothed == 'miniMulti':
        minMulti = 2 ** pow
        maxMulti = 2 ** (pow + 1)
        testMulti = minMulti
        while testMulti - int(testMulti) == 0:  # 试验倍数是否整数
            if stepSize <= pathLen / testMulti:
                minMulti = testMulti
            else:
                maxMulti = testMulti
            testMulti = minMulti + (maxMulti - minMulti) / 2
        stepSize = pathLen / minMulti
        return stepSize
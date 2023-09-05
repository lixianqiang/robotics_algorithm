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

def CalcCurvature(curr_p, prev_p, next_p):
    """计算门格尔曲率
    trick: 向量外积=向量组成的四边形的面积 向量组成三角形面积=0.5*向量外积
    参考资料: 
        https://github.dev/lixianqiang/robotics_algorithm/blob/master/common/script/common.py
    """
    denominator = CalcDistance(prev_p, curr_p) * CalcDistance(curr_p, next_p) * CalcDistance(prev_p, next_p)
    return 2.0 * ((curr_p[0] - prev_p[0]) * (next_p[1] - prev_p[1]) - (curr_p[1] - prev_p[1]) * (
            next_p[0] - prev_p[0])) / denominator

def IsShiftPoint(curr_p, prev_p, next_p):
    """计算换档位
    trick: 车辆实际轨迹不存在突变(除了换档点), 那么换档点与前后轨迹点组成的夹角>=90,对应的余弦值<=0
    """
    dot_product = (curr_p[0] - prev_p[0]) * (next_p[0] - curr_p[0]) + (curr_p[1] - prev_p[1]) * (
            next_p[1] - curr_p[1])
    norm_vector1 = CalcDistance(prev_p, curr_p)
    norm_vector2 = CalcDistance(curr_p, next_p)
    cos_theta = dot_product / (norm_vector1 * norm_vector2)
    if (cos_theta < 0):
        return True
    return False

def ConvertQuaternionToYaw(quaternion):
    """四元数转航向角
    x,y,z,w对应ROS的geometry_msgs的Quaternion是一一对应
    Returns:
        yaw: 弧度制, 区间范围[-pi, pi]
    """
    x, y, z, w = quaternion
    t0 = +2.0 * (w * z + x * y)
    t1 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t0, t1)
    return yaw

def ConvertYawToQuaternion(yaw):
    """yaw角转四元数
    输出的四元数与ROS的geometry_msgs的Quaternion是一一对应
    """
    quaternion = [0.0, 0.0, 0.0, 0.0]
    quaternion[0] = 0.0
    quaternion[1] = 0.0
    quaternion[2] = math.sin(yaw / 2.0)
    quaternion[3] = math.cos(yaw / 2.0)
    return quaternion

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

def WrapToPi(radian):
    """转换到[-pi, pi]
    Args:
        radian: float类，待转换弧度
    Returns:
        tf_radian: float类，转换后的弧度
    """
    tf_radian = fmod(radian + np.pi, 2 * np.pi)
    if tf_radian < 0.0:
        tf_radian += (2.0 * np.pi)
    return tf_radian - np.pi

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


def MinMaxScaler(mat):
    """归一化
    按列向量的方式进行归一化处理
    """
    norm_mat = (mat - np.min(mat, axis=0)) / (np.max(mat, axis=0) - np.min(mat, axis=0))
    return norm_mat


def QuickSort(array, left=None, right=None, key=None):
    """快速排序"""
    def Partition(array, left, right):
        pivot = left
        index = pivot + 1
        i = index
        while i <= right:
            if key(array[i]) < key(array[pivot]):
                Swap(array, i, index)
                index += 1
            i += 1
        Swap(array, pivot, index - 1)
        return index - 1

    def Swap(array, i, j):
        array[i], array[j] = array[j], array[i]

    if key is None:
        key = lambda x: x
    left = 0 if not isinstance(left, (int, float)) else left
    right = len(array) - 1 if not isinstance(right, (int, float)) else right
    if left < right:
        partitionIndex = Partition(array, left, right)
        QuickSort(array, left, partitionIndex - 1, key)
        QuickSort(array, partitionIndex + 1, right, key)
    return array


# TODO 未测试
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

def GetNext(pattern):
    """返回Next数组，即部分匹配表（Partial Match Table）

    Args:
        pattern: 匹配模板，string类

    Returns:
        next: 部分匹配表，用于KMP匹配算法，list类
    """
    pidx = 0  # index of the current character in the pattern
    nidx = 1  # index of the current character in the next array (Partial Match Table)
    next = [0] 
    while nidx < len(pattern):
        if pattern[pidx] == pattern[nidx]:
            pidx += 1
            nidx += 1
            next.append(pidx)
        elif pidx == 0 and pattern[pidx] != pattern[nidx]:
            nidx += 1
            next.append(0)
        elif pidx != 0 and pattern[pidx] != pattern[nidx]:
            pidx = next[pidx - 1]
    return next

def KMP(str, pattern):
    """KMP算法实现

    Args:
        str: 待匹配主字符串，string类
        pattern: 匹配模板，string类

    Returns:
        idxList: 索引列表，主串中所有匹配的位置索引，list类

    参考资料:
        https://www.ruanyifeng.com/blog/2013/05/Knuth%E2%80%93Morris%E2%80%93Pratt_algorithm.html
        https://www.zhihu.com/question/21923021
    """
    idxList = []
    if len(pattern) == 0:
        return idxList
    pidx = 0
    sidx = 0
    next = GetNext(pattern)
    while sidx < len(str):
        if str[sidx] == pattern[pidx]:
            sidx += 1
            pidx += 1
        elif pidx == 0 and str[sidx] != pattern[pidx]:
            sidx += 1
        elif pidx != 0 and str[sidx] != pattern[pidx]:
            pidx = next[pidx - 1]
        if pidx == len(pattern):
            idxList.append(sidx - pidx)
            pidx = 0

    return idxList
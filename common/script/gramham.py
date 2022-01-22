"""
/******************************************************************************
License: GNU General Public License v2.0
File name: gramham.py
Author: LiXianQiang     Date:2022/01/08      Version: 0.0.1
Description: Gramham算法实现
Class List:
Function List:
    GenerateConvexHull(pointSet, onlineProcess='Off'): 根据给定的点集生成对应的凸包
    Pop(pointSet, index): 将pointSet中第index个的点从pointSet中弹出
    SortedByCos(pointSet, point_center): 对pointSet的每一个点，按照其cos值从大到小
        重新进行排序
History:
    <author>            <date>              <version>           <desc>
    LiXianQiang         2022/01/08            0.0.1          Gramham算法实现
******************************************************************************/
"""

import math
import matplotlib.pyplot as plt
import numpy as np

# 用列表(list)代替栈(stack)
Stack = list

def GenerateConvexHull(pointSet, onlineProcess='Off'):
    """根据给定的点集生成对应的凸包
    使用Graham扫描法来实现凸包的生成
    注意：目前算法只针对2维凸包，对于3维凸包并不能处理

    Args:
        pointSet: list类，待处理的点集，
            pointSet[0]: list类，待处理点集中每一个点的X坐标
            pointSet[1]: list类，待处理点集中每一个点的Y坐标

        # TODO 目前并未对此进行实现
        onlineProcess: String类，开启在线处理，默认缺省值：'Off'
            当该变量为'Off'时，pointSet将一次性接收所有点集，并返回相应的凸包
            当该变量为'On'时，pointSet将持续接收，并进行处理。当由'On'->'Off'时，
                将返回相应的凸包

    参考资料:
        https://blog.csdn.net/u013066730/article/details/106661030
        https://www.bilibili.com/video/av78002153
        https://baike.baidu.com/item/%E5%87%B8%E5%8C%85/179150?fr=aladdin
    """
    # 参数合法性检查
    if len(pointSet) != 2:
        raise Exception('参数维度不正确，应该是2维，但现在是 {dim}维'
                        .format(dim=len(pointSet)))
    elif len(pointSet[0]) != len(pointSet[1]):
        raise Exception('点集的数目不匹配，X轴上有{size_x}个, Y轴上有{size_y}个'
                        .format(size_x=len(pointSet[0]),
                                size_y=len(pointSet[1])))
    elif len(pointSet[0]) <= 2:
        print('警告：当前点集的数目少于2个，虽然仍满足凸包的定义，但不具有应用价值')

    # 创建栈对象，用于存放凸包点
    stack = Stack()

    # 寻找点集中y值最小的坐标所对应的索引，并从点集中弹出，记为p0，并加入栈中
    index = pointSet[1].index(min(pointSet[1]))
    p0 = Pop(pointSet, index)
    stack.append(p0)

    # 计算点集中每一个点与p0的余弦值，并按照余弦值从大到小的顺序对点集进行排序
    SortedByCos(pointSet, p0)

    # 滤除共线点（可选）
    FilterCollinearPoint(pointSet, p0)

    for idx in range(len(pointSet[0])):
        point = np.array([pointSet[0][idx], pointSet[1][idx]])
        if idx == 0:
            stack.append(point)
            continue

        while True:
            # 叉积大于0，表示向量A(即：point-stack[-2])在向量B(即：stack[-1]-stack[2])的逆时针方向
            if np.cross(stack[-1] - stack[-2], point - stack[-1]) > 0:
                stack.append(point)
                break
            # 反之，顺时针方向
            else:
                stack.pop()
                continue
    return stack

def Pop(pointSet, index):
    """将pointSet中第index个的点从pointSet中弹出

    Args:
        pointSet: list类，待处理的点集，
            pointSet[0]: list类，待处理点集中每一个点的X坐标
            pointSet[1]: list类，待处理点集中每一个点的Y坐标

    Returns:
        point: numpy.array类，pointSet中第index个点
    """
    x, y = pointSet[0].pop(index), pointSet[1].pop(index)
    return np.array([x, y])

def SortedByCos(pointSet, point_center):
    """对pointSet的每一个点，按照其cos值从大到小重新进行排序
    关于cos值的计算：pointSet的每一个point与point_center所成夹角的余弦值

    Args:
        pointSet: list类，待处理的点集，
            pointSet[0]: list类，待处理点集中每一个点的X坐标
            pointSet[1]: list类，待处理点集中每一个点的Y坐标
        point_center: numpy.array类，中心点（即：待生成的凸包中y值最小的坐标）
    """
    cosVal_List = []
    for point in zip(pointSet[0], pointSet[1]):
        dx, dy = point[0] - point_center[0], point[1] - point_center[1]
        angle = math.atan2(dy, dx)
        cosVal_List.append(math.cos(angle))

    # 根据cos值的大小，按照从大到小的方式，对（cos值的）索引进行排序
    index_sorted = sorted(range(len(cosVal_List)),
                          key=lambda index: cosVal_List[index],
                          reverse=True)

    # 利用排序后的索引，对点集重新排序
    after_pointSet = [[None] * len(pointSet[0]), [None] * len(pointSet[1])]
    for i, index in enumerate(index_sorted):
        after_pointSet[0][i] = pointSet[0][index]
        after_pointSet[1][i] = pointSet[1][index]

    pointSet[0], pointSet[1] = np.array(after_pointSet[0]), np.array(after_pointSet[1])

# TODO 待实现
def FilterCollinearPoint(pointSet, p0):
    """去除凸包内的共线点
    目的是减少无效的判断次数。使用霍夫变换检测直线，进一步找到共线部分
    """
    pass

if __name__ == '__main__':
    samples = [[36, 78.8, 73.5, 54.1, 29, 69.4, 76.6, 28, 46.7, 42.2, 38.1, 70, 38.3, 86.3, 62, 15.5],
               [19, 18.1, 32.3, 42.2, 12, 40.4, 24.4, 39.3, 40.4, 35.1, 40.2, 53, 59.7, 31.9, 45, 28.3]]

    # samples = [np.random.randn(10).tolist(),
    #            np.random.randn(10).tolist()]

    ps = GenerateConvexHull(samples)
    print(ps)
    X = []
    Y = []
    for xy in ps:
        X.append(xy[0])
        Y.append(xy[1])
    X.append(ps[0][0])
    Y.append(ps[0][1])

    plt.plot(X, Y)
    plt.scatter(samples[0], samples[1])

    plt.show()
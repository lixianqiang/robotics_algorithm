"""
/******************************************************************************
License: GNU General Public License v2.0
File name: cartesian2frenet.py
Author: LiXianQiang     Date:2021/11/11      Version: 0.0.1
Description: 笛卡尔坐标系转Frenet坐标系
Class List:
Function List:
    FrenetAt(func, t, is_2D): 返回曲线函数func在点t处对应的Frenet坐标系
    GetArcFunc(func, domains, is_2D): 返回曲线函数func所对应的弧长函数句柄
    Cartesian2Frenet(refPoints, projPoints, simpleNumber, UsePreciseConversion):
        将投影点 projPoints 的笛卡尔坐标转换成以参考点 refPoints 所组成的曲线的 Frenet 坐标

History:
    <author>            <date>              <version>           <desc>
    LiXianQiang         2021/11/11          0.0.1       笛卡尔坐标系转Frenet坐标系
******************************************************************************/
"""

import numpy as np
from scipy.interpolate import interp1d, lagrange
from scipy.misc import derivative
from scipy import integrate


def FrenetAt(func, t, is_2D=True):
    """返回曲线函数func在点t处对应的Frenet坐标系

    Args:
        func: inter1d类（已测试）, 多项式曲线函数，
            (注意需要添加字段：fill_value="extrapolate")
        t：float类，函数func的输入参数，取值范围：函数func的有效定义域
        is_2D: bool类，True: 计算以二维形式，False: 计算以三维形式
            默认缺省值：True

    Returns:
        orgin：np.array类，曲线函数func在点t所对应的曲线函数值，n×1维，n=2,3
            （同时对应为Frenet坐标系的原点o）
        tanVec：np.array类，曲线函数func在点t所对应的曲线切线，n×1维，n=2,3
            （同时对应为Frenet坐标系的s轴）
        normVec：np.array类，曲线函数func在点t所对应的曲线法线，n×1维，n=2,3
            （同时对应为Frenet坐标系的d轴）
        subnormVec：np.array类，曲线函数func在点t所对应的曲线副法线，
            n×1维，n=2,3（同时对应为Frenet坐标系的b轴）
            其中，当is_2D=True时，其返回值为None
    """
    funcVal = func(t)
    # 若函数derivative使用默认精度(dx=1)会使得绘图出现异常，不相切/不垂直
    gradVal = derivative(func, t, dx=1e-6, n=1)
    unit_gradVal = gradVal / np.linalg.norm(gradVal)

    origin = np.array([funcVal]).T
    tanVec = np.array([unit_gradVal]).T
    if is_2D:
        normVec = np.array([[0, -1], [1, 0]]) @ tanVec
        subnormVec = None
    else:
        normVec = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]) @ tanVec
        subnormVec = np.cross(tanVec, normVec)
    return origin, tanVec, normVec, subnormVec


def GetArcFunc(func, domains, is_2D=True):
    """返回曲线函数func所对应的弧长函数句柄

    Args:
        func: inter1d类（已测试）, 多项式曲线函数
        domains: np.array类，函数func的有效定义域
        is_2D: bool类，True: 计算以二维形式，False: 计算以三维形式
            默认缺省值：True

    Returns:
        ArcFunc: 函数句柄，曲线函数func所对应的弧长函数，其输入参数t：float类，
            其返回值：点t处所对应的弧长长度
    """
    ranges = []
    for t in np.linspace(0, 1, domains.shape[0]):
        # dot_x = dx / dt, dot_y = dy / dt， dot_z = dz / dt
        # 分别指沿着x方向的速度，沿着y方向的速度, 沿着z方向的速度
        if is_2D:
            # 若函数derivative使用默认精度(dx=1)会使得绘图出现异常，不相切/不垂直
            dot_x, dot_y = derivative(func, t, dx=1e-6, n=1)
            dv = np.hypot(dot_x, dot_y)
        else:
            dot_x, dot_y, dot_z = derivative(func, t, dx=1e-6, n=1)
            dv = np.sqrt(dot_x**2 + dot_y**2 + dot_z**2)
        # dv的本质是曲线函数func在点t处（关于梯度方向）的方向导数，也是速率本身
        ranges.append(dv)

    # # 五次多项式插值
    integfunc = interp1d(domains, ranges, kind=5, fill_value="extrapolate")
    # 拉格朗日插值
    # integfunc = lagrange(domains, ranges)
    t0, t1 = domains[0], domains[-1]
    def ArcFunc(t):
        if t0 <= t <= t1:
            arcLen, _ = integrate.quad(integfunc, t0, t)  # 插值型积分
        else:
            raise ValueError("超出有效定义域范围")
        return arcLen
    return ArcFunc


def Cartesian2Frenet(refPoints, projPoints, simpleNumber=None,
                     UsePreciseConversion=False):
    """将投影点 projPoints 的笛卡尔坐标转换成以参考点 refPoints 所组成的曲线的 Frenet 坐标

    Args:
        refPoints: tuple类，参考点（参考线对应的离散采样点）的笛卡尔坐标，
            其元组元素：list类，每个对应维度下的坐标序列。例如：
            refPoints = (x, y, z), 其中 x = [0,1,2,...],
            y = [3,4,3,...], z = [0, 0, 0,...]
        projPoints: tuple类，待投影点的笛卡尔坐标，其元组元素：list类，
            每个对应维度下的坐标序列。具体参数形式请参考'refPoints'
        simpleNumber: int类，区间采样数目，默认缺省值是：None
            注意：1，采样数目越大，转换精度越高，同时数目的增加对运算效率不会有较大的影响，
                    建议设置较大的采样数目；
                2，当 simpleNumber 是 None 时，会根据 'refPoints' 数目的两倍进行设置
        UsePreciseConversion: bool类，是否进行精确转换，默认缺省值：False
            注意：开启此选项会降低运行效率

    Returns:
        projPoints4Frenet: tuple类，待投影点的 Frenet 坐标(自然坐标)，
            其元组元素：list类，每个对应维度下的坐标序列。具体输出形式请参考'refPoints'
            需要注意：二维输出形式为：(s, d)，三维输出形式为: (s, d, b)；
                其坐标含义：s:纵向距离，d:横向距离，b:垂直高度。
                对于无效投影点(曲线之外的坐标点)，默认缺省值：(None, None, None)
    """

    def CalculateProjectError(subSpace, projectVector, is_2D=True):
        """计算投影向量projectVector到子空间subSpace的投影误差
        利用投影矩阵计算投影点到超平面的误差距离
        具体原理请参考：https://www.bilibili.com/video/BV1ix411f7Yp?p=15

        Args:
            subSpace: np.array类，投影子空间，n×m维，n×m=3×2,2×1
            projectVector: np.array类，投影向量，n×1维，n=3,2
            is_2D: bool类，True: 以二维形式计算，Flase: 以三维形式计算；
                默认缺省值：True

        Returns:
            projErr: float类，投影误差 projection error
        """
        # TODO 三维的并未测试
        if is_2D:
            a = subSpace
            b = projectVector
            P = a @ a.T / (a.T @ a)
            projErr = np.linalg.norm(b - P @ b, ord=2)
        else:
            A = subSpace
            b = projectVector
            P = A @ np.linalg.inv(A.T @ A) @ A.T
            projErr = np.linalg.norm(b - P @ b, ord=2)
        return projErr

    # 参数合法性检查
    for inputArgs in [refPoints, projPoints]:
        # 若输入参数类型均为是list类，那么列表推导式应该全为True
        result = np.array([isinstance(seqObj, list) for seqObj in inputArgs])
        isList4All = np.all(result == True)
        if not isList4All:
            raise TypeError("参数类型不正确，请检查")

        # 如果输入参数的维度是一致的（空间上的坐标点的xyz的数目是一一对应的），
        # 那么集合推导式所生成的集合应该只有一个元素
        setObj = {len(seqObj) for seqObj in inputArgs}
        hasSameDimension = len(setObj) == 1
        if not hasSameDimension:
            raise ValueError("坐标点数目(维度)不正确，请检查")

    # refPntDim = (rows, cols) 其中，refPntDim：reference point's dimension
    refPntDim = (len(refPoints), len(refPoints[0]))
    # TODO 三维情况并未进行验证
    if refPntDim[0] == 2:
        is_2D = True
    elif refPntDim[0] == 3:
        is_2D = False
    else:
        raise Exception("Cartesian坐标系转Frenet坐标系只支持二维或三维")

    # 定义域，值域（将[0, 1]区间与参考点建立映射关系）
    domains, ranges = np.linspace(0, 1, refPntDim[1]), np.array(refPoints)
    # 五次多项式曲线
    polyFunc = interp1d(domains, ranges, kind=5, fill_value="extrapolate")

    # 弧长函数
    arcFunc = GetArcFunc(polyFunc, domains, is_2D)

    # 存放离散点的时间戳及其所对应的超平面参数
    timeStamp = []  # 在确定投影点距离最近的超平面位置时用到
    hyperPlanes_a = []
    hyperPlanes_b = []

    if simpleNumber is None:
        simpleNumber = refPntDim[1] * 2
    elif not isinstance(simpleNumber, int):
        raise TypeError("参数 simpleNumber 不是 int类型，请检查")

    for t in np.linspace(0, 1, simpleNumber):
        timeStamp.append(t)
        # 超平面：a.T * x = b
        # 需要强调的是：这里的超平面，是由曲线离散点的法向量所张成的；
        # 相对地，曲线离散点的切向量则是超平面的法向量
        orgin, tanVec, normVec, subnormVec = FrenetAt(polyFunc, t)
        a, x = tanVec, orgin
        b = a.T @ x
        hyperPlanes_a.append(a)
        hyperPlanes_b.append(b)

    # 转换成矩阵形式，加快运行速度
    hyperPlanes_a = np.concatenate(hyperPlanes_a, axis=1)
    hyperPlanes_b = np.concatenate(hyperPlanes_b)
    projPoints = np.array(projPoints)

    # 超平面分离定理:
    # 若点x与超平面的法向量在同一侧时，有: a.T * x >= b
    mask_onSameSide = hyperPlanes_a.T @ projPoints >= hyperPlanes_b

    # 超平面的位置矩阵，该矩阵每一行表示不同的投影点，每一列表示对应每一个超平面，
    # 若[i,j]=True表示第j个投影点位于第i个和第i+1个超平面之间
    # 需要强调一点：下面的掩码方法无法识别刚好位于轨迹末端点的超平面上的投影点，
    # 这是根据实际情况所做出的取舍
    posMat4HypePlanes = mask_onSameSide[:-1, :] ^ mask_onSameSide[1:, :]

    # 用于（临时）存放变换后的Frenet坐标
    tempStorage = dict()

    for j in range(projPoints.shape[1]):
        # 取出第j个投影点
        projPoint = projPoints[:, [j]]

        # 第j个投影点能投影到哪些超平面上
        idx4HyperPlanes = np.nonzero(posMat4HypePlanes[:, j])[0].tolist()

        if idx4HyperPlanes != []:
            # 关于距离投影点最近的超平面的参数信息，
            # 对应的时间戳，投影点投影到轨迹的距离，对应的超平面区间
            t_mD = None  # mD是minDist的缩写
            minDist = float('inf')
            idx_minDist4HyperPlanes = None

            for i in idx4HyperPlanes:
                t0, t1 = timeStamp[i], timeStamp[i + 1]
                tm = t0 + (t1 - t0) / 2  # tm：t0与t1之间的中点，m是middle的意思
                tm_orign, _, tm_normVec, _ = FrenetAt(polyFunc, tm, is_2D)
                # 将投影点投影到tm处的超平面(法向量)，得到投影点到轨迹的纵向距离d
                tm_dist = (tm_normVec.T @ (projPoint - tm_orign))[0, 0]
                if np.fabs(minDist) > np.fabs(tm_dist):
                    t_mD = tm
                    minDist = tm_dist
                    idx_minDist4HyperPlanes = i

            # 如果使用精确转换，则在前面的基础上使用二分法进一步确定
            if UsePreciseConversion:
                i = idx_minDist4HyperPlanes
                t0, t1 = timeStamp[i], timeStamp[i + 1]
                a0, a1 = hyperPlanes_a[:, i], hyperPlanes_a[:, i + 1]
                b0, b1 = hyperPlanes_b[i], hyperPlanes_b[i + 1]

                last_errDist = float('inf')
                iterNum = 15
                while iterNum:
                    tm = t0 + (t1 - t0) / 2  # m是middle的意思
                    tm_orign, tm_tanVec, tm_normVec, _ \
                        = FrenetAt(polyFunc, tm, is_2D)
                    xm, am = tm_orign, tm_tanVec
                    bm = am.T @ xm

                    # 当t0与tm所对应的超平面相对于投影点均在同一侧时，
                    # 令t0 = tm, a0 = am, b0 = bm
                    # 通过"=="来实现同或运算
                    if (am.T @ projPoint >= bm) == (a0.T @ projPoint >= b0):
                        t0, a0, b0 = tm, am, bm
                    else:
                        t1, a1, b1 = tm, am, bm

                    errDist = CalculateProjectError(tm_normVec,
                                                    projPoint - tm_orign,
                                                    is_2D)
                    if errDist <= 1e-6:
                        break
                    iterNum -= 1

                    # 由于曲线变化，误差收敛不是单调递减，但存在减少趋势，
                    # 这种情况在曲率变化剧烈的区间尤为明显
                    if errDist < last_errDist:
                        last_errDist = errDist
                    else:
                        iterNum += 1

                t_mD = tm
                minDist = (tm_normVec.T @ (projPoint - tm_orign))[0, 0]

            (s, d) = arcFunc(t_mD), minDist
            tempStorage.setdefault('{idx}'.format(idx=j), (s, d))

    projPoints4Frenet = np.empty(projPoints.shape, dtype=object)
    for idx, coord in tempStorage.items():
        projPoints4Frenet[:, int(idx)] = coord
    projPoints4Frenet = projPoints4Frenet.tolist()

    return projPoints4Frenet

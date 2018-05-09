# -*- encoding:utf8 -*-

from numpy import *
from math import *


# 载入数据
def loadDataSet(fileName):
    dataMat = []
    f = open(fileName)
    for i in f.readlines():
        nline = i.strip().split('\t')
        fline = map(float, nline)
        dataMat.append(fline)
    return dataMat


# 定义距离计算函数，这里使用欧氏距离
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))

# 编写创建质心函数，为给定数据集创建包含k个随机质心的集合。
# 随机生成k个点作为初始质心
def randCent(dataSet, k):
    n = shape(dataSet)[1]                   # n是列数(坐标维数)
    # 创建一个矩阵保存随机生成的k个质心
    centroids = mat(zeros((k, n)))
    for j in range(n):

        print(type(array(dataSet[:, j].max())))
        # print(min(dataSet[:, j]))

        rangej = float(array(dataSet[:, j].max()) - array(dataSet[:, j].min()))     # 求第j列最大值与最小值的差
        centroids[:, j] = min(dataSet[:, j]) + rangej * random.rand(k, 1)         # 生成k行1列的在(0, 1)之间的随机数矩阵
    return centroids


# 实现kMeans算法。
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    # 获取数据总量
    m = shape(dataSet)[0]
    # 使用一个矩阵辅助记录，第一列保存所属质心下标，
    # 第二列保存到该质心的距离的平方
    clusterAssment = mat(zeros((m, 2)))
    centroids = createCent(dataSet, k)
    # 使用一个标记记录质心是否发生变化
    # 若没变化则说明算法已经收敛
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            # 设置标记记录数据点到所有质心的最小距离及质心下标
            minDist = inf;
            minIndex = -1
            for j in range(k):
                # 调用函数计算数据点到质心的距离
                # 保存到变量distJI
                # 并更新相关标记
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:  # 更新最小距离和质心下标
                    minDist = distJI;
                    minIndex = j
            # 比较clusterAssment中这一行的第一列记录的下标是否等
            # 于前面更新的质心下标，若相等则说明质心已收敛，否则
            # 还没收敛，据此设置相关标记
            # 然后记录更新clusterAssment中的数据
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            # 记录最小距离质心下标，最小距离的平方
            clusterAssment[i, :] = minIndex, minDist ** 2
        # 打印质心
        print(centroids)

        # 根据新的聚类结果重新计算质心
        for cent in range(k):  # 更新质心位置
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]  # 获得距离同一个质心最近的所有点的下标，即同一簇的坐标
            centroids[cent, :] = mean(ptsInClust, axis=0)  # 求同一簇的坐标平均值，axis=0表示按列求均值
    return centroids, clusterAssment


if __name__ == "__main__":
    dataMat = mat(loadDataSet('testSet.txt'))
    centroids, clustAssing = kMeans(dataMat, 4)
    print(centroids)
    print(clustAssing)
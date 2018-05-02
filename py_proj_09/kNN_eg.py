# -*- coding: utf-8 -*-

from numpy import *
import operator


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


# 分类函数
def kNNClassify(newInput, dataSet, labels, k):
    dataSize = dataSet.shape[0]
    newInput = tile(newInput, (dataSize, 1))

    # 计算距离并且按照返回排序后的下标值列表
    distance = (((newInput-dataSet)**2).sum(axis=1))**0.5
    sortedDistIndices = distance.argsort()

    # 创建一个字典来统计前k个最小距离中各个类别出现的次数
    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDistIndices[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0)+1

    predictedClass = sorted(list(classCount.items()), key=lambda d: d[1], reverse=True)

    return predictedClass

    # # 选取出现的类别次数最多的类别
    # maxCount = 0
    # for key, value in predictedClass:
    #     if value > maxCount:
    #         maxCount = value
    #         classes = key
    #
    # return classes


# >>> import sys
# >>> sys.path.append("kNN_eg.py")
# >>> import kNN_eg
# >>> from numpy import *
# >>> dataSet,labels=kNN_eg.createDataSet()
# >>> output=kNN_eg.kNNClassify([0,0],dataSet,labels,3)
# >>> print(output)

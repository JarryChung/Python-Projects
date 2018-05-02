# -*- coding: utf-8 -*-

import numpy as np
import operator
from os import listdir


def img2vector(filename):
    # 初始化返回向量returnVect
    returnVect = np.zeros((1, 1024))
    # 打开文件
    fileIn = open(filename)
    for i in range(32):
        # 读取整行数据
        lineStr = fileIn.readline()
        for j in range(32):
            # 将头32个字符保存在返回变量中
            returnVect[0, 32*i+j] = int(lineStr[j])

    # 关闭文件
    fileIn.close()
    return returnVect


def loadDataSet():
    # Step1: 获取训练集
    print("---Getting training set...")
    # 根据实际情况设置数据集路径
    dataSetDir = "./"
    trainingFileList = listdir(dataSetDir + "trainingDigits")
    numSamples = len(trainingFileList)

    train_x = np.zeros((numSamples, 1024))
    # 保存标签数据，即分类结果
    train_y = []
    for i in range(numSamples):
        filename = trainingFileList[i]
        train_x[i, :] = \
        img2vector(dataSetDir + "trainingDigits/%s" % filename)
        label = int(filename.split('_')[0])
        train_y.append(label)

    # Step2: 获取测试集
    print("---Getting test set...")
    # 要求使用test_x保存数据，test_y保存类别
    testFileList = listdir(dataSetDir + "testDigits")
    numTest = len(testFileList)

    test_x = np.zeros((numTest, 1024))

    test_y = []
    for j in range(numTest):
        filename = testFileList[j]
        test_x[j, :] = \
        img2vector(dataSetDir + "testDigits/%s" % filename)
        label = int(filename.split('_')[0])
        test_y.append(label)

    return train_x, train_y, test_x, test_y


# 分类器
def kNNClassify(newInput, dataSet, labels, k):
    dataSize = dataSet.shape[0]
    input = np.tile(newInput, (dataSize, 1))

    # 计算距离并且按照返回排序后的下标值列表
    distance = (((input-dataSet)**2).sum(axis=1))**0.5
    sortedDistIndices = distance.argsort()

    # 创建一个字典来统计前k个最小距离中各个类别出现的次数
    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDistIndices[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0)+1

    predictedClass = sorted(list(classCount.items()), key=lambda d: d[1], reverse=True)

    return predictedClass


# 测试函数
def testHandWritingClass():
    # 获取数据集
    print("Step 1: Load data...")
    train_x, train_y, test_x, test_y = loadDataSet()

    # 由于knn是不需要训练步骤的，所以这里直接使用pass跳过
    print("Step 2: Training...")

    print("Step 3: Testing...")
    numTestSamples = test_x.shape[0]
    matchCount = 0  # 用以统计分类正确的数目
    for i in range(numTestSamples):
        # 对测试集中的数据进行分类，取k=3，将得到的结果与标签对比，如果相等则分类正确的数目加一
        classifierResult = kNNClassify(test_x[i], train_x[i], train_y[i], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult[0][0], test_y[i]))
        if classifierResult[0][0] == train_y[i]:
            matchCount += 1.0

    # 所有测试集数据都跑完后计算分类的正确率，保存到acuuracy变量
    accuracy = matchCount/float(numTestSamples)

    print("Step 4: Show the result...")
    print("The classify accuracy is %.2f%%" % (accuracy * 100))


if __name__ == "__main__":
    testHandWritingClass()

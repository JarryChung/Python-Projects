# -*- encoding: utf-8 -*-

from numpy import *
import array


# 准备数据
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec=[0,1,0,1,0,1]                      #1表示侮辱性言论，0表示正常言论
    return postingList, classVec


# 构建词汇表生成函数
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # 取两个集合的并集
    return list(vocabSet)


# 构建词向量
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)  # 生成零向量的array
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1  # 单词出现则记为1
        else:
            print('the word:%s is not in my Vocabulary!' % word)
    return returnVec  # 返回全为0和1的向量


# 根据训练集计算概率
def trainNB1(trainMat,listClasses):
    numTrainDocs = len(trainMat)
    numWords = len(trainMat[0])
    pAb = sum(listClasses) / float(numTrainDocs)
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if listClasses[i] == 1:
            p1Num = p1Num + trainMat[i]
            p1Denom = p1Denom + sum(trainMat[i])
        else:
            p0Num = p1Num + trainMat[i]
            p0Denom = p1Denom + sum(trainMat[i])
    p1Vec = log(p1Num / p1Denom)
    p0Vec = log(p0Num / p0Denom)
    return p0Vec, p1Vec, pAb


# 根据上一步计算出来的概率编写分类器函数
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify*p1Vec)+log(pClass1)
    p0 = sum(vec2Classify*p0Vec)+log(1-pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


# 编写测试函数。
def testingNB():
    listPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listPosts)
    trainMat=[]
    for postinDoc in listPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    p0V,p1V,pAb=trainNB1(trainMat,listClasses)
    testEntry=['love','my','dalmation']
    thisDoc=setOfWords2Vec(myVocabList,testEntry)
    print(testEntry,'classified as:',classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry=['stupid','garbage']
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
    print(testEntry,'classified as:',classifyNB(thisDoc,p0V,p1V,pAb))

'''
在python命令提示符中输入以下语句进行测试：
	import bayes
	bayes.testingNB()
'''
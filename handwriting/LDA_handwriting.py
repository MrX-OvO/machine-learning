# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 22:24:59 2020

@author   : MrX_OvO
@email    : 1176471624@qq.com
@copyright: MrX_OvO Fri Oct 16 22:24:59 2020

To:
    One is never too old to learn.

/////////////////////////////////////////////////////////////////////
//                          _ooOoo_                               //
//                         o8888888o                              //
//                         88" . "88                              //
//                         (| ^_^ |)                              //
//                         O\  =  /O                              //
//                      ____/`---'\____                           //
//                    .'  \\|     |//  `.                         //
//                   /  \\|||  :  |||//  \                        //
//                  /  _||||| -:- |||||-  \                       //
//                  |   | \\\  -  /// |   |                       //
//                  | \_|  ''\---/''  |   |                       //
//                  \  .-\__  `-`  ___/-. /                       //
//                ___`. .'  /--.--\  `. . ___                     //
//              ."" '<  `.___\_<|>_/___.'  >'"".                  //
//            | | :  `- \`.;`\ _ /`;.`/ - ` : | |                 //
//            \  \ `-.   \_ __\ /__ _/   .-` /  /                 //
//      ========`-.____`-.___\_____/___.-`____.-'========         //
//                           `=---='                              //
//      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^        //
//             佛祖保佑      永不宕机      永无BUG                  //
////////////////////////////////////////////////////////////////////
"""


import numpy as np
import os

# 将图片转为向量（1x1024）
def img2vec(filename):
    retVet = []
    with open(filename) as f:
        for item in f.readlines():
            for i in range(32):
                retVet.append(int(item[i]))
    retVet = np.array(retVet)
    return retVet

# 线性判别 训练模型
def LDA():
    path = 'trainingData'
    trainingFileList = os.listdir(path)
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    u0, u = np.zeros((1, 1024)), np.zeros((1, 1024))
    w = np.zeros((1, 1024))
    for i in range(m):
        trainingMat[i, :] = img2vec(path+'/'+trainingFileList[i])
    u0 = np.mean(trainingMat[:100, :], axis = 0).reshape((1, 1024))
    u = np.mean(trainingMat[100:, :], axis = 0).reshape((1, 1024))
    S0 = np.cov(trainingMat[:100, :]-u0, rowvar = False)
    S = np.cov(trainingMat[100:, :]-u, rowvar = False)
    Sw = S0+S
    U, sigma, Vt = np.linalg.svd(np.mat(Sw), full_matrices = False)
    SwInv = Vt*np.linalg.inv(np.diag(sigma))*U
    w = np.dot((u0-u), SwInv)
    return w, u0, u

# 测试
def validation():
    w, u0, u = LDA()
    path = 'testData'
    validationFileList = os.listdir(path)
    m = len(validationFileList)
    testMat = np.zeros((m, 1024))
    labels = [ ]
    for i in range(m):
        testMat[i, :] = img2vec(path+'/'+validationFileList[i])
        filenameStr = validationFileList[i]
        fileStr = filenameStr.split('.')[0]
        classNum = int(fileStr.split('_')[0])
        labels.append(classNum)
    ldaMat = np.dot(testMat, w.reshape((-1, 1)))
    y0 = np.dot((u0+u)/2, w.reshape((-1, 1)))
    errorCount, errorRate = 0, 0
    for i in range(m):
        if ldaMat[i, :]-y0<0 and labels[i] == 0 :
            print('第%d张图片分类正确'%(i+1))
        else:
            print('第%d张图片分类错误'%(i+1))
            errorCount += 1
    errorRate = errorCount/m
    print('验证集分类错误率为：%.3f' % errorRate)

# 分类器
def lda_classifier():
    w, u0, u = LDA()
    path = 'testData'
    testFileList = os.listdir(path)
    m = len(testFileList)
    y0 = np.dot((u0+u)/2, w.reshape((-1, 1)))
    errorCount, errorRate = 0, 0
    for i in range(m):
        vec = img2vec(path+'/'+testFileList[i])
        ldaVec = np.dot(vec, w.reshape((-1, 1)))
        if ldaVec-y0<0 :
            print('第%d张输入图片是数字0'%(i+1))
        else:
            print('第%d张输入图片不是数字0'%(i+1))
            errorCount += 1
    errorRate = errorCount/m
    print('测试集分类错误率为：%.3f' % errorRate)
    print('accuracy_score:%.3f \n' % (1 - errorRate))

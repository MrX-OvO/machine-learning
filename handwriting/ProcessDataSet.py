# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 20:49:10 2020

@author   : MrX_OvO
@email    : 1176471624@qq.com
@copyright: MrX_OvO Mon Nov  9 20:49:10 2020

To:
    One is never too old to learn.

////////////////////////////////////////////////////////////////////
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

# 将图片32x32转为1x1024
def img2vector(filename):
    retVector = []
    with open(filename) as f:
        for item in f.readlines():
            for i in range(32):
                retVector.append(int(item[i]))
    retVector = np.array(retVector).reshape((1, 1024))
    return retVector

# 获得数据集 X:mx1024, y:mx1
def getDataSet(path):
    fileList = os.listdir(path)
    m = len(fileList)
    X, y = [], []
    for i in range(m):
        vector_i = img2vector(path + '/' + fileList[i])
        label_i = int(fileList[i].split('.')[0].split('_')[0])
        X.append(vector_i)
        y.append(label_i)
    X, y = np.array(X).reshape((m, 1024)), np.array(y).reshape((m, 1))
    return X, y
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 22:05:52 2020

@author   : MrX_OvO
@email    : 1176471624@qq.com
@copyright: MrX_OvO Wed Nov 11 22:05:52 2020

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


from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from ProcessDataSet import getDataSet

def decisionTree():
    path1 = 'trainingData'
    X_train, y_train = getDataSet(path1)
    tree_clf = DecisionTreeClassifier(random_state = 1)
    tree_clf.fit(X_train, y_train)
    path2 = 'testData'
    X_test, y_test = getDataSet(path2)
    y_pred = tree_clf.predict(X_test)
    m = len(y_pred)
    errorCount, errorRate = 0, 0
    for i in range(m):
        if y_pred[i] ==  y_test[i]:
            print('第%d张输入图片是数字0'%(i+1))
        else:
            print('第%d张输入图片不是数字0'%(i+1))
            errorCount +=  1
    errorRate = errorCount / m
    print('测试集分类错误率为：%.3f' % errorRate)
    print('accuracy_score:%.3f \n' % accuracy_score(y_test, y_pred))

def bagging():
    path1 = 'trainingData'
    X_train, y_train = getDataSet(path1)
    bag_clf = BaggingClassifier(DecisionTreeClassifier(random_state = 1), \
                                n_estimators = 500, max_samples = 200, \
                                    bootstrap = True, n_jobs = -1, random_state = 1)
    bag_clf.fit(X_train, y_train)
    path2 = 'testData'
    X_test, y_test = getDataSet(path2)
    y_pred = bag_clf.predict(X_test)
    m = len(y_pred)
    errorCount, errorRate = 0, 0
    for i in range(m):
        if y_pred[i]  ==  y_test[i]:
            print('第%d张输入图片是数字0'%(i+1))
        else:
            print('第%d张输入图片不是数字0'%(i+1))
            errorCount +=  1
    errorRate = errorCount / m
    print('测试集分类错误率为：%.3f' % errorRate)
    print('accuracy_score:%.3f \n' % accuracy_score(y_test, y_pred))

def randomForest():
    path1 = 'trainingData'
    X_train, y_train = getDataSet(path1)
    rf_clf = RandomForestClassifier(n_estimators = 500, max_leaf_nodes = 16, \
                                    n_jobs = -1, random_state = 1)
    rf_clf.fit(X_train, y_train)
    path2 = 'testData'
    X_test, y_test = getDataSet(path2)
    y_pred = rf_clf.predict(X_test)
    m = len(y_pred)
    errorCount, errorRate = 0, 0
    for i in range(m):
        if y_pred[i]  ==  y_test[i]:
            print('第%d张输入图片是数字0'%(i+1))
        else:
            print('第%d张输入图片不是数字0'%(i+1))
            errorCount +=  1
    errorRate = errorCount / m
    print('测试集分类错误率为：%.3f' % errorRate)
    print('accuracy_score:%.3f \n' % accuracy_score(y_test, y_pred))
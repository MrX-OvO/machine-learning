# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 18:57:17 2020

@author   : MrX_OvO
@email    : 1176471624@qq.com
@copyright: MrX_OvO Wed Nov 11 18:57:17 2020

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


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from ProcessDataSet import getDataSet

def adaBoost():
    path1 = 'trainingData'
    X_train, y_train = getDataSet(path1)
    ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 1), \
                                 n_estimators = 200, algorithm = 'SAMME.R', \
                                     learning_rate = 0.5, random_state = 1)
    ada_clf.fit(X_train, y_train)
    path2 = 'testData'
    X_test, y_test = getDataSet(path2)
    y_pred = ada_clf.predict(X_test)
    m = len(y_pred)
    errorCount, errorRate = 0, 0
    for i in range(m):
        if y_pred[i] == y_test[i]:
            print('第%d张输入图片是数字0'%(i+1))
        else:
            print('第%d张输入图片不是数字0'%(i+1))
            errorCount +=  1
    errorRate = errorCount / m
    print('测试集分类错误率为：%.3f' % errorRate)
    print('accuracy_score:%.3f \n' % accuracy_score(y_test, y_pred))

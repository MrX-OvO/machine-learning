# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 22:24:34 2020

@author   : MrX_OvO
@email    : 1176471624@qq.com
@copyright: MrX_OvO Fri Oct 16 22:24:34 2020

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


from LDA_handwriting import lda_classifier
from bagging import decisionTree, bagging, randomForest
from adaBoost import adaBoost

if __name__ == '__main__':
    print('lda_classifier():')
    lda_classifier()

    print('decisionTree():')
    decisionTree()

    print('bagging():')
    bagging()

    print('randomForest():')
    randomForest()

    print('adaBoost():')
    adaBoost()

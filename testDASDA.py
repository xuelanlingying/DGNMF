import NinaProDB5DataOperation
import UCIOperation
import DASDADataOperation

import myNMF
import myDLNMF
import myMulDNMF

from sklearn.decomposition import NMF
import os

import h5py
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import math
from sklearn.svm import SVC
from sklearn import svm
from sklearn.metrics import accuracy_score

from sklearn import preprocessing

import MaximumMeanDiscrepancy as MMD



min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

ylist = [1,2,3,4]
# srcid = 1
# tagid = 2
###########################################################

tartrainlist = []
tartestlist = []




ylist = [1,2,3,4]
tagid = 1
###########################################################


tartrainlist = []
tartestlist = []



for i in ylist:
    for j in ylist:
        if i!=j:

            yid = i
            tagid = j

            featuredata1,labeldata1 = DASDADataOperation.getSimPeoDatabyactionlist(yid, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            N = len(labeldata1)
            index = list(np.random.permutation(N))
            data1 = featuredata1[index, :360]
            label1 = labeldata1[index]
            data1 = data1
            X1 = data1.T

            featuredata2, labeldata2 = DASDADataOperation.getSimPeoDatabyactionlist(tagid, [1,2,3,4,5,6,7,8,9,10])

            N = len(labeldata2)
            index = list(np.random.permutation(N))
            data2 = featuredata2[index, :360]
            label2 = labeldata2[index]
            data2 = data2
            X2 = data2.T
            print("tagid", tagid)

            print('X1,X2')
            print(X1.shape)
            print(X2.shape)


            X = np.concatenate((X1, X2), axis=1)


            # mmddist = MMD.MK_MMD(np.array(featuredata1),np.array(featuredata2))
            #
            # print(mmddist)

            testacclist = []

            for di in range(1,450,10):
                mynmf = myMulDNMF.MyMulDNMF(n_components=di,  # k value,默认会保留全部特征
                                            init='nndsvd',
                                            # W H 的初始化方法，包括'random' | 'nndsvd'(默认) |  'nndsvda' | 'nndsvdar' | 'custom'.
                                            solver='mu',  # 'cd' | 'mu'
                                            beta_loss='frobenius',  # {'frobenius', 'kullback-leibler', 'itakura-saito'}，一般默认就好
                                            tol=1e-4,  # 停止迭代的公差
                                            max_iter=2000,  # 最大迭代次数
                                            random_state=111,
                                            l1_ratio=0.,  # 正则化参数
                                            )


                ni = 10

                # for ni in range(15,21):
                w = mynmf.fit_transform(X, 2, y=label1 ,n_neighbors = ni)
                h = mynmf.components_  # H矩阵

                HS = (h.T)[:len(data1)].T
                HT = (h.T)[-len(data2):].T

                print("源域：", str(yid), "-->目标域：", str(tagid), ", 降维后维度：", di)

                print(HS.T.shape)
                print(HT.T.shape)

                # clf = svm.SVC(decision_function_shape='ovo')
                # clf.fit(X1.T, label1)
                # y_test_pred = clf.predict(X2.T)
                # trainacc = round(accuracy_score(label1, clf.predict(X1.T)), 4)
                # print('未降维train准确率：', trainacc)
                # testacc = round(accuracy_score(label2, clf.predict(X2.T)), 4)
                # print('未降维test准确率：', testacc)

                clf = svm.SVC(decision_function_shape='ovo')
                clf.fit(HS.T, label1)
                y_test_pred = clf.predict(HT.T)
                trainacc = round(accuracy_score(label1, clf.predict(HS.T)), 4)
                print('降维后train准确率：', trainacc)
                testacc = round(accuracy_score(label2, clf.predict(HT.T)), 4)
                print('降维后test准确率：', testacc)
                testacclist.append(testacc)

            print('testacclist:',testacclist)



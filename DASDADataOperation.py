import os
import time

import h5py
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import math


basepath = "F:/HAR_dataset/DailyandSportActivitiesDataset/TDF/"
feanamelist =['mavlist', 'varlist', 'rmslist', 'wllist', 'damvlist', 'dasdvlist', 'zclist', 'myoplist', 'wamplist', 'ssclist']


poepleidlist = [1, 2, 3, 4, 5, 6, 7, 8]
actionnamelist = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]



def getSimPeoData(pid=poepleidlist[0],fealist = feanamelist):
    # p1_FeaAndLabel.mat
    matdata = scio.loadmat(basepath+"p"+str(pid)+"_FeaAndLabel.mat")

    # print(str(matdata.keys())[11:-2].replace("'","").split(', ')[3:])
    # 'mavlist', 'varlist', 'rmslist', 'wllist', 'damvlist', 'dasdvlist', 'zclist', 'myoplist', 'wamplist', 'ssclist', 'labellist'

    dataarray = matdata[fealist[0]]

    for feai in feanamelist[1:]:
        tmp = matdata[feai]
        dataarray = np.concatenate((dataarray,tmp),axis=1)


    labelarray = matdata['labellist'][0]

    return dataarray,labelarray

def getMulPeoData(pidlist = poepleidlist[0],fealist = feanamelist):

    if len(pidlist) < 1:
        return None,None

    else:
        datalist,labellist = getSimPeoData(pidlist[0])

        if len(pidlist) > 1 :
            for pid in pidlist[1:]:
                # print('pid',pid)
                tmpdata,tmpalbel = getSimPeoData(pid,fealist)
                # print(tmpdata.shape)
                # print(tmpdata[-1])

                datalist = np.concatenate((datalist,tmpdata),axis=0)
                labellist = np.concatenate((labellist,tmpalbel),axis=0)

    return datalist,labellist

def getSimPeoDatabyactionlist(pid=poepleidlist[0],actionlist = actionnamelist,fealist = feanamelist):
    data, label = getSimPeoData(pid)

    adata = []
    alabel = []
    for il in range(len(label)):
        if label[il] in actionlist:
            adata.append(data[il])
            alabel.append(label[il])
    return np.array(adata),np.array(alabel)

def getMulPeoDatabyactionlist(pidlist=poepleidlist,actionlist = actionnamelist,fealist = feanamelist):

    if len(pidlist) < 1:
        return None,None

    else:
        datalist,labellist = getSimPeoDatabyactionlist(pidlist[0],actionlist)

        if len(pidlist) > 1 :
            for pid in pidlist[1:]:
                # print('pid',pid)
                tmpdata,tmpalbel = getSimPeoDatabyactionlist(pid,actionlist)
                # print(tmpdata.shape)
                # print(tmpdata[-1])

                datalist = np.concatenate((datalist,tmpdata),axis=0)
                labellist = np.concatenate((labellist,tmpalbel),axis=0)

    return datalist,labellist


def getSimPeoDataforTDandFD(pid = poepleidlist[0],actionidlist = actionnamelist):
    TFbasepath = 'F:/HAR_dataset/DailyandSportActivitiesDataset/TDF-FDF/'
    matdata = scio.loadmat(TFbasepath + "p" + str(pid) + "_FeaAndLabel.mat")

    print(matdata.keys())
    dataarray = matdata['featurelist']
    labelarray = matdata['labellist'][0]

    index = []

    for li in range(len(labelarray)):
        if labelarray[li] in actionidlist:
            index.append(li)

    dataarray = dataarray[index]
    labelarray = labelarray[index]


    return dataarray, labelarray

def getMulPeoDataforTDandFD(pidlist = poepleidlist[0],actionidlist = actionnamelist):
    if len(pidlist) < 1:
        return None,None

    else:
        datalist,labellist = getSimPeoDataforTDandFD(pidlist[0],actionidlist)

        if len(pidlist) > 1 :
            for pid in pidlist[1:]:
                # print('pid',pid)
                tmpdata,tmpalbel = getSimPeoDataforTDandFD(pid,actionidlist)
                # print(tmpdata.shape)
                # print(tmpdata[-1])

                datalist = np.concatenate((datalist,tmpdata),axis=0)
                labellist = np.concatenate((labellist,tmpalbel),axis=0)

    return datalist,labellist




if __name__=='__main__':
    pidlist = poepleidlist

    # data,label = getMulPeoData(pidlist)

    # data,label = getSimPeoData()

    # data, label = getMulPeoDataforTDandFD([1,2])

    # data, label = getSimPeoDatabyactionlist(1, [1,2,3,4,5,6,7,8,9,10])
    data, label = getMulPeoDatabyactionlist([1,2], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    print(data.shape)
    print(label.tolist())

    # lnum = 4
    # for i in range((lnum-1)*60,(lnum)*60):
    #     print(i,':',list(data[i]))
    # print(label.shape)
    # print(list(label))

    # data, label = getSimPeoDatabyactionlist(1, [1,2,3,4,5,6,7,8,9,10])
    # data, label = getSimPeoDatabyactionlist(1, [lnum])
    # print(data.shape)
    # for i in range(0,60):
    #     print(i, ':', list(data[i]))
    # print(data)






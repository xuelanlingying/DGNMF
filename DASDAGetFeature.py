'''
采样频率25hz，每个动作文件包含5秒，125位的传感器数据
'''

import os
import numpy as np
from feature_utils import *

basepath = "F:/HAR_dataset/DailyandSportActivitiesDataset/OriginalData/"

aid = 1
pid = 1
sid = 2

for pid in range(1,9):

    featurelist = []
    labellist = []
    mavlist = []
    varlist = []
    rmslist = []
    wllist = []
    damvlist = []
    dasdvlist = []
    zclist = []
    myoplist = []
    wamplist = []
    ssclist = []

# 遍历20个动作的  搜索一个人的文件
    for aid in range(1,20):
        dirlist = os.listdir(basepath+'a'+ str(aid).zfill(2)+"/p"+ str(pid)+"/")
        print(basepath+'a'+ str(aid).zfill(2)+"/p"+ str(pid)+"/")
        print("aid:",aid,dirlist)

        print(len(dirlist))


        for snum in dirlist:
            spath = basepath+'a'+ str(aid).zfill(2)+"/p"+ str(pid)+"/"+ snum
            print(spath)

            f = open(spath, 'r', encoding='utf-8')
            txtdata = f.readlines()
            # print(txtdata)
            f.close()
            datalist = []
            for line in txtdata:
                tmp = line[:-1].split(',')
                tmpdata = [float(i) for i in tmp]
                datalist.append(tmpdata)
            dataarray = np.array(datalist)


            print('dataarray.shape',dataarray.shape)

            print(dataarray)
            print(dataarray.shape)
            mav = featureMAV(dataarray)
            var = featureVAR(dataarray)
            rms = featureRMS(dataarray)
            wl = featureWL(dataarray)
            damv = featureDAMV(dataarray)
            dasdv = featureDASDV(dataarray)
            zc = featureZC(dataarray, 0)
            myop = featureMYOP(dataarray, 0)
            wamp = featureWAMP(dataarray)
            ssc = featureSSC(dataarray)

            mavlist.append(mav)
            varlist.append(var)
            rmslist.append(rms)
            wllist.append(wl)
            damvlist.append(damv)
            dasdvlist.append(dasdv)
            zclist.append(zc)
            myoplist.append(myop)
            wamplist.append(wamp)
            ssclist.append(ssc)

            labellist.append(aid)

    print(np.array(mavlist).shape)
    print(len(mavlist))
    print("dataarray.shape", np.array(dataarray).shape)
    print("mavlist.shape", np.array(mavlist).shape)
    print("damvlist.shape", np.array(damvlist).shape)
    print("ssclist.shape", np.array(ssclist).shape)
    print('labellist',len(labellist),labellist)


    savebasepath = 'F:/HAR_dataset/DailyandSportActivitiesDataset/TDF/'


    savefile = savebasepath+"p" + str(pid) + '_FeaAndLabel.mat'
    # print(savefile)
    # scio.savemat(savefile, {'mavlist': mavlist, 'varlist': varlist, 'rmslist': rmslist,
    #                         'wllist': wllist, 'damvlist': damvlist, 'dasdvlist': dasdvlist,
    #                         'zclist': zclist, 'myoplist': myoplist, 'wamplist': wamplist,
    #                         'ssclist': ssclist,'labellist':labellist})







    # testpath = basepath+'a'+ str(aid).zfill(2)+"/p"+ str(pid)+"/s"+ str(sid).zfill(2)+".txt"
    #
    # print(testpath)

# f = open(testpath, 'r', encoding='utf-8')
# txtdata = f.readlines()
# print(txtdata)
# f.close()
# print(len(txtdata))
# datalist = []
# for line in txtdata:
#     tmp = line[:-1].split(',')
#     tmpdata = [float(i) for i in tmp]
#     datalist.append(tmpdata)
#
# dataarray = np.array(datalist)
# print(dataarray.shape)
#
#
# aa = feature_utils.featureMAV(dataarray)
# print(aa.shape)








# -*- coding: utf-8 -*-
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri
import rpy2.robjects.packages as rpackages
import snf
from sklearn.cluster import SpectralClustering
from sklearn import metrics
import matplotlib.pyplot as plt
import csv
import pandas as pd
import numpy as np
import sklearn
robjects.r('rm(list=ls())')


def snf_Cluster(affinity_mydatGE, affinity_mydatME, affinity_mydatMI, pd_DataGE, pd_DataME, pd_DataMI, K=20, t=20,
                alpha=1.0):
    simGE = pd.DataFrame(affinity_mydatGE)
    simME = pd.DataFrame(affinity_mydatME)
    simMI = pd.DataFrame(affinity_mydatMI)

    # address GE
    for i in range(len(simGE.index)):
        for j in range(len(simGE.columns)):
            if pd_DataGE.iloc[i, j] == 0:
                simGE.iloc[i, j] = simGE.iloc[i, j] * 2
            elif pd_DataGE.iloc[i, j] == 1:
                simGE.iloc[i, j] = simGE.iloc[i, j]
            else:
                simGE.iloc[i, j] = simGE.iloc[i, j] / pd_DataGE.iloc[i, j]

    # address ME
    for i in range(len(simME.index)):
        for j in range(len(simME.columns)):
            if pd_DataME.iloc[i, j] == 0:
                simME.iloc[i, j] = simME.iloc[i, j] * 2
            elif pd_DataME.iloc[i, j] == 1:
                simME.iloc[i, j] = simME.iloc[i, j]
            else:
                simME.iloc[i, j] = simME.iloc[i, j] / pd_DataME.iloc[i, j]

    # address MI
    for i in range(len(simMI.index)):
        for j in range(len(simMI.columns)):
            if pd_DataMI.iloc[i, j] == 0:
                simMI.iloc[i, j] = simMI.iloc[i, j] * 2
            elif pd_DataMI.iloc[i, j] == 1:
                simMI.iloc[i, j] = simMI.iloc[i, j]
            else:
                simMI.iloc[i, j] = simMI.iloc[i, j] / pd_DataMI.iloc[i, j]
    fused_df = simGE
    for i in range(len(simMI.index)):
        for j in range(len(simMI.columns)):
            fused_df.iloc[i, j] = (simGE.iloc[i, j] + simME.iloc[i, j] + simMI.iloc[i, j]) / 3
    return fused_df


def snf_plus_altered_sim(*aff,pd_DataGE, pd_DataME, pd_DataMI, K=20, t=20, alpha=1.0):
    aff = snf.compute._check_SNF_inputs(aff)
    Wk = [0] * len(aff)
    Wsum = np.zeros(aff[0].shape)

    # get number of modalities informing each subject x subject affinity
    n_aff = len(aff) - np.sum([np.isnan(a) for a in aff], axis=0)

    for n, mat in enumerate(aff):
        # normalize affinity matrix based on strength of edges
        mat = mat / np.nansum(mat, axis=1, keepdims=True)
        aff[n] = sklearn.utils.validation.check_symmetric(mat, raise_warning=False)  # sklearn.utils.validation.check_symmetric
        # # apply KNN threshold to normalized affinity matrix
        # Wk[n] = snf.compute._find_dominate_set(aff[n], int(K))
        temp_aff = aff
        temp_DF = pd.DataFrame(temp_aff[n])
        if n == 0:
            for i in range(len(temp_DF.index)):
                for j in range(len(temp_DF.columns)):
                    if pd_DataGE.iloc[i, j] == 0:
                        temp_DF.iloc[i, j] = temp_DF.iloc[i, j] * 2
                    elif pd_DataGE.iloc[i, j] == 1:
                        temp_DF.iloc[i, j] = temp_DF.iloc[i, j]
                    else:
                        temp_DF.iloc[i, j] = temp_DF.iloc[i, j] / pd_DataGE.iloc[i, j]
            temp_aff[n] = temp_DF.to_numpy()
            Wk[n] = temp_aff[n]
        if n == 1:
            for i in range(len(temp_DF.index)):
                for j in range(len(temp_DF.columns)):
                    if pd_DataME.iloc[i, j] == 0:
                        temp_DF.iloc[i, j] = temp_DF.iloc[i, j] * 2
                    elif pd_DataME.iloc[i, j] == 1:
                        temp_DF.iloc[i, j] = temp_DF.iloc[i, j]
                    else:
                        temp_DF.iloc[i, j] = temp_DF.iloc[i, j] / pd_DataME.iloc[i, j]
            temp_aff[n] = temp_DF.to_numpy()
            Wk[n] = temp_aff[n]
        if n == 2:
            for i in range(len(temp_DF.index)):
                for j in range(len(temp_DF.columns)):
                    if pd_DataMI.iloc[i, j] == 0:
                        temp_DF.iloc[i, j] = temp_DF.iloc[i, j] * 2
                    elif pd_DataMI.iloc[i, j] == 1:
                        temp_DF.iloc[i, j] = temp_DF.iloc[i, j]
                    else:
                        temp_DF.iloc[i, j] = temp_DF.iloc[i, j] / pd_DataMI.iloc[i, j]
            temp_aff[n] = temp_DF.to_numpy()
            Wk[n] = temp_aff[n]



    # take sum of all normalized (not thresholded) affinity matrices
    Wsum = np.nansum(aff, axis=0)

    for iteration in range(t):
        for n, mat in enumerate(aff):
            # temporarily convert nans to 0 to avoid propagation errors
            nzW = np.nan_to_num(Wk[n])
            aw = np.nan_to_num(mat)
            # propagate `Wsum` through masked affinity matrix (`nzW`)
            aff0 = nzW @ (Wsum - aw) @ nzW.T / (n_aff - 1)  # TODO: / by 0
            # ensure diagonal retains highest similarity
            aff[n] = snf.compute._B0_normalized(aff0, alpha=alpha)

        # compute updated sum of normalized affinity matrices
        Wsum = np.nansum(aff, axis=0)

    # all entries of `aff` should be identical after the fusion procedure
    # dividing by len(aff) is hypothetically equivalent to selecting one
    # however, if fusion didn't converge then this is just average of `aff`
    W = Wsum / len(aff)

    # normalize fused matrix and update diagonal similarity
    W = W / np.nansum(W, axis=1, keepdims=True)  # TODO: / by NaN
    W = (W + W.T + np.eye(len(W))) / 2

    return W


def full_snf_plus_cluster(*aff,pd_DataGE, pd_DataME, pd_DataMI, K=20, t=20, alpha=1.0):
    aff = snf.compute._check_SNF_inputs(aff)
    Wk = [0] * len(aff)
    knn_Wk = [0] * len(aff)
    Wsum = np.zeros(aff[0].shape)

    # get number of modalities informing each subject x subject affinity
    n_aff = len(aff) - np.sum([np.isnan(a) for a in aff], axis=0)

    for n, mat in enumerate(aff):
        temp_aff = aff
        knn_aff = aff
        # normalize affinity matrix based on strength of edges
        mat = mat / np.nansum(mat, axis=1, keepdims=True)
        aff[n] = sklearn.utils.validation.check_symmetric(mat, raise_warning=False)  # sklearn.utils.validation.check_symmetric
        # # apply KNN threshold to normalized affinity matrix
        knn_Wk[n] = snf.compute._find_dominate_set(knn_aff[n], int(K))
        temp_DF = pd.DataFrame(temp_aff[n])
        if n == 0:
            for i in range(len(temp_DF.index)):
                for j in range(len(temp_DF.columns)):
                    if pd_DataGE.iloc[i, j] == 0:
                        temp_DF.iloc[i, j] = temp_DF.iloc[i, j] * 2
                    elif pd_DataGE.iloc[i, j] == 1:
                        temp_DF.iloc[i, j] = temp_DF.iloc[i, j]
                    else:
                        temp_DF.iloc[i, j] = temp_DF.iloc[i, j] / pd_DataGE.iloc[i, j]
            temp_aff[n] = temp_DF.to_numpy()
            Wk[n] = temp_aff[n]
        if n == 1:
            for i in range(len(temp_DF.index)):
                for j in range(len(temp_DF.columns)):
                    if pd_DataME.iloc[i, j] == 0:
                        temp_DF.iloc[i, j] = temp_DF.iloc[i, j] * 2
                    elif pd_DataME.iloc[i, j] == 1:
                        temp_DF.iloc[i, j] = temp_DF.iloc[i, j]
                    else:
                        temp_DF.iloc[i, j] = temp_DF.iloc[i, j] / pd_DataME.iloc[i, j]
            temp_aff[n] = temp_DF.to_numpy()
            Wk[n] = temp_aff[n]
        if n == 2:
            for i in range(len(temp_DF.index)):
                for j in range(len(temp_DF.columns)):
                    if pd_DataMI.iloc[i, j] == 0:
                        temp_DF.iloc[i, j] = temp_DF.iloc[i, j] * 2
                    elif pd_DataMI.iloc[i, j] == 1:
                        temp_DF.iloc[i, j] = temp_DF.iloc[i, j]
                    else:
                        temp_DF.iloc[i, j] = temp_DF.iloc[i, j] / pd_DataMI.iloc[i, j]
            temp_aff[n] = temp_DF.to_numpy()
            Wk[n] = temp_aff[n]



    # take sum of all normalized (not thresholded) affinity matrices
    Wsum = np.nansum(aff, axis=0)
    knn_Wsum = np.nansum(knn_aff, axis=0)

    for iteration in range(t):
        for n, mat in enumerate(aff):
            # temporarily convert nans to 0 to avoid propagation errors
            nzW = np.nan_to_num(Wk[n])
            knn_nzW = np.nan_to_num(knn_Wk[n])
            aw = np.nan_to_num(mat)
            # propagate `Wsum` through masked affinity matrix (`nzW`)
            aff0 = nzW @ (Wsum - aw) @ nzW.T / (n_aff - 1)  # TODO: / by 0
            # aff0 = knn_nzW @ (knn_Wsum - aw) @ knn_nzW.T / (n_aff - 1)  # TODO: / by 0
            # ensure diagonal retains highest similarity
            aff[n] = snf.compute._B0_normalized(aff0, alpha=alpha)

        # compute updated sum of normalized affinity matrices
        Wsum = np.nansum(aff, axis=0)

    # knn plus process
    for n, mat in enumerate(knn_aff):
        # normalize affinity matrix based on strength of edges
        mat = mat / np.nansum(mat, axis=1, keepdims=True)
        knn_aff[n] = sklearn.utils.validation.check_symmetric(mat, raise_warning=False)  # sklearn.utils.validation.check_symmetric
        # apply KNN threshold to normalized affinity matrix
        knn_Wk[n] = snf.compute._find_dominate_set(knn_aff[n], int(K))

    # take sum of all normalized (not thresholded) affinity matrices
    knn_Wsum = np.nansum(knn_aff, axis=0)

    # plus knn to iteration
    for iteration in range(t):
        for n, mat in enumerate(aff):
            # temporarily convert nans to 0 to avoid propagation errors
            # nzW = np.nan_to_num(Wk[n])
            knn_nzW = np.nan_to_num(knn_Wk[n])
            aw = np.nan_to_num(mat)
            # propagate `Wsum` through masked affinity matrix (`nzW`)
            # aff0 = nzW @ (Wsum - aw) @ nzW.T / (n_aff - 1)  # TODO: / by 0
            aff0 = knn_nzW @ (knn_Wsum - aw) @ knn_nzW.T / (n_aff - 1)  # TODO: / by 0
            # ensure diagonal retains highest similarity
            aff[n] = snf.compute._B0_normalized(aff0, alpha=alpha)

        # compute updated sum of normalized affinity matrices
        Wsum = np.nansum(aff, axis=0)

    # all entries of `aff` should be identical after the fusion procedure
    # dividing by len(aff) is hypothetically equivalent to selecting one
    # however, if fusion didn't converge then this is just average of `aff`
    W = Wsum / len(aff)

    # normalize fused matrix and update diagonal similarity
    W = W / np.nansum(W, axis=1, keepdims=True)  # TODO: / by NaN
    W = (W + W.T + np.eye(len(W))) / 2

    return W


def load_data(filename):
    """
    载入数据
    :param filename: 文件名
    :return:
    """
    data = pd.read_csv(filename)
    # data = np.loadtxt(filename, delimiter='\t')
    return data


def plotRes(data, clusterResult, clusterNum):
    """
    结果可似化
    :param data:  样本集
    :param clusterResult: 聚类结果
    :param clusterNum: 聚类个数
    :return:
    """
    n = len(data)
    scatterColors = ['black', 'blue', 'green', 'yellow', 'red', 'purple', 'orange']
    for i in range(clusterNum):
        color = scatterColors[i % len(scatterColors)]
        x1 = [];
        y1 = []
        for j in range(n):
            if clusterResult[j] == i:
                x1.append(data.iloc[j, 0])
                y1.append(data.iloc[j, 1])
        plt.scatter(x1, y1, c=color, marker='+')
    # plt.show()


numpy2ri.activate()  # From numpy to rpy2

# import R packages
base = rpackages.importr('base')
utils = rpackages.importr('utils')
stats = rpackages.importr('stats')
PINS = rpackages.importr('PINS')
rpackages.importr('survival')
rpackages.importr('flexclust')

# import R scripts
robjects.r('source("R_scripts/BasicFunctions.R")')

# initialize and set some parameters
robjects.r('''
    dataPath="Projects/SubTyping/DataGEO/"
    resultPath="result/PINSResult/"
    Kmax=10  # 10
    iter=200  # 200
    kmIter=200  # 200
    noisePercent="med"
''')

datasets = robjects.r(
    'datasets=c("AML2004", "GSE10245", "GSE15061", "GSE19188","GSE14924", "Brain2002", "GSE43580","Lung2001")')


number = robjects.r('c(0.25, 0.3, 0.35,  0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75)')
for dataset in datasets:
    for percentile in number:
        robjects.r('''
            set.seed(1)
            print(paste("SUBTYPING ", "{}", sep=""))
            load(paste(dataPath,"{}",".RData",sep=""))
        '''.format(dataset, dataset))

        robjects.r('''
            data=(get(paste("gene_","{}",sep="")))
            group=get(paste("group_","{}",sep=""))
        '''.format(dataset, dataset))
        robjects.r('''
            #remove healthy samples
            data=data[!rownames(data)%in%rownames(group)[group[,2]=="healthy"],]
            group=group[!rownames(group)%in%rownames(group)[group[,2]=="healthy"],]
        ''')
        robjects.r('''
            #result = PerturbationClustering(data=data, Kmax = Kmax, noisePercent = 0.5, iter = iter, kmIter = kmIter)

            result = PerturbationClustering(data=data, Kmax = Kmax, noisePercent = {}, iter = iter)
            message("{}", ": {} ", {}, "; RI ", randIndex(result$groups, group$Group, correct = FALSE), "; ARI ", randIndex(result$groups, group$Group, correct = TRUE))
            resultFile=paste(resultPath, "{}", "/", "PINS_", "{}","_percentile", {}, ".RData" ,sep="")
            print(resultFile) 
            save(result, file=resultFile)
        '''.format(percentile, dataset, percentile, percentile, dataset, dataset, percentile))
# show the results
for dataset in datasets:
    robjects.r('ret = NULL')
    for percentile in number:
        robjects.r('''
            load(paste(dataPath,"{}",".RData",sep=""))

            data=(get(paste("gene_","{}",sep="")))
            group=get(paste("group_","{}",sep=""))
        '''.format(dataset, dataset, dataset))
        robjects.r('''
            #remove healthy samples
            data=data[!rownames(data)%in%rownames(group)[group[,2]=="healthy"],]
            group=group[!rownames(group)%in%rownames(group)[group[,2]=="healthy"],]
        ''')
        robjects.r('''
            #result = PerturbationClustering(data=data, Kmax = Kmax, noisePercent = 0.5, iter = iter, kmIter = kmIter)

            resultFile=paste(resultPath, "{}", "/", "PINS_", "{}","_percentile", {}, ".RData" ,sep="") 
            load(file=resultFile)
            ret=c(ret,round(randIndex(result$groups, group$Group, correct = TRUE), digits = 2))
        '''.format(dataset, dataset, percentile))
    robjects.r('message("{}", ": ", paste0(ret, collapse = "&"))'.format(dataset))


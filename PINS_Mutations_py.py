# -*- coding: utf-8 -*-
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri
import rpy2.robjects.packages as rpackages
import snf
import math
from sklearn.cluster import SpectralClustering
from sklearn import metrics
import matplotlib.pyplot as plt
import csv
import pandas as pd
import numpy as np
import sklearn
robjects.r('rm(list=ls())')

numpy2ri.activate()  # From numpy to rpy2

# import R packages
rpackages.importr('base')
rpackages.importr('utils')
rpackages.importr('stats')
rpackages.importr('PINS')
rpackages.importr('survival')

def snf_plus_altered_sim(*aff,pd_DataGE, pd_DataCNV, K=20, t=20, alpha=1.0):
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
                    if pd_DataCNV.iloc[i, j] == 0:
                        temp_DF.iloc[i, j] = temp_DF.iloc[i, j] * 2
                    elif pd_DataCNV.iloc[i, j] == 1:
                        temp_DF.iloc[i, j] = temp_DF.iloc[i, j]
                    else:
                        temp_DF.iloc[i, j] = temp_DF.iloc[i, j] / pd_DataCNV.iloc[i, j]
            temp_aff[n] = temp_DF.to_numpy()
            Wk[n] = temp_aff[n]
        # if n == 2:
        #     for i in range(len(temp_DF.index)):
        #         for j in range(len(temp_DF.columns)):
        #             if pd_DataMI.iloc[i, j] == 0:
        #                 temp_DF.iloc[i, j] = temp_DF.iloc[i, j] * 2
        #             elif pd_DataMI.iloc[i, j] == 1:
        #                 temp_DF.iloc[i, j] = temp_DF.iloc[i, j]
        #             else:
        #                 temp_DF.iloc[i, j] = temp_DF.iloc[i, j] / pd_DataMI.iloc[i, j]
        #     temp_aff[n] = temp_DF.to_numpy()
        #     Wk[n] = temp_aff[n]



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


# import and define some necessary information
robjects.r('''
    source("R_scripts/BasicFunctions.R")
    dataPath="Projects/SubTyping/DataTCGA/"
    resultPath="result/DataTCGA/"
    pdfPath="result/Plots/Figures/"
    Kmax=10  # it should be 10
    iter=1  # it should be 200
    kmIter=2  # it should be 200
''')
robjects.r('''
    dataset="GBM"
    resultFile=paste(resultPath, dataset, "/", "PINS_", dataset, ".RData" ,sep="")
    load(resultFile)
    mutFile=paste(dataPath, dataset, "/", dataset, "_Mutation.RData", sep="")
    load(mutFile)
    groups=result$groups
    groups2=result$groups2
    group1Mut=apply(mydatMut[groups2=="1-2",], 2, FUN = sum)
    group2Mut=apply(mydatMut[groups2=="2",], 2, FUN = sum)
    
    pdfFile=paste(pdfPath,"GBM_Mutation.pdf",sep="")
    print(pdfFile)
    pdf(pdfFile, height=5, width=7)
    par(tcl=0.3,mgp=c(1.7,0.4,0),mar=c(3,3,3,1.2), xpd=T)
    plot(group1Mut, group2Mut, pch=8, cex.lab=1.3, cex.axis=1.2, cex.main=1.5, xlab="Group 1-2 (short-term survival)", ylab="Group 2 (long-term survival)")
    title(paste("Mutations in ", dataset, " subtypes", sep=""), cex.main=1.5)
    for (mut in c("IDH1","ATRX","TP53")) {
        x=group1Mut[mut]
        y=group2Mut[mut]
        text(x+1, y, mut, col="blue")
    }
    for (mut in c("PIK3R1", "PTEN","TTN","EGFR")) {
        x=group1Mut[mut]
        y=group2Mut[mut]
        text(x, y+0.5, mut, col="blue")
    }
    dev.off()
    
    dataset="KIRC"
    resultFile=paste(resultPath, dataset, "/", "PINS_", dataset, ".RData" ,sep="")
    print(resultFile)
    load(resultFile)
    
    mutFile=paste(dataPath,dataset, "/", dataset, "_Mutation.RData", sep="")
    print(mutFile)
    load(mutFile)
    groups=result$groups
    groups2=result$groups2
    group2Mut=apply(mydatMut[groups2%in%c("1-2","2","3"),], 2, FUN = sum)
    group1Mut=apply(mydatMut[groups2%in%c("1-1"),], 2, FUN = sum)
    
    # pdfFile=paste(pdfPath,"KIRC_Mutation.pdf",sep="")
    # pdf(pdfFile, height=5, width=7)
    par(tcl=0.3,mgp=c(1.7,0.4,0),mar=c(3,3,3,1.2), xpd=T)
    plot(group1Mut, group2Mut, pch=8, cex.lab=1.3, cex.axis=1.2, cex.main=1.5, xlab="Group 1-1", ylab="Group 1-2, 2, and 3")
    title(paste("Mutations in ", dataset, " subtypes", sep=""), cex.main=1.5)
    for (mut in c("VHL")) {
        x=group1Mut[mut]
        y=group2Mut[mut]
        text(x, y+0.5, mut, col="blue")
    }
    
    dataset="LAML"
    resultFile=paste(resultPath, dataset, "/", "PINS_", dataset, ".RData" ,sep="")
    print(resultFile)
    load(resultFile)
    
    mutFile=paste(dataPath,dataset, "/", dataset, "_Mutation.RData", sep="")
    print(mutFile)
    load(mutFile)
    mydatMut=mydatMut[,!colnames(mydatMut)%in%c("Unknown")]
    groups=result$groups
    groups2=result$groups2
    group2Mut=apply(mydatMut[groups2%in%c("1","2","3"),], 2, FUN = sum)
    group1Mut=apply(mydatMut[groups2%in%c("4"),], 2, FUN = sum)
    
    pdfFile=paste(pdfPath,"LAML_Mutation.pdf",sep="")
    print(pdfFile)
    pdf(pdfFile, height=5, width=7)
    par(tcl=0.3,mgp=c(1.7,0.4,0),mar=c(3,3,3,1.2), xpd=T)
    plot(group1Mut, group2Mut, pch=8, cex.lab=1.3, cex.axis=1.2, cex.main=1.5, xlab="Group 4 (short-term survival)", ylab="Group 1, 2, and 3 (long-term survival)")
    title(paste("Mutations in ", dataset, " subtypes", sep=""), cex.main=1.5)
    for (mut in c("DNMT3A","FLT3","NPM1")) {
        x=group1Mut[mut]
        y=group2Mut[mut]
        text(x + 0.9, y, mut, col="blue")
    }
    for (mut in c("RUNX1","TP53")) {
        x=group1Mut[mut]
        y=group2Mut[mut]
        text(x, y+3, mut, col="blue")
    }
    dev.off()
    
    
    
    
''')
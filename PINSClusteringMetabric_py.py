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
    dataPath="Projects/SubTyping/METABRIC/"
    resultPath="result/METABRIC/"
    Kmax=10  # it should be 10
    iter=200  # it should be 200
    kmIter=200  # it should be 200
''')
p_values = {'METABRIC_discovery': 1, 'METABRIC_validation': 1}
datasets = robjects.r('datasets=c("METABRIC_discovery","METABRIC_validation")')
SC_values = []
AH_values = []
Scores = []  # Silhouette Coefficient, if it only has 1 class, it will be 0
Scores1 = []  # AH Coefficient, if it only has 1 class, it will be 0
for dataset in datasets:
    robjects.r('''
    set.seed(1)
    file=paste(dataPath, "{}", ".RData" ,sep="")
    load(file)
    
    t1=Sys.time()

    mydatCNV=t(mydatCNV[,-c(1:5)])

    patients=rownames(survival)
    patients=intersect(patients,rownames(mydatGE))
    patients=intersect(patients,rownames(mydatCNV))

    mydatGE=mydatGE[patients,]
    mydatCNV=mydatCNV[patients,]

    dataList <- list(mydatGE, mydatCNV) 
    names(dataList) = c("GE", "CNV")
    
    rowGE<-row.names(mydatGE)
    rowCNV<-row.names(mydatCNV)
    # result=SubtypingOmicsData(dataList = dataList, Kmax = Kmax, noisePercent = "med", iter = iter)
    '''.format(dataset))

    robjects.r('''
        agreementCutoff=0.5
        dataTypeResult <- list()
        for (i in 1:length(dataList)) {
            message(paste("Data type: ", i, sep=""))
            dataTypeResult[[i]] <- PerturbationClustering(data=dataList[[i]], Kmax=Kmax, noisePercent="med", iter=iter, kmIter=kmIter)
            # break
        }
        origGE=dataTypeResult[[1]]$origS[[dataTypeResult[[1]]$k]]
        PWGE = dataTypeResult[[1]]$origS[[dataTypeResult[[1]]$k]]
        pertGE=dataTypeResult[[1]]$pertS[[dataTypeResult[[1]]$k]]

        origCNV=dataTypeResult[[2]]$origS[[dataTypeResult[[1]]$k]]
        PWCNV = dataTypeResult[[2]]$origS[[dataTypeResult[[1]]$k]]
        pertCNV=dataTypeResult[[2]]$pertS[[dataTypeResult[[1]]$k]]
        
        # origME=dataTypeResult[[2]]$origS[[dataTypeResult[[1]]$k]]
        # PWME = dataTypeResult[[2]]$origS[[dataTypeResult[[1]]$k]]
        # pertME=dataTypeResult[[2]]$pertS[[dataTypeResult[[1]]$k]]

        # origMI=dataTypeResult[[3]]$origS[[dataTypeResult[[1]]$k]]
        # PWMI = dataTypeResult[[3]]$origS[[dataTypeResult[[1]]$k]]
        # pertMI=dataTypeResult[[3]]$pertS[[dataTypeResult[[1]]$k]]

        # complete code to per multi-omics data
        orig=dataTypeResult[[1]]$origS[[dataTypeResult[[1]]$k]]
        PW = dataTypeResult[[1]]$origS[[dataTypeResult[[1]]$k]]
        pert=dataTypeResult[[1]]$pertS[[dataTypeResult[[1]]$k]]
        for (i in 2:length(dataTypeResult)) {
            orig=orig+dataTypeResult[[i]]$origS[[dataTypeResult[[i]]$k]]
            PW = PW * dataTypeResult[[i]]$origS[[dataTypeResult[[i]]$k]]
            pert=pert+dataTypeResult[[i]]$pertS[[dataTypeResult[[i]]$k]]
        }
        orig=orig/length(dataTypeResult)
        pert=pert/length(dataTypeResult)

        groupings <- list()

        for (i in 1:length(dataTypeResult)) {
            k=dataTypeResult[[i]]$k
            groupings[[i]] <- kmeansSS(dataTypeResult[[i]]$pertS[[k]],k)$cluster
        }

        hierPart <- clusterUsingHierarchical(orig = orig, pert = pert, Kmax=Kmax, groupings=groupings)
        pamPart <- clusterUsingPAM(orig = orig, pert = pert, Kmax=Kmax, groupings=groupings)

        hcP <- hclust(as.dist(1-pert), method="average")  
        groupRP <- dynamicTreeCut::cutreeDynamic(hcP, distM=1-pert, cutHeight = 0.9*max(hcP$height))
        if (length(which(groupRP==0))>0) groupRP=groupRP+1
        names(groupRP) <- rownames(orig)
        kmRP <- structure(list(cluster=groupRP), class="kmeans")

        agreement = (sum(orig==0) + sum(orig==1)-nrow(orig))/(nrow(orig)^2-nrow(orig))
    ''')
    robjects.r('''
        hcW <- hclust(dist(PW))
        maxK = min(Kmax, dim(unique(PW,MARGIN=2))[2])
        maxHeight = findMaxHeight(hcW, maxK = maxK)
        groups <- cutree(hcW, maxHeight)
        message("Check if we can proceed to stage II")
        groups2 = groups
    ''')
    robjects.r('''

        if (agreement>agreementCutoff) {
            hcW <- hclust(dist(PW))
            maxK = min(Kmax, dim(unique(PW,MARGIN=2))[2])
            maxHeight = findMaxHeight(hcW, maxK = maxK)
            groups <- cutree(hcW, maxHeight)

            message("Check if we can proceed to stage II")
            groups2 = groups
            for (g in sort(unique(groups))) {
                miniGroup <- names(groups[groups==g])
                # this is just to make sure we don't split a group that is already very small
                if (length(miniGroup) > 30) {
                    tmpList <- list()
                    for (i in 1:length(dataList)) {
                        tmpList[[i]] <- PerturbationClustering(data=dataList[[i]][miniGroup,], Kmax=Kmax/2, noisePercent="med", iter=iter, kmIter=kmIter)
                    }
    
                    origM=tmpList[[1]]$origS[[tmpList[[1]]$k]]
                    PWM = tmpList[[1]]$origS[[tmpList[[1]]$k]]
                    pertM=tmpList[[1]]$pertS[[tmpList[[1]]$k]]
                    for (i in 2:length(tmpList)) {
                        origM=origM+tmpList[[i]]$origS[[tmpList[[i]]$k]]
                        PWM = PWM * tmpList[[i]]$origS[[tmpList[[i]]$k]]
                        pertM=pertM+tmpList[[i]]$pertS[[tmpList[[i]]$k]]
                    }
                    origM=origM/length(tmpList)
                    pertM=pertM/length(tmpList)
    
                    agreementM = (sum(origM==0)+sum(origM==1)-nrow(origM))/(nrow(origM)^2-nrow(origM))
                    if (agreementM >= agreementCutoff) {
                        hcPWM <- hclust(dist(PWM))
                        maxK = min(Kmax/2, dim(unique(PWM,MARGIN=2))[2]-1)        
                        maxHeightM = findMaxHeight(hcPWM, maxK=maxK)
                        groupsM <- cutree(hcPWM, maxHeightM)
                        groupsM <- paste(g, groupsM, sep="-")
                        groups2[miniGroup] <- groupsM
                    } 
              }
            }

        } else {
            if (hierPart$diff[hierPart$k]<pamPart$diff[pamPart$k]) {
                km <- hierPart$km
            } else {
                km <- pamPart$km
            }
            l1 = groupings; l1[[length(l1)+1]]=kmRP$cluster
            l2 = groupings; l2[[length(l2)+1]]=km$cluster
            if (clusterAgreement(l1, nrow(orig)) > clusterAgreement(l2, nrow(orig)))
                km <- kmRP

            groups <- km$cluster
            message("Check if can proceed to stage II")
            groups2 <- groups
            normalizedEntropy=entropy(table(groups))/log(length(unique(groups)),exp(1))
            if (normalizedEntropy<0.5) {
                for (g in sort(unique(groups))) {
                    miniGroup <- names(groups[groups==g])
                    #this is just to make sure we don't split a group that is already very small
                    if (length(miniGroup) > 30) {
                        gapCount=0
                        for (i in 1:length(dataList)) {
                            tmp=clusGap(prcomp(dataList[[i]][miniGroup,])$x,FUN=kmeans, K.max=Kmax/2, B=100)
                            if (maxSE(tmp$Tab[,"gap"], tmp$Tab[,"SE.sim"], method="firstSEmax")>1) gapCount=gapCount+1
                        }
                        if (length(miniGroup) > 30 && gapCount > length(dataList)/2) {
                            tmpList <- list()
                            for (i in 1:length(dataList)) {
                                tmpList[[i]] <- PerturbationClustering(data=dataList[[i]][miniGroup,], Kmax=Kmax/2, noisePercent="med", iter=iter, kmIter=kmIter)
                            }

                            origM=tmpList[[1]]$origS[[tmpList[[1]]$k]]
                            PWM = tmpList[[1]]$origS[[tmpList[[1]]$k]]
                            pertM=tmpList[[1]]$pertS[[tmpList[[1]]$k]]
                            for (i in 2:length(tmpList)) {
                                origM=origM+tmpList[[i]]$origS[[tmpList[[i]]$k]]
                                PWM = PWM * tmpList[[i]]$origS[[tmpList[[i]]$k]]
                                pertM=pertM+tmpList[[i]]$pertS[[tmpList[[i]]$k]]
                            }
                            origM=origM/length(tmpList)
                            pertM=pertM/length(tmpList)

                            hcPWM <- hclust(dist(PWM))
                            maxK = min(Kmax/2, dim(unique(PWM,MARGIN=2))[2]-1)        
                            maxHeightM = findMaxHeight(hcPWM, maxK)
                            groupsM <- cutree(hcPWM, maxHeightM)
                            groupsM <- paste(g, groupsM, sep="-")
                            groups2[miniGroup] <- groupsM
                        }
                    }
                }
            }
        }

        result <- list()
        result$groups <- groups
        result$groups2 <- groups2
        result$dataTypeResult <- dataTypeResult
        result$hierPart  <- hierPart
        result$pamPart <- pamPart
        result$kmRP <- kmRP
        result

        t2=Sys.time()
        print(t2-t1)
    ''')
    robjects.r('resultFile=paste(resultPath, "{}", "/", "PINS_", "{}", ".RData" ,sep="")'.format(dataset, dataset))
    # UnicodeDecodeError: 'utf-8' codec can't decode byte 0xb1 in position 78: invalid start byte
    robjects.r('save(Kmax, dataList, survival, clinical, result, t1, t2, file=resultFile)')

for dataset in datasets:
    robjects.r('pdfFile=paste("result/METABRIC/", "{}", "/", "PINS_", "{}", ".pdf" ,sep="")'.format(dataset, dataset))
    robjects.r('''
        
        pdf(pdfFile)
        groups = result$groups
        groups2=result$groups2
        survi=survivalDFS[patients,]
        
        # gene expression
        groups=as.factor(result$dataTypeResult[[1]]$groups)
        coxp <- round(summary(coxph(Surv(time = Survival, event = Death) ~ groups, data = survi, ties="exact"))$sctest[3],10)
        CI=round(summary(coxph(Surv(time = Survival, event = Death) ~ groups, data = survi))$concordance[1],3)
        mfit <- survfit(Surv(Survival, Death == 1) ~ groups, data = survi)
        plot(mfit, col=unique(groups), main = paste("DFS survival curves for gene expression of ","{}", " (PINS)", sep=""), xlab = "Days", ylab="Survival", lwd=2)
        legend("topright", legend = paste("Cox p-value:", coxp, ", CI=", CI, sep=""))
        
        # CNV
        groups=as.factor(result$dataTypeResult[[2]]$groups)
        coxp <- round(summary(coxph(Surv(time = Survival, event = Death) ~ groups, data = survi, ties="exact"))$sctest[3],10)
        CI=round(summary(coxph(Surv(time = Survival, event = Death) ~ groups, data = survi))$concordance[1],3)
        mfit <- survfit(Surv(Survival, Death == 1) ~ groups, data = survi)
        plot(mfit, col=unique(groups), main = paste("DFS survival curves for CNV of ", "{}", " (PINS)", sep=""), xlab = "Days", ylab="Survival", lwd=2)
        legend("topright", legend = paste("Cox p-value:", coxp, ", CI=", CI, sep=""))
        
        # level 1
        groups = as.factor(result$groups)
        coxp <- round(summary(coxph(Surv(time = Survival, event = Death) ~ groups, data = survi, ties="exact"))$sctest[3],10)
        CI=round(summary(coxph(Surv(time = Survival, event = Death) ~ groups, data = survi))$concordance[1],3)
        mfit <- survfit(Surv(Survival, Death == 1) ~ groups, data = survi)
        plot(mfit, col=unique(groups), main = paste("DFS survival curves for ", "{}", ", level 1 (PINS)", sep=""), xlab = "Days", ylab="Survival", lwd=2)
        legend("topright", legend = paste("Cox p-value:", coxp, ", CI=", CI, sep=""))
    
        # level 2
        groups = as.factor(result$groups2)
        coxp <- round(summary(coxph(Surv(time = Survival, event = Death) ~ groups, data = survi, ties="exact"))$sctest[3],10)
        CI=round(summary(coxph(Surv(time = Survival, event = Death) ~ groups, data = survi))$concordance[1],3)
        mfit <- survfit(Surv(Survival, Death == 1) ~ groups, data = survi)
        plot(mfit, col=unique(groups), main = paste("DFS survival curves for ", "{}", ", level 2 (PINS)", sep=""), xlab = "Days", ylab="Survival", lwd=2)
        legend("topright", legend = paste("Cox p-value:", coxp, ", CI=", CI, sep=""))
        
        survi=survival[patients,]
        # gene expression
        groups=as.factor(result$dataTypeResult[[1]]$groups)
        coxp <- round(summary(coxph(Surv(time = Survival, event = Death) ~ groups, data = survi, ties="exact"))$sctest[3],10)
        CI=round(summary(coxph(Surv(time = Survival, event = Death) ~ groups, data = survi))$concordance[1],3)
        mfit <- survfit(Surv(Survival, Death == 1) ~ groups, data = survi)
        plot(mfit, col=unique(groups), main = paste("Overall survival curves for gene expression of ","{}", " (PINS)", sep=""), xlab = "Days", ylab="Survival", lwd=2)
        legend("topright", legend = paste("Cox p-value:", coxp, ", CI=", CI, sep=""))
    
        # CNV
        groups=as.factor(result$dataTypeResult[[2]]$groups)
        coxp <- round(summary(coxph(Surv(time = Survival, event = Death) ~ groups, data = survi, ties="exact"))$sctest[3],10)
        CI=round(summary(coxph(Surv(time = Survival, event = Death) ~ groups, data = survi))$concordance[1],3)
        mfit <- survfit(Surv(Survival, Death == 1) ~ groups, data = survi)
        plot(mfit, col=unique(groups), main = paste("Overall survival curves for CNV of ", "{}", " (PINS)", sep=""), xlab = "Days", ylab="Survival", lwd=2)
        legend("topright", legend = paste("Cox p-value:", coxp, ", CI=", CI, sep=""))
    
        # level 1
        groups = as.factor(result$groups)
        coxp <- round(summary(coxph(Surv(time = Survival, event = Death) ~ groups, data = survi, ties="exact"))$sctest[3],10)
        CI=round(summary(coxph(Surv(time = Survival, event = Death) ~ groups, data = survi))$concordance[1],3)
        mfit <- survfit(Surv(Survival, Death == 1) ~ groups, data = survi)
        plot(mfit, col=unique(groups), main = paste("Overall survival curves for ", "{}", ", level 1 (PINS)", sep=""), xlab = "Days", ylab="Survival", lwd=2)
        legend("topright", legend = paste("Cox p-value:", coxp, ", CI=", CI, sep=""))
    
        # level 2
        groups = as.factor(result$groups2)
        coxp <- round(summary(coxph(Surv(time = Survival, event = Death) ~ groups, data = survi, ties="exact"))$sctest[3],10)
        CI=round(summary(coxph(Surv(time = Survival, event = Death) ~ groups, data = survi))$concordance[1],3)
        mfit <- survfit(Surv(Survival, Death == 1) ~ groups, data = survi)
        plot(mfit, col=unique(groups), main = paste("Overall survival curves for ", "{}", ", level 2 (PINS)", sep=""), xlab = "Days", ylab="Survival", lwd=2)
        legend("topright", legend = paste("Cox p-value:", coxp, ", CI=", CI, sep=""))
        
    '''.format(dataset, dataset, dataset, dataset, dataset, dataset, dataset, dataset))

    # obtain pertS connective matrices data
    origGE = robjects.r['origGE']
    origCNV = robjects.r['origCNV']
    # origME = robjects.r['origME']
    # origMI = robjects.r['origMI']
    survival = robjects.r('survival=survival[patients,]')
    survival_idx = robjects.r('row.names(survival)')
    survival_col = robjects.r('colnames(survival)')

    pertGE = robjects.r['pertGE']
    pertCNV = robjects.r['pertCNV']
    # pertME = robjects.r['pertME']
    # pertMI = robjects.r['pertMI']

    # print(type(pertGE))  # <class 'numpy.ndarray'>
    pd_pertGE = pd.DataFrame(pertGE)
    pd_pertCNV = pd.DataFrame(pertCNV)
    # pd_pertME = pd.DataFrame(pertME)
    # pd_pertMI = pd.DataFrame(pertMI)
    pd_survival = pd.DataFrame(survival)
    pd_survival.to_csv('Projects/SubTyping/METABRIC/{}_survival.csv'.format(dataset, dataset))
    pd_pertGE.index = robjects.r['rowGE']
    pd_pertGE.columns = robjects.r['rowGE']
    pd_pertCNV.index = robjects.r['rowCNV']
    pd_pertCNV.columns = robjects.r['rowCNV']
    # pd_pertME.index = robjects.r['rowME']
    # pd_pertME.columns = robjects.r['rowGE']
    # pd_pertMI.index = robjects.r['rowMI']
    # pd_pertMI.columns = robjects.r['rowGE']
    pd_pertGE.to_csv('result/METABRIC/{}/{}PertGE.csv'.format(dataset, dataset), header=True,
                     index=True)  # AttributeError: 'ListVector' object has no attribute 'to_csv'
    pd_pertCNV.to_csv('result/METABRIC/{}/{}PertCNV.csv'.format(dataset, dataset), header=True,
                     index=True)  # AttributeError: 'ListVector' object has no attribute 'to_csv'
    # pd_pertME.to_csv('result/METABRIC/{}/{}PertME.csv'.format(dataset, dataset), header=True,
    #                  index=True)  # AttributeError: 'ListVector' object has no attribute 'to_csv'
    # pd_pertMI.to_csv('result/METABRIC/{}/{}PertMI.csv'.format(dataset, dataset), header=True,
    #                  index=True)  # AttributeError: 'ListVector' object has no attribute 'to_csv'

    # load perturbation result datasets
    DataGE = pd.read_csv('result/METABRIC/{}/{}PertGE.csv'.format(dataset, dataset))  # perturbation result
    DataCNV = pd.read_csv('result/METABRIC/{}/{}PertCNV.csv'.format(dataset, dataset))  # perturbation result
    # DataME = pd.read_csv('result/METABRIC/{}/{}PertME.csv'.format(dataset, dataset))
    # DataMI = pd.read_csv('result/METABRIC/{}/{}PertMI.csv'.format(dataset, dataset))
    m = 'sqeuclidean'
    K = 20
    mu = 0.5

    # input [511 rows x 19580 columns] mydatGE mydatME mydatMI
    mydatGE = robjects.r['mydatGE']
    mydatCNV = robjects.r['mydatCNV']
    # mydatME = robjects.r['mydatME']
    # mydatMI = robjects.r['mydatMI']
    mydatGE = pd.DataFrame(mydatGE)
    mydatCNV = pd.DataFrame(mydatCNV)
    # if dataset == 'KIRC' or dataset == 'LUSC' or dataset == 'BRCA' or dataset == 'GBM' or dataset == 'COAD':
    #     mydatGE = mydatGE.T
    # mydatME = pd.DataFrame(mydatME)
    # if dataset == 'LUSC' or dataset == 'BRCA' or dataset == 'GBM':
    #     mydatME = mydatME.T
    # mydatMI = pd.DataFrame(mydatMI)
    # if dataset == 'KIRC' or dataset == 'GBM':
    #     mydatMI = mydatMI.T
    mydatGE.index = robjects.r('row.names(mydatGE)')
    mydatCNV.index = robjects.r('row.names(mydatCNV)')
    # mydatME.index = robjects.r('row.names(mydatME)')
    # mydatMI.index = robjects.r('row.names(mydatMI)')
    mydatGE.columns = robjects.r('colnames(mydatGE)')
    mydatCNV.columns = robjects.r('colnames(mydatCNV)')
    # mydatME.columns = robjects.r('colnames(mydatME)')
    # mydatMI.columns = robjects.r('colnames(mydatMI)')
    print('mydatGE.index=={}, mydatGE.col=={}'.format(len(mydatGE.index), len(mydatGE.columns)))
    print('mydatCNV.index=={}, mydatCNV.col=={}'.format(len(mydatCNV.index), len(mydatCNV.columns)))
    # print('mydatME.index=={}, mydatME.col=={}'.format(len(mydatME.index), len(mydatME.columns)))
    # print('mydatMI.index=={}, mydatMI.col=={}'.format(len(mydatMI.index), len(mydatMI.columns)))
    if mydatGE.shape[0] != mydatCNV.shape[0]:
        print('Input files must have same samples.')
        exit(1)

    pd_DataGE = pd.DataFrame(pd_pertGE)
    pd_DataCNV = pd.DataFrame(pd_pertCNV)

    print("{}-----mydatGE".format(mydatGE))
    print("{}-----mydatCNV".format(mydatCNV))
    affinity_mydatGE = snf.make_affinity(mydatGE.iloc[:, 1:].values.astype(np.float), metric=m, K=K, mu=mu)
    affinity_mydatCNV = snf.make_affinity(mydatCNV.iloc[:, 1:].values.astype(np.float), metric=m, K=K, mu=mu)

    affinity_mydatGE = pd.DataFrame(affinity_mydatGE)
    affinity_mydatCNV = pd.DataFrame(affinity_mydatCNV)

    affinity_mydatGE.index = robjects.r('row.names(mydatGE)')
    affinity_mydatGE.index = robjects.r('row.names(mydatGE)')
    affinity_mydatCNV.columns = robjects.r('row.names(mydatCNV)')
    affinity_mydatCNV.columns = robjects.r('row.names(mydatCNV)')

    affinity_nets = snf.make_affinity([mydatGE.iloc[:, 1:].values.astype(np.float), mydatCNV.iloc[:, 1:].values.astype(np.float)], metric=m, K=K, mu=mu)

    # two version!
    # P values = ....{'METABRIC_discovery': 0.0027831558797430274, 'METABRIC_validation': 1}
    # P values = ....{'METABRIC_discovery': 0.0027831558797430374, 'METABRIC_validation': 1} this may be data 2.
    fused_net = snf_plus_altered_sim(affinity_nets, pd_DataGE=pd_DataGE, pd_DataCNV=pd_DataCNV, K=K)
    print('snf done--!')

    print('Save fused adjacency matrix...')
    DataGEList = DataGE.columns.tolist()
    del DataGEList[0]
    fused_df = pd.DataFrame(fused_net)
    fused_df.columns = DataGEList
    fused_df.index = DataGEList
    fused_df.to_csv('result/METABRIC/{}/{}_fused_matrix.csv'.format(dataset, dataset), header=True, index=True)
    np.fill_diagonal(fused_df.values, 0)

    print('spectral clustering...........')
    origin_survivaldata = pd.DataFrame(survival)
    survivaldata = pd.DataFrame(survival)
    cluster_num = 4
    filename = 'result/METABRIC/{}/{}_fused_matrix.csv'.format(dataset, dataset)
    datas = load_data(filename=filename)
    data = datas.iloc[:, 1:]
    groups_y = []
    for cluster_num in range(2, 11):
        print('cluster_num', cluster_num)
        survivaldata = origin_survivaldata
        print("{}---cluster_num".format(cluster_num))
        y_pred = SpectralClustering(gamma=0.5, n_clusters=cluster_num).fit_predict(data)
        groups_y.append(y_pred)
        Scores.append(metrics.silhouette_score(data, y_pred, metric='euclidean'))
        Scores1.append(metrics.calinski_harabasz_score(data, y_pred))
        # metrics.calinski_harabasz_score(data, y_pred)
        # y_pred = SpectralClustering(gamma=0.1, n_clusters=4).fit_predict(data)
        print('groups_y', groups_y)
        print('len.groups_y', len(groups_y))
        print('cluster_num-2', cluster_num - 2)
        print('cluster_num', cluster_num)
        f = groups_y[cluster_num - 2]
        temp_f = []
        for f_num in f:
            temp_f.append(f_num + 1)
        f = temp_f
        # f = f.tolist()
        #     survivaldata = survivaldata.T
        #     survivaldata = pd.DataFrame(survivaldata.values.T, index=survivaldata.columns, columns=survivaldata.index)  # [4 rows x 124 columns]
        ''' zhushi
        survivaldata.index RangeIndex(start=0, stop=3, step=1)
survivaldata.columns RangeIndex(start=0, stop=124, step=1)
f 124
        '''
        if len(survivaldata.index) < 10:
            survivaldata = survivaldata.T
        print('survivaldata', survivaldata)
        print('survivaldata.index', survivaldata.index)
        print('survivaldata.columns', survivaldata.columns)
        print('f', len(f))
        survivaldata.insert(loc=3, column='groups', value=f)
        survivaltPath = 'result/METABRIC/{}/{}_survival_{}.csv'.format(dataset, dataset, cluster_num)
        with open(survivaltPath, 'w+', newline='',
                  encoding='utf-8') as alter_survival:
            writer = csv.writer(alter_survival)
            title = survivaldata.loc[0]
            writer.writerow(title)
            for i in range(1, len(survivaldata.index)):
                row = survivaldata.loc[i]
                writer.writerow(row)
        proSurvival = pd.DataFrame(pd.read_csv(survivaltPath))
        surindex = np.array(survival_idx).tolist()
        surcolums = np.array(survival_col).tolist()
        del surindex[0]
        surcolums.append('groups')
        proSurvival.index = surindex
        proSurvival.columns = surcolums
        proSurvival.drop(columns='PatientID', inplace=True)
        proSurvival.to_csv('result/METABRIC/{}/{}_survival_{}.csv'.format(dataset, dataset, cluster_num))
        robjects.r('''
                BRCA_survival=read.csv("result/METABRIC/{}/{}_survival_{}.csv")
                vector1 <- BRCA_survival$groups
                column.names <- BRCA_survival$sample
                groups<-array(vector1,dimnames=list(column.names))
                coxFit <- coxph(Surv(time = Survival, event = Death) ~ as.factor(groups), data = BRCA_survival, ties="exact")
                mfit <- survfit(Surv(Survival, Death == 1) ~ as.factor(groups), data = BRCA_survival)
                plot(mfit, col=levels(factor(groups)), main = paste("Survival curves for ", "{}", ", level N (spectral clustering ) cluster number", {}, sep=""), xlab = "Days", ylab="Survival", lwd=2)
                legend("top", legend = paste("Cox p-value:", round(summary(coxFit)$sctest[3],digits = 5), sep=""))
                legend("topright", fill=levels(factor(groups)), legend=paste("Group ",levels(factor(groups)), ": ", table(groups)[levels(factor(groups))], sep=""))
            '''.format(dataset, dataset, cluster_num, dataset, cluster_num))
    robjects.r('dev.off()')

    # plot finial(spectral clistering) figure
    # robjects.r('''
    #     BRCA_survival=read.csv("result/DataTCGA/{}/{}_survival.csv")
    #     vector1 <- BRCA_survival$groups
    #     column.names <- BRCA_survival$sample
    #     groups<-array(vector1,dimnames=list(column.names))
    #
    #     coxFit <- coxph(Surv(time = Survival, event = Death) ~ as.factor(groups), data = BRCA_survival, ties="exact")
    #     mfit <- survfit(Surv(Survival, Death == 1) ~ as.factor(groups), data = BRCA_survival)
    #     plot(mfit, col=levels(factor(groups)), main = paste("Survival curves for ", "{}", ", level N (spectral clustering)", sep=""), xlab = "Days", ylab="Survival", lwd=2)
    #     legend("top", legend = paste("Cox p-value:", round(summary(coxFit)$sctest[3],digits = 5), sep=""))
    #     legend("topright", fill=levels(factor(groups)), legend=paste("Group ",levels(factor(groups)), ": ", table(groups)[levels(factor(groups))], sep=""))
    #     dev.off()
    # '''.format(dataset, dataset, dataset))
    Rp_value = robjects.r('summary(coxFit)$sctest[3]')
    p_values[dataset] = Rp_value[0]
    SC_values.append(Scores)
    AH_values.append(Scores1)
    print('Scores', Scores)
    print('Scores1', Scores1)
    print("Silhouette Coefficient Score", Scores)
    temp_Scores = []
    for j in range(len(Scores)):
        temp_Scores.append(math.log(10, np.abs(Scores[j])))
    print("len(Scores)", len(Scores))
    print("len(temp_Scores)", len(temp_Scores))
    print("len(Scores1)", len(Scores1))
    print("temp_Scores", temp_Scores)
    print("AH Score", Scores1)
    i = range(2, len(Scores1) + 2)
    plt.xlabel('k')
    plt.ylabel('value')
    plt.plot(i, temp_Scores, 'g.-', i, Scores1, 'b.-')
    plotRes(data, y_pred, cluster_num)
    print("P values = ....{}".format(p_values))
    # P values = ....{'METABRIC_discovery': 0.002783155879743035, 'METABRIC_validation': 0.0027831558797430374}
    # P values = ....{'METABRIC_discovery': 0.000799949444055895, 'METABRIC_validation': 1}
    # break
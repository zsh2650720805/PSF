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
import snf_plus_altered_sim_all
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
rpackages.importr('SNFtool')

# import R scripts
robjects.r('source("R_scripts/BasicFunctions.R")')

# initialize and set some parameters
dataPath = robjects.r('dataPath="Projects/SubTyping/DataTCGA/"')
robjects.r('resultPath="result/DataTCGA2/"')
Kmax = robjects.r('Kmax=10')  # it should be 10
iter = robjects.r('iter=200')  # it should be 200
noisePercent = robjects.r('noisePercent="med"')
kmIter = robjects.r('kmIter=200')  # it should be 200
datasets = robjects.r('datasets=c("KIRC", "LUSC", "BRCA", "LAML", "GBM", "COAD")')
p_values1 = {'KIRC': 1, 'LUSC': 1, 'BRCA': 1, 'LAML': 1, 'GBM': 1, 'COAD': 1}
p_values2 = {'KIRC': 1, 'LUSC': 1, 'BRCA': 1, 'LAML': 1, 'GBM': 1, 'COAD': 1}
p_values3 = {'KIRC': 1, 'LUSC': 1, 'BRCA': 1, 'LAML': 1, 'GBM': 1, 'COAD': 1}
p_values_list = []
for dataset in datasets:
    robjects.r('''
        set.seed(1)
        
        ####Using Perturbation-------------------------->
        file=paste(dataPath,"{}", "/", "{}", "_ProcessedData.RData" ,sep="") 
        load(file)
        
        t1=Sys.time()
  
        patients=rownames(survival)
        patients=intersect(patients,rownames(mydatGE))
        patients=intersect(patients,rownames(mydatME))
        patients=intersect(patients,rownames(mydatMI))
      
        mydatGE=mydatGE[patients,]
        mydatME=mydatME[patients,]
        mydatMI=mydatMI[patients,]
        norm_mydatGE = standardNormalization(mydatGE)
        norm_mydatME = standardNormalization(mydatME)
        norm_mydatMI = standardNormalization(mydatMI)
        survival=survival[patients,]
    '''.format(dataset, dataset))
    survival = robjects.r('survival=survival[patients,]')
    survival_idx = robjects.r('row.names(survival)')
    survival_col = robjects.r('colnames(survival)')
    robjects.r('''  
        clinical <- read.table(file=paste(dataPath, "{}", "/", "{}" ,"_Clinical.txt", sep=""), sep="\t", header=T, row.names=1,stringsAsFactors = F, fill=T)
        clinical <- clinical[-1,];clinical <- clinical[-1,]
        a<-rownames(clinical)
        rownames(clinical)<-paste(substr(a,1,4),substr(a,6,7),substr(a,9,12),sep=".")
        clinical <- clinical[rownames(survival),]
        
        rowGE<-row.names(mydatGE)
        rowME<-row.names(mydatME)
        rowMI<-row.names(mydatMI)
        colGE<-colnames(mydatGE)
        colME<-colnames(mydatME)
        colMI<-colnames(mydatMI)
        
        dataList <- list (mydatGE, mydatME) 
        names(dataList) = c("GE", "ME")
        
        # result=SubtypingOmicsData(dataList = dataList, Kmax = Kmax, noisePercent = noisePercent, iter = iter, kmIter = kmIter)
    '''.format(dataset, dataset, dataset, dataset, dataset, dataset, dataset))

    # GE ME
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

        origME=dataTypeResult[[2]]$origS[[dataTypeResult[[1]]$k]]
        PWME = dataTypeResult[[2]]$origS[[dataTypeResult[[1]]$k]]
        pertME=dataTypeResult[[2]]$pertS[[dataTypeResult[[1]]$k]]
        
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
                    maxHeightM = findMaxHeight(hcPWM, maxK)
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
    robjects.r('resultFile=paste(resultPath, "{}", "/", "PINS_", "{}", "GEME", ".RData" ,sep="")'.format(dataset, dataset))
    robjects.r('save(Kmax, dataList, survival, clinical, result, t1, t2, file=resultFile)'.format(dataset))
    robjects.r('pdfFile=paste(resultPath, "{}", "/", "PINS_", "{}", "GEME", ".pdf" ,sep="")'.format(dataset, dataset))
    robjects.r('''
        pdf(pdfFile)
        groups = result$groups
        groups2=result$groups2
        plot(result$dataTypeResult[[1]]$Discrepancy$AUC, ylab= "Area under the curve", xlab="Cluster number", main=paste("AUC of gene expression for ", "{}", " data", sep=""))
        lines(1:Kmax, result$dataTypeResult[[1]]$Discrepancy$AUC)
        points(result$dataTypeResult[[1]]$k, result$dataTypeResult[[1]]$Discrepancy$AUC[result$dataTypeResult[[1]]$k],col="red")

        plot(result$dataTypeResult[[2]]$Discrepancy$AUC, ylab= "Area under the curve", xlab="Cluster number", main=paste("AUC of methylation for ", "{}", " data", sep=""))
        lines(1:Kmax, result$dataTypeResult[[2]]$Discrepancy$AUC)
        points(result$dataTypeResult[[2]]$k, result$dataTypeResult[[2]]$Discrepancy$AUC[result$dataTypeResult[[2]]$k],col="red")

        # plot(result$dataTypeResult[[3]]$Discrepancy$AUC, ylab= "Area under the curve", xlab="Cluster number", main=paste("AUC of microRNA for ", "{}", " data", sep=""))
        # lines(1:Kmax, result$dataTypeResult[[3]]$Discrepancy$AUC)
        # points(result$dataTypeResult[[3]]$k, result$dataTypeResult[[3]]$Discrepancy$AUC[result$dataTypeResult[[3]]$k],col="red")

        coxFit <- coxph(Surv(time = Survival, event = Death) ~ as.factor(result$dataTypeResult[[1]]$groups), data = survival, ties="exact")
        mfit <- survfit(Surv(Survival, Death == 1) ~ as.factor(result$dataTypeResult[[1]]$groups), data = survival)
        plot(mfit, col=unique(result$dataTypeResult[[1]]$groups), main = paste("Survival curves for gene expression of ","{}", " (PertCluster)", sep=""), xlab = "Days", ylab="Survival", lwd=2)
        legend("topright", legend = paste("Cox p-value:", round(summary(coxFit)$sctest[3],digits = 5), sep=""))

        coxFit <- coxph(Surv(time = Survival, event = Death) ~ as.factor(result$dataTypeResult[[2]]$groups), data = survival, ties="exact")
        mfit <- survfit(Surv(Survival, Death == 1) ~ as.factor(result$dataTypeResult[[2]]$groups), data = survival)
        plot(mfit, col=unique(result$dataTypeResult[[2]]$groups), main = paste("Survival curves for methylation of ", "{}", " (PertCluster)", sep=""), xlab = "Days", ylab="Survival", lwd=2)
        legend("topright", legend = paste("Cox p-value:", round(summary(coxFit)$sctest[3],digits = 5), sep=""))

        # coxFit <- coxph(Surv(time = Survival, event = Death) ~ as.factor(result$dataTypeResult[[3]]$groups), data = survival, ties="exact")
        # mfit <- survfit(Surv(Survival, Death == 1) ~ as.factor(result$dataTypeResult[[3]]$groups), data = survival)
        # plot(mfit, col=unique(result$dataTypeResult[[3]]$groups), main = paste("Survival curves for microRNA of ", "{}", " (PertCluster)", sep=""), xlab = "Days", ylab="Survival", lwd=2)
        # legend("topright", legend = paste("Cox p-value:", round(summary(coxFit)$sctest[3],digits = 5), sep=""))

        ageCol <- abs(as.numeric(clinical$"birth_days_to"))/365
        names(ageCol) <- rownames(clinical)
    '''.format(dataset, dataset, dataset, dataset, dataset, dataset, dataset, dataset, dataset, dataset))
    robjects.r('''
        age <- list()
    '''.format(dataset, dataset, dataset, dataset, dataset, dataset, dataset, dataset, dataset, dataset))
    robjects.r('''
        for (j in levels(factor(groups))) {
            age[[j]] <- ageCol[names(groups[groups==j])]
        }
    ''')
    robjects.r('''
        boxplot(age, main=paste("Age distribution, ", "{}", sep=""), xlab="Groups", ylab="Age")
    '''.format(dataset))
    robjects.r('''
        coxFit <- coxph(Surv(time = Survival, event = Death) ~ as.factor(groups), data = survival[names(groups),], ties="exact")
        mfit <- survfit(Surv(Survival, Death == 1) ~ as.factor(groups), data = survival[names(groups),])

        a <-intersect(unique(groups2), unique(groups));names(a) <- intersect(unique(groups2), unique(groups)); a[setdiff(unique(groups2), unique(groups))] <- seq(setdiff(unique(groups2), unique(groups)))+max(groups)
        colors <- a[levels(factor(groups2))]

        plot(mfit, col=levels(factor(groups)), main = paste("Survival curves for ", "{}", ", level 1 (PertCluster)", sep=""), xlab = "Days", ylab="Survival", lwd=2)
        legend("top", legend = paste("Cox p-value:", round(summary(coxFit)$sctest[3],digits = 5), sep=""))
        legend("topright", fill=levels(factor(groups)), legend=paste("Group ",levels(factor(groups)), ": ", table(groups)[levels(factor(groups))], sep=""))
        ageCol <- abs(as.numeric(clinical$"birth_days_to"))/365
        names(ageCol) <- rownames(clinical)
    
    '''.format(dataset, dataset, dataset, dataset, dataset, dataset, dataset, dataset, dataset, dataset))
    robjects.r('''
        age <- list()
    '''.format(dataset, dataset, dataset, dataset, dataset, dataset, dataset, dataset, dataset, dataset))
    robjects.r('''
        for (j in levels(factor(groups))) {
            age[[j]] <- ageCol[names(groups[groups==j])]
        }
    ''')
    robjects.r('''
        boxplot(age, main=paste("Age distribution, ", "{}", sep=""), xlab="Groups", ylab="Age")
    '''.format(dataset))
    robjects.r('''
        coxFit <- coxph(Surv(time = Survival, event = Death) ~ as.factor(groups2), data = survival, ties="exact")
        mfit <- survfit(Surv(Survival, Death == 1) ~ as.factor(groups2), data = survival)
        a <-intersect(unique(groups2), unique(groups));names(a) <- intersect(unique(groups2), unique(groups)); a[setdiff(unique(groups2), unique(groups))] <- seq(setdiff(unique(groups2), unique(groups)))+max(groups)
        colors <- a[levels(factor(groups2))]
        plot(mfit, col=colors, main = paste("Survival curves for ", "{}", ", level 2 (PertCluster)", sep=""), xlab = "Days", ylab="Survival", lwd=2)
        legend("top", legend = paste("Cox p-value:", round(summary(coxFit)$sctest[3],digits = 5), sep=""))
        legend("topright", fill=colors, legend=paste("Group ",levels(factor(groups2)), ": ", table(groups2)[levels(factor(groups2))], sep=""))
    '''.format(dataset, dataset, dataset, dataset, dataset, dataset, dataset, dataset, dataset, dataset))
    # obtain pertS connective matrices
    origGE = robjects.r['origGE']
    origME = robjects.r['origME']
    # origMI = robjects.r['origMI']

    pertGE = robjects.r['pertGE']
    pertME = robjects.r['pertME']
    # pertMI = robjects.r['pertMI']
    # print(type(pertGE))  # <class 'numpy.ndarray'>
    pd_pertGE = pd.DataFrame(pertGE)
    pd_pertME = pd.DataFrame(pertME)
    # pd_pertMI = pd.DataFrame(pertMI)
    pd_survival = pd.DataFrame(survival)
    pd_survival.to_csv('Projects/SubTyping/DataTCGA/{}/{}_survival.csv'.format(dataset, dataset))
    pd_pertGE.index = robjects.r['rowGE']
    pd_pertGE.columns = robjects.r['rowGE']
    pd_pertME.index = robjects.r['rowME']
    pd_pertME.columns = robjects.r['rowGE']
    # pd_pertMI.index = robjects.r['rowMI']
    # pd_pertMI.columns = robjects.r['rowGE']
    pd_pertGE.to_csv('result/DataTCGA2/{}/{}PertGE.csv'.format(dataset, dataset), header=True,
                     index=True)  # AttributeError: 'ListVector' object has no attribute 'to_csv'
    pd_pertME.to_csv('result/DataTCGA2/{}/{}PertME.csv'.format(dataset, dataset), header=True,
                     index=True)  # AttributeError: 'ListVector' object has no attribute 'to_csv'
    # pd_pertMI.to_csv('result/DataTCGA2/{}/{}PertMI.csv'.format(dataset, dataset), header=True,
    #                  index=True)  # AttributeError: 'ListVector' object has no attribute 'to_csv'

    # load perturbation result datasets
    DataGE = pd.read_csv('result/DataTCGA2/{}/{}PertGE.csv'.format(dataset, dataset))  # perturbation result
    DataME = pd.read_csv('result/DataTCGA2/{}/{}PertME.csv'.format(dataset, dataset))
    # DataMI = pd.read_csv('result/DataTCGA2/{}/{}PertMI.csv'.format(dataset, dataset))
    m = 'sqeuclidean'
    K = 20
    mu = 0.5

    # input [511 rows x 19580 columns] mydatGE mydatME mydatMI
    mydatGE = robjects.r['mydatGE']
    mydatME = robjects.r['mydatME']
    # mydatMI = robjects.r['mydatMI']
    mydatGE = pd.DataFrame(mydatGE)
    if dataset == 'KIRC' or dataset == 'LUSC' or dataset == 'BRCA' or dataset == 'GBM' or dataset == 'COAD':
        mydatGE = mydatGE.T
    mydatME = pd.DataFrame(mydatME)
    if dataset == 'LUSC' or dataset == 'BRCA' or dataset == 'GBM':
        mydatME = mydatME.T
    # mydatMI = pd.DataFrame(mydatMI)
    # if dataset == 'KIRC' or dataset == 'GBM':
    #     mydatMI = mydatMI.T
    mydatGE.index = robjects.r('row.names(mydatGE)')
    mydatME.index = robjects.r('row.names(mydatME)')
    # mydatMI.index = robjects.r('row.names(mydatMI)')
    mydatGE.columns = robjects.r('colnames(mydatGE)')
    mydatME.columns = robjects.r('colnames(mydatME)')
    # mydatMI.columns = robjects.r('colnames(mydatMI)')

    if mydatGE.shape[0] != mydatME.shape[0]: #or mydatGE.shape[0] != mydatMI.shape[0]
        print('Input files must have same samples.')
        exit(1)

    pd_DataGE = pd.DataFrame(pd_pertGE)
    pd_DataME = pd.DataFrame(pd_pertME)
    # pd_DataMI = pd.DataFrame(pd_pertMI)

    affinity_mydatGE = snf.make_affinity(mydatGE.iloc[:, 1:].values.astype(np.float), metric=m, K=K, mu=mu)
    affinity_mydatME = snf.make_affinity(mydatME.iloc[:, 1:].values.astype(np.float), metric=m, K=K, mu=mu)
    # affinity_mydatMI = snf.make_affinity(mydatMI.iloc[:, 1:].values.astype(np.float), metric=m, K=K, mu=mu)

    affinity_mydatGE = pd.DataFrame(affinity_mydatGE)
    affinity_mydatME = pd.DataFrame(affinity_mydatME)
    # affinity_mydatMI = pd.DataFrame(affinity_mydatMI)

    affinity_mydatGE.index = robjects.r('row.names(mydatGE)')
    affinity_mydatME.index = robjects.r('row.names(mydatME)')
    # affinity_mydatMI.index = robjects.r('row.names(mydatMI)')
    affinity_mydatGE.columns = robjects.r('row.names(mydatGE)')
    affinity_mydatME.columns = robjects.r('row.names(mydatME)')
    # affinity_mydatMI.columns = robjects.r('row.names(mydatMI)')

    affinity_nets = snf.make_affinity([mydatGE.iloc[:, 1:].values.astype(np.float), mydatME.iloc[:, 1:].values.astype(np.float)],metric=m, K=K, mu=mu)

    fused_net = snf_plus_altered_sim_all.snf_plus_altered_sim_GEME(affinity_nets, pd_DataGE=pd_DataGE, pd_DataME=pd_DataME, K=K)
    print('snf done--!')

    print('Save fused adjacency matrix...')
    DataGEList = DataGE.columns.tolist()
    del DataGEList[0]
    fused_df = pd.DataFrame(fused_net)
    fused_df.columns = DataGEList
    fused_df.index = DataGEList
    fused_df.to_csv('result/DataTCGA2/{}/{}_GEME_fused_matrix.csv'.format(dataset, dataset), header=True, index=True)
    np.fill_diagonal(fused_df.values, 0)

    print('spectral clustering...........')
    survivaldata = pd.DataFrame(survival)
    cluster_num = 4
    filename = 'result/DataTCGA2/{}/{}_GEME_fused_matrix.csv'.format(dataset, dataset)
    datas = load_data(filename=filename)
    data = datas.iloc[:, 1:]
    y_pred = SpectralClustering(gamma=0.1, n_clusters=4).fit_predict(data)
    f = y_pred
    temp_f = []
    for f_num in f:
        temp_f.append(f_num + 1)
    f = temp_f
    # f = f.tolist()
    survivaldata = survivaldata.T
    survivaldata.insert(loc=3, column='groups', value=f)
    survivaltPath = 'result/DataTCGA2/{}/{}_GEME_survival.csv'.format(dataset, dataset)
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
    proSurvival.to_csv(survivaltPath)
    # plot finial(spectral clistering) figure
    robjects.r('''
           BRCA_survival=read.csv("result/DataTCGA2/{}/{}_GEME_survival.csv")
           vector1 <- BRCA_survival$groups
           column.names <- BRCA_survival$sample
           groups<-array(vector1,dimnames=list(column.names))

           coxFit <- coxph(Surv(time = Survival, event = Death) ~ as.factor(groups), data = BRCA_survival, ties="exact")
           mfit <- survfit(Surv(Survival, Death == 1) ~ as.factor(groups), data = BRCA_survival)
           plot(mfit, col=levels(factor(groups)), main = paste("Survival curves for ", "{}", ", level N (spectral clustering)", sep=""), xlab = "Days", ylab="Survival", lwd=2)
           legend("top", legend = paste("Cox p-value:", round(summary(coxFit)$sctest[3],digits = 5), sep=""))
           legend("topright", fill=levels(factor(groups)), legend=paste("Group ",levels(factor(groups)), ": ", table(groups)[levels(factor(groups))], sep=""))
           dev.off()
       '''.format(dataset, dataset, dataset))
    Rp_value = robjects.r('summary(coxFit)$sctest[3]')
    p_values1[dataset] = Rp_value[0]
    print("Calinski-Harabasz Score", metrics.calinski_harabasz_score(data, y_pred))
    plotRes(data, y_pred, cluster_num)
    # print("P values = ....{}".format(p_values))
    p_values_list.append("GEME")
    p_values_list.append(p_values1)
    p_values_list.append("\n")
    print("P values list = ....{}".format(p_values_list))

    # ME MI
    robjects.r('''
        agreementCutoff=0.5
        dataTypeResult <- list()
        for (i in 1:length(dataList)) {
            message(paste("Data type: ", i, sep=""))
            dataTypeResult[[i]] <- PerturbationClustering(data=dataList[[i]], Kmax=Kmax, noisePercent="med", iter=iter, kmIter=kmIter)
            # break
        }


        origME=dataTypeResult[[1]]$origS[[dataTypeResult[[1]]$k]]
        PWME = dataTypeResult[[1]]$origS[[dataTypeResult[[1]]$k]]
        pertME=dataTypeResult[[1]]$pertS[[dataTypeResult[[1]]$k]]
        
        origMI=dataTypeResult[[2]]$origS[[dataTypeResult[[1]]$k]]
        PWMI = dataTypeResult[[2]]$origS[[dataTypeResult[[1]]$k]]
        pertMI=dataTypeResult[[2]]$pertS[[dataTypeResult[[1]]$k]]
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
                    maxHeightM = findMaxHeight(hcPWM, maxK)
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
    robjects.r(
        'resultFile=paste(resultPath, "{}", "/", "PINS_", "{}", "MEMI", ".RData" ,sep="")'.format(dataset, dataset))
    robjects.r('save(Kmax, dataList, survival, clinical, result, t1, t2, file=resultFile)'.format(dataset))
    robjects.r('pdfFile=paste(resultPath, "{}", "/", "PINS_", "{}", "MEMI", ".pdf" ,sep="")'.format(dataset, dataset))
    robjects.r('''
        pdf(pdfFile)
        groups = result$groups
        groups2=result$groups2
        plot(result$dataTypeResult[[1]]$Discrepancy$AUC, ylab= "Area under the curve", xlab="Cluster number", main=paste("AUC of gene expression for ", "{}", " data", sep=""))
        lines(1:Kmax, result$dataTypeResult[[1]]$Discrepancy$AUC)
        points(result$dataTypeResult[[1]]$k, result$dataTypeResult[[1]]$Discrepancy$AUC[result$dataTypeResult[[1]]$k],col="red")

        plot(result$dataTypeResult[[2]]$Discrepancy$AUC, ylab= "Area under the curve", xlab="Cluster number", main=paste("AUC of methylation for ", "{}", " data", sep=""))
        lines(1:Kmax, result$dataTypeResult[[2]]$Discrepancy$AUC)
        points(result$dataTypeResult[[2]]$k, result$dataTypeResult[[2]]$Discrepancy$AUC[result$dataTypeResult[[2]]$k],col="red")

        # plot(result$dataTypeResult[[3]]$Discrepancy$AUC, ylab= "Area under the curve", xlab="Cluster number", main=paste("AUC of microRNA for ", "{}", " data", sep=""))
        # lines(1:Kmax, result$dataTypeResult[[3]]$Discrepancy$AUC)
        # points(result$dataTypeResult[[3]]$k, result$dataTypeResult[[3]]$Discrepancy$AUC[result$dataTypeResult[[3]]$k],col="red")

        coxFit <- coxph(Surv(time = Survival, event = Death) ~ as.factor(result$dataTypeResult[[1]]$groups), data = survival, ties="exact")
        mfit <- survfit(Surv(Survival, Death == 1) ~ as.factor(result$dataTypeResult[[1]]$groups), data = survival)
        plot(mfit, col=unique(result$dataTypeResult[[1]]$groups), main = paste("Survival curves for gene expression of ","{}", " (PertCluster)", sep=""), xlab = "Days", ylab="Survival", lwd=2)
        legend("topright", legend = paste("Cox p-value:", round(summary(coxFit)$sctest[3],digits = 5), sep=""))

        coxFit <- coxph(Surv(time = Survival, event = Death) ~ as.factor(result$dataTypeResult[[2]]$groups), data = survival, ties="exact")
        mfit <- survfit(Surv(Survival, Death == 1) ~ as.factor(result$dataTypeResult[[2]]$groups), data = survival)
        plot(mfit, col=unique(result$dataTypeResult[[2]]$groups), main = paste("Survival curves for methylation of ", "{}", " (PertCluster)", sep=""), xlab = "Days", ylab="Survival", lwd=2)
        legend("topright", legend = paste("Cox p-value:", round(summary(coxFit)$sctest[3],digits = 5), sep=""))

        # coxFit <- coxph(Surv(time = Survival, event = Death) ~ as.factor(result$dataTypeResult[[3]]$groups), data = survival, ties="exact")
        # mfit <- survfit(Surv(Survival, Death == 1) ~ as.factor(result$dataTypeResult[[3]]$groups), data = survival)
        # plot(mfit, col=unique(result$dataTypeResult[[3]]$groups), main = paste("Survival curves for microRNA of ", "{}", " (PertCluster)", sep=""), xlab = "Days", ylab="Survival", lwd=2)
        # legend("topright", legend = paste("Cox p-value:", round(summary(coxFit)$sctest[3],digits = 5), sep=""))

        ageCol <- abs(as.numeric(clinical$"birth_days_to"))/365
        names(ageCol) <- rownames(clinical)
    '''.format(dataset, dataset, dataset, dataset, dataset, dataset, dataset, dataset, dataset, dataset))
    robjects.r('''
        age <- list()
    '''.format(dataset, dataset, dataset, dataset, dataset, dataset, dataset, dataset, dataset, dataset))
    robjects.r('''
        for (j in levels(factor(groups))) {
            age[[j]] <- ageCol[names(groups[groups==j])]
        }
    ''')
    robjects.r('''
        boxplot(age, main=paste("Age distribution, ", "{}", sep=""), xlab="Groups", ylab="Age")
    '''.format(dataset))
    robjects.r('''
        coxFit <- coxph(Surv(time = Survival, event = Death) ~ as.factor(groups), data = survival[names(groups),], ties="exact")
        mfit <- survfit(Surv(Survival, Death == 1) ~ as.factor(groups), data = survival[names(groups),])

        a <-intersect(unique(groups2), unique(groups));names(a) <- intersect(unique(groups2), unique(groups)); a[setdiff(unique(groups2), unique(groups))] <- seq(setdiff(unique(groups2), unique(groups)))+max(groups)
        colors <- a[levels(factor(groups2))]

        plot(mfit, col=levels(factor(groups)), main = paste("Survival curves for ", "{}", ", level 1 (PertCluster)", sep=""), xlab = "Days", ylab="Survival", lwd=2)
        legend("top", legend = paste("Cox p-value:", round(summary(coxFit)$sctest[3],digits = 5), sep=""))
        legend("topright", fill=levels(factor(groups)), legend=paste("Group ",levels(factor(groups)), ": ", table(groups)[levels(factor(groups))], sep=""))
        ageCol <- abs(as.numeric(clinical$"birth_days_to"))/365
        names(ageCol) <- rownames(clinical)

    '''.format(dataset, dataset, dataset, dataset, dataset, dataset, dataset, dataset, dataset, dataset))
    robjects.r('''
        age <- list()
    '''.format(dataset, dataset, dataset, dataset, dataset, dataset, dataset, dataset, dataset, dataset))
    robjects.r('''
        for (j in levels(factor(groups))) {
            age[[j]] <- ageCol[names(groups[groups==j])]
        }
    ''')
    robjects.r('''
        boxplot(age, main=paste("Age distribution, ", "{}", sep=""), xlab="Groups", ylab="Age")
    '''.format(dataset))
    robjects.r('''
        coxFit <- coxph(Surv(time = Survival, event = Death) ~ as.factor(groups2), data = survival, ties="exact")
        mfit <- survfit(Surv(Survival, Death == 1) ~ as.factor(groups2), data = survival)
        a <-intersect(unique(groups2), unique(groups));names(a) <- intersect(unique(groups2), unique(groups)); a[setdiff(unique(groups2), unique(groups))] <- seq(setdiff(unique(groups2), unique(groups)))+max(groups)
        colors <- a[levels(factor(groups2))]
        plot(mfit, col=colors, main = paste("Survival curves for ", "{}", ", level 2 (PertCluster)", sep=""), xlab = "Days", ylab="Survival", lwd=2)
        legend("top", legend = paste("Cox p-value:", round(summary(coxFit)$sctest[3],digits = 5), sep=""))
        legend("topright", fill=colors, legend=paste("Group ",levels(factor(groups2)), ": ", table(groups2)[levels(factor(groups2))], sep=""))
    '''.format(dataset, dataset, dataset, dataset, dataset, dataset, dataset, dataset, dataset, dataset))
    # obtain pertS connective matrices
    # origGE = robjects.r['origGE']
    origME = robjects.r['origME']
    origMI = robjects.r['origMI']

    # pertGE = robjects.r['pertGE']
    pertME = robjects.r['pertME']
    pertMI = robjects.r['pertMI']
    # print(type(pertGE))  # <class 'numpy.ndarray'>
    # pd_pertGE = pd.DataFrame(pertGE)
    pd_pertME = pd.DataFrame(pertME)
    pd_pertMI = pd.DataFrame(pertMI)
    pd_survival = pd.DataFrame(survival)
    pd_survival.to_csv('Projects/SubTyping/DataTCGA/{}/{}_survival.csv'.format(dataset, dataset))
    # pd_pertGE.index = robjects.r['rowGE']
    # pd_pertGE.columns = robjects.r['rowGE']
    pd_pertME.index = robjects.r['rowME']
    pd_pertME.columns = robjects.r['rowGE']
    pd_pertMI.index = robjects.r['rowMI']
    pd_pertMI.columns = robjects.r['rowMI']
    # pd_pertGE.to_csv('result/DataTCGA2/{}/{}PertGE.csv'.format(dataset, dataset), header=True,
    #                  index=True)  # AttributeError: 'ListVector' object has no attribute 'to_csv'
    pd_pertME.to_csv('result/DataTCGA2/{}/{}PertME.csv'.format(dataset, dataset), header=True,
                     index=True)  # AttributeError: 'ListVector' object has no attribute 'to_csv'
    pd_pertMI.to_csv('result/DataTCGA2/{}/{}PertMI.csv'.format(dataset, dataset), header=True,
                     index=True)  # AttributeError: 'ListVector' object has no attribute 'to_csv'

    # load perturbation result datasets
    # DataGE = pd.read_csv('result/DataTCGA2/{}/{}PertGE.csv'.format(dataset, dataset))  # perturbation result
    DataME = pd.read_csv('result/DataTCGA2/{}/{}PertME.csv'.format(dataset, dataset))
    DataMI = pd.read_csv('result/DataTCGA2/{}/{}PertMI.csv'.format(dataset, dataset))
    m = 'sqeuclidean'
    K = 20
    mu = 0.5

    # input [511 rows x 19580 columns] mydatGE mydatME mydatMI
    # mydatGE = robjects.r['mydatGE']
    mydatME = robjects.r['mydatME']
    mydatMI = robjects.r['mydatMI']
    # mydatGE = pd.DataFrame(mydatGE)
    # if dataset == 'KIRC' or dataset == 'LUSC' or dataset == 'BRCA' or dataset == 'GBM' or dataset == 'COAD':
    #     mydatGE = mydatGE.T
    mydatME = pd.DataFrame(mydatME)
    if dataset == 'LUSC' or dataset == 'BRCA' or dataset == 'GBM':
        mydatME = mydatME.T
    mydatMI = pd.DataFrame(mydatMI)
    if dataset == 'KIRC' or dataset == 'GBM':
        mydatMI = mydatMI.T
    # mydatGE.index = robjects.r('row.names(mydatGE)')
    mydatME.index = robjects.r('row.names(mydatME)')
    mydatMI.index = robjects.r('row.names(mydatMI)')
    # mydatGE.columns = robjects.r('colnames(mydatGE)')
    mydatME.columns = robjects.r('colnames(mydatME)')
    mydatMI.columns = robjects.r('colnames(mydatMI)')

    if mydatMI.shape[0] != mydatME.shape[0]:  # or mydatGE.shape[0] != mydatMI.shape[0]
        print('Input files must have same samples.')
        exit(1)

    # pd_DataGE = pd.DataFrame(pd_pertGE)
    pd_DataME = pd.DataFrame(pd_pertME)
    pd_DataMI = pd.DataFrame(pd_pertMI)

    # affinity_mydatGE = snf.make_affinity(mydatGE.iloc[:, 1:].values.astype(np.float), metric=m, K=K, mu=mu)
    affinity_mydatME = snf.make_affinity(mydatME.iloc[:, 1:].values.astype(np.float), metric=m, K=K, mu=mu)
    affinity_mydatMI = snf.make_affinity(mydatMI.iloc[:, 1:].values.astype(np.float), metric=m, K=K, mu=mu)

    # affinity_mydatGE = pd.DataFrame(affinity_mydatGE)
    affinity_mydatME = pd.DataFrame(affinity_mydatME)
    affinity_mydatMI = pd.DataFrame(affinity_mydatMI)

    # affinity_mydatGE.index = robjects.r('row.names(mydatGE)')
    affinity_mydatME.index = robjects.r('row.names(mydatME)')
    affinity_mydatMI.index = robjects.r('row.names(mydatMI)')
    # affinity_mydatGE.columns = robjects.r('row.names(mydatGE)')
    affinity_mydatME.columns = robjects.r('row.names(mydatME)')
    affinity_mydatMI.columns = robjects.r('row.names(mydatMI)')

    affinity_nets = snf.make_affinity(
        [mydatME.iloc[:, 1:].values.astype(np.float), mydatMI.iloc[:, 1:].values.astype(np.float)], metric=m, K=K,
        mu=mu)

    fused_net = snf_plus_altered_sim_all.snf_plus_altered_sim_MEMI(affinity_nets, pd_DataME=pd_DataME,
                                                                   pd_DataMI=pd_DataMI, K=K)
    print('snf done--!')

    print('Save fused adjacency matrix...')
    DataGEList = DataGE.columns.tolist()
    del DataGEList[0]
    fused_df = pd.DataFrame(fused_net)
    fused_df.columns = DataGEList
    fused_df.index = DataGEList
    fused_df.to_csv('result/DataTCGA2/{}/{}_MEMI_fused_matrix.csv'.format(dataset, dataset), header=True, index=True)
    np.fill_diagonal(fused_df.values, 0)

    print('spectral clustering...........')
    survivaldata = pd.DataFrame(survival)
    cluster_num = 4
    filename = 'result/DataTCGA2/{}/{}_MEMI_fused_matrix.csv'.format(dataset, dataset)
    datas = load_data(filename=filename)
    data = datas.iloc[:, 1:]
    y_pred = SpectralClustering(gamma=0.1, n_clusters=4).fit_predict(data)
    f = y_pred
    temp_f = []
    for f_num in f:
        temp_f.append(f_num + 1)
    f = temp_f
    # f = f.tolist()
    survivaldata = survivaldata.T
    survivaldata.insert(loc=3, column='groups', value=f)
    survivaltPath = 'result/DataTCGA2/{}/{}_MEMI_survival.csv'.format(dataset, dataset)
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
    proSurvival.to_csv(survivaltPath)
    # plot finial(spectral clistering) figure
    robjects.r('''
           BRCA_survival=read.csv("result/DataTCGA2/{}/{}_MEMI_survival.csv")
           vector1 <- BRCA_survival$groups
           column.names <- BRCA_survival$sample
           groups<-array(vector1,dimnames=list(column.names))

           coxFit <- coxph(Surv(time = Survival, event = Death) ~ as.factor(groups), data = BRCA_survival, ties="exact")
           mfit <- survfit(Surv(Survival, Death == 1) ~ as.factor(groups), data = BRCA_survival)
           plot(mfit, col=levels(factor(groups)), main = paste("Survival curves for ", "{}", ", level N (spectral clustering)", sep=""), xlab = "Days", ylab="Survival", lwd=2)
           legend("top", legend = paste("Cox p-value:", round(summary(coxFit)$sctest[3],digits = 5), sep=""))
           legend("topright", fill=levels(factor(groups)), legend=paste("Group ",levels(factor(groups)), ": ", table(groups)[levels(factor(groups))], sep=""))
           dev.off()
       '''.format(dataset, dataset, dataset))
    Rp_value = robjects.r('summary(coxFit)$sctest[3]')
    p_values2[dataset] = Rp_value[0]
    print("Calinski-Harabasz Score", metrics.calinski_harabasz_score(data, y_pred))
    plotRes(data, y_pred, cluster_num)
    # print("P values = ....{}".format(p_values))
    p_values_list.append("MEMI")
    p_values_list.append(p_values2)
    p_values_list.append("\n")
    print("P values list = ....{}".format(p_values_list))

    # GE MI
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
        
        origMI=dataTypeResult[[2]]$origS[[dataTypeResult[[1]]$k]]
        PWMI = dataTypeResult[[2]]$origS[[dataTypeResult[[1]]$k]]
        pertMI=dataTypeResult[[2]]$pertS[[dataTypeResult[[1]]$k]]
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
                    maxHeightM = findMaxHeight(hcPWM, maxK)
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
    robjects.r(
        'resultFile=paste(resultPath, "{}", "/", "PINS_", "{}", "GEMI", ".RData" ,sep="")'.format(dataset, dataset))
    robjects.r('save(Kmax, dataList, survival, clinical, result, t1, t2, file=resultFile)'.format(dataset))
    robjects.r('pdfFile=paste(resultPath, "{}", "/", "PINS_", "{}", "GEMI", ".pdf" ,sep="")'.format(dataset, dataset))
    robjects.r('''
        pdf(pdfFile)
        groups = result$groups
        groups2=result$groups2
        plot(result$dataTypeResult[[1]]$Discrepancy$AUC, ylab= "Area under the curve", xlab="Cluster number", main=paste("AUC of gene expression for ", "{}", " data", sep=""))
        lines(1:Kmax, result$dataTypeResult[[1]]$Discrepancy$AUC)
        points(result$dataTypeResult[[1]]$k, result$dataTypeResult[[1]]$Discrepancy$AUC[result$dataTypeResult[[1]]$k],col="red")

        plot(result$dataTypeResult[[2]]$Discrepancy$AUC, ylab= "Area under the curve", xlab="Cluster number", main=paste("AUC of methylation for ", "{}", " data", sep=""))
        lines(1:Kmax, result$dataTypeResult[[2]]$Discrepancy$AUC)
        points(result$dataTypeResult[[2]]$k, result$dataTypeResult[[2]]$Discrepancy$AUC[result$dataTypeResult[[2]]$k],col="red")

        # plot(result$dataTypeResult[[3]]$Discrepancy$AUC, ylab= "Area under the curve", xlab="Cluster number", main=paste("AUC of microRNA for ", "{}", " data", sep=""))
        # lines(1:Kmax, result$dataTypeResult[[3]]$Discrepancy$AUC)
        # points(result$dataTypeResult[[3]]$k, result$dataTypeResult[[3]]$Discrepancy$AUC[result$dataTypeResult[[3]]$k],col="red")

        coxFit <- coxph(Surv(time = Survival, event = Death) ~ as.factor(result$dataTypeResult[[1]]$groups), data = survival, ties="exact")
        mfit <- survfit(Surv(Survival, Death == 1) ~ as.factor(result$dataTypeResult[[1]]$groups), data = survival)
        plot(mfit, col=unique(result$dataTypeResult[[1]]$groups), main = paste("Survival curves for gene expression of ","{}", " (PertCluster)", sep=""), xlab = "Days", ylab="Survival", lwd=2)
        legend("topright", legend = paste("Cox p-value:", round(summary(coxFit)$sctest[3],digits = 5), sep=""))

        coxFit <- coxph(Surv(time = Survival, event = Death) ~ as.factor(result$dataTypeResult[[2]]$groups), data = survival, ties="exact")
        mfit <- survfit(Surv(Survival, Death == 1) ~ as.factor(result$dataTypeResult[[2]]$groups), data = survival)
        plot(mfit, col=unique(result$dataTypeResult[[2]]$groups), main = paste("Survival curves for methylation of ", "{}", " (PertCluster)", sep=""), xlab = "Days", ylab="Survival", lwd=2)
        legend("topright", legend = paste("Cox p-value:", round(summary(coxFit)$sctest[3],digits = 5), sep=""))

        # coxFit <- coxph(Surv(time = Survival, event = Death) ~ as.factor(result$dataTypeResult[[3]]$groups), data = survival, ties="exact")
        # mfit <- survfit(Surv(Survival, Death == 1) ~ as.factor(result$dataTypeResult[[3]]$groups), data = survival)
        # plot(mfit, col=unique(result$dataTypeResult[[3]]$groups), main = paste("Survival curves for microRNA of ", "{}", " (PertCluster)", sep=""), xlab = "Days", ylab="Survival", lwd=2)
        # legend("topright", legend = paste("Cox p-value:", round(summary(coxFit)$sctest[3],digits = 5), sep=""))

        ageCol <- abs(as.numeric(clinical$"birth_days_to"))/365
        names(ageCol) <- rownames(clinical)
    '''.format(dataset, dataset, dataset, dataset, dataset, dataset, dataset, dataset, dataset, dataset))
    robjects.r('''
        age <- list()
    '''.format(dataset, dataset, dataset, dataset, dataset, dataset, dataset, dataset, dataset, dataset))
    robjects.r('''
        for (j in levels(factor(groups))) {
            age[[j]] <- ageCol[names(groups[groups==j])]
        }
    ''')
    robjects.r('''
        boxplot(age, main=paste("Age distribution, ", "{}", sep=""), xlab="Groups", ylab="Age")
    '''.format(dataset))
    robjects.r('''
        coxFit <- coxph(Surv(time = Survival, event = Death) ~ as.factor(groups), data = survival[names(groups),], ties="exact")
        mfit <- survfit(Surv(Survival, Death == 1) ~ as.factor(groups), data = survival[names(groups),])

        a <-intersect(unique(groups2), unique(groups));names(a) <- intersect(unique(groups2), unique(groups)); a[setdiff(unique(groups2), unique(groups))] <- seq(setdiff(unique(groups2), unique(groups)))+max(groups)
        colors <- a[levels(factor(groups2))]

        plot(mfit, col=levels(factor(groups)), main = paste("Survival curves for ", "{}", ", level 1 (PertCluster)", sep=""), xlab = "Days", ylab="Survival", lwd=2)
        legend("top", legend = paste("Cox p-value:", round(summary(coxFit)$sctest[3],digits = 5), sep=""))
        legend("topright", fill=levels(factor(groups)), legend=paste("Group ",levels(factor(groups)), ": ", table(groups)[levels(factor(groups))], sep=""))
        ageCol <- abs(as.numeric(clinical$"birth_days_to"))/365
        names(ageCol) <- rownames(clinical)

    '''.format(dataset, dataset, dataset, dataset, dataset, dataset, dataset, dataset, dataset, dataset))
    robjects.r('''
        age <- list()
    '''.format(dataset, dataset, dataset, dataset, dataset, dataset, dataset, dataset, dataset, dataset))
    robjects.r('''
        for (j in levels(factor(groups))) {
            age[[j]] <- ageCol[names(groups[groups==j])]
        }
    ''')
    robjects.r('''
        boxplot(age, main=paste("Age distribution, ", "{}", sep=""), xlab="Groups", ylab="Age")
    '''.format(dataset))
    robjects.r('''
        coxFit <- coxph(Surv(time = Survival, event = Death) ~ as.factor(groups2), data = survival, ties="exact")
        mfit <- survfit(Surv(Survival, Death == 1) ~ as.factor(groups2), data = survival)
        a <-intersect(unique(groups2), unique(groups));names(a) <- intersect(unique(groups2), unique(groups)); a[setdiff(unique(groups2), unique(groups))] <- seq(setdiff(unique(groups2), unique(groups)))+max(groups)
        colors <- a[levels(factor(groups2))]
        plot(mfit, col=colors, main = paste("Survival curves for ", "{}", ", level 2 (PertCluster)", sep=""), xlab = "Days", ylab="Survival", lwd=2)
        legend("top", legend = paste("Cox p-value:", round(summary(coxFit)$sctest[3],digits = 5), sep=""))
        legend("topright", fill=colors, legend=paste("Group ",levels(factor(groups2)), ": ", table(groups2)[levels(factor(groups2))], sep=""))
    '''.format(dataset, dataset, dataset, dataset, dataset, dataset, dataset, dataset, dataset, dataset))
    # obtain pertS connective matrices
    origGE = robjects.r['origGE']
    # origME = robjects.r['origME']
    origMI = robjects.r['origMI']

    pertGE = robjects.r['pertGE']
    # pertME = robjects.r['pertME']
    pertMI = robjects.r['pertMI']
    # print(type(pertGE))  # <class 'numpy.ndarray'>
    pd_pertGE = pd.DataFrame(pertGE)
    # pd_pertME = pd.DataFrame(pertME)
    pd_pertMI = pd.DataFrame(pertMI)
    pd_survival = pd.DataFrame(survival)
    pd_survival.to_csv('Projects/SubTyping/DataTCGA/{}/{}_survival.csv'.format(dataset, dataset))
    pd_pertGE.index = robjects.r['rowGE']
    pd_pertGE.columns = robjects.r['rowGE']
    # pd_pertME.index = robjects.r['rowME']
    # pd_pertME.columns = robjects.r['rowGE']
    pd_pertMI.index = robjects.r['rowMI']
    pd_pertMI.columns = robjects.r['rowGE']
    pd_pertGE.to_csv('result/DataTCGA2/{}/{}PertGE.csv'.format(dataset, dataset), header=True,
                     index=True)  # AttributeError: 'ListVector' object has no attribute 'to_csv'
    # pd_pertME.to_csv('result/DataTCGA2/{}/{}PertME.csv'.format(dataset, dataset), header=True,
    #                  index=True)  # AttributeError: 'ListVector' object has no attribute 'to_csv'
    pd_pertMI.to_csv('result/DataTCGA2/{}/{}PertMI.csv'.format(dataset, dataset), header=True,
                     index=True)  # AttributeError: 'ListVector' object has no attribute 'to_csv'

    # load perturbation result datasets
    DataGE = pd.read_csv('result/DataTCGA2/{}/{}PertGE.csv'.format(dataset, dataset))  # perturbation result
    # DataME = pd.read_csv('result/DataTCGA2/{}/{}PertME.csv'.format(dataset, dataset))
    DataMI = pd.read_csv('result/DataTCGA2/{}/{}PertMI.csv'.format(dataset, dataset))
    m = 'sqeuclidean'
    K = 20
    mu = 0.5

    # input [511 rows x 19580 columns] mydatGE mydatME mydatMI
    mydatGE = robjects.r['mydatGE']
    # mydatME = robjects.r['mydatME']
    mydatMI = robjects.r['mydatMI']
    mydatGE = pd.DataFrame(mydatGE)
    if dataset == 'KIRC' or dataset == 'LUSC' or dataset == 'BRCA' or dataset == 'GBM' or dataset == 'COAD':
        mydatGE = mydatGE.T
    # mydatME = pd.DataFrame(mydatME)
    # if dataset == 'LUSC' or dataset == 'BRCA' or dataset == 'GBM':
    #     mydatME = mydatME.T
    mydatMI = pd.DataFrame(mydatMI)
    if dataset == 'KIRC' or dataset == 'GBM':
        mydatMI = mydatMI.T
    mydatGE.index = robjects.r('row.names(mydatGE)')
    # mydatME.index = robjects.r('row.names(mydatME)')
    mydatMI.index = robjects.r('row.names(mydatMI)')
    mydatGE.columns = robjects.r('colnames(mydatGE)')
    # mydatME.columns = robjects.r('colnames(mydatME)')
    mydatMI.columns = robjects.r('colnames(mydatMI)')

    if mydatGE.shape[0] != mydatMI.shape[0]:  # or mydatGE.shape[0] != mydatMI.shape[0]
        print('Input files must have same samples.')
        exit(1)

    pd_DataGE = pd.DataFrame(pd_pertGE)
    # pd_DataME = pd.DataFrame(pd_pertME)
    pd_DataMI = pd.DataFrame(pd_pertMI)

    affinity_mydatGE = snf.make_affinity(mydatGE.iloc[:, 1:].values.astype(np.float), metric=m, K=K, mu=mu)
    # affinity_mydatME = snf.make_affinity(mydatME.iloc[:, 1:].values.astype(np.float), metric=m, K=K, mu=mu)
    affinity_mydatMI = snf.make_affinity(mydatMI.iloc[:, 1:].values.astype(np.float), metric=m, K=K, mu=mu)

    affinity_mydatGE = pd.DataFrame(affinity_mydatGE)
    # affinity_mydatME = pd.DataFrame(affinity_mydatME)
    affinity_mydatMI = pd.DataFrame(affinity_mydatMI)

    affinity_mydatGE.index = robjects.r('row.names(mydatGE)')
    # affinity_mydatME.index = robjects.r('row.names(mydatME)')
    affinity_mydatMI.index = robjects.r('row.names(mydatMI)')
    affinity_mydatGE.columns = robjects.r('row.names(mydatGE)')
    # affinity_mydatME.columns = robjects.r('row.names(mydatME)')
    affinity_mydatMI.columns = robjects.r('row.names(mydatMI)')

    affinity_nets = snf.make_affinity(
        [mydatGE.iloc[:, 1:].values.astype(np.float), mydatMI.iloc[:, 1:].values.astype(np.float)], metric=m, K=K,
        mu=mu)

    fused_net = snf_plus_altered_sim_all.snf_plus_altered_sim_GEMI(affinity_nets, pd_DataGE=pd_DataGE,
                                                                   pd_DataMI=pd_DataMI, K=K)
    print('snf done--!')

    print('Save fused adjacency matrix...')
    DataGEList = DataGE.columns.tolist()
    del DataGEList[0]
    fused_df = pd.DataFrame(fused_net)
    fused_df.columns = DataGEList
    fused_df.index = DataGEList
    fused_df.to_csv('result/DataTCGA2/{}/{}_GEMI_fused_matrix.csv'.format(dataset, dataset), header=True, index=True)
    np.fill_diagonal(fused_df.values, 0)

    print('spectral clustering...........')
    survivaldata = pd.DataFrame(survival)
    cluster_num = 4
    filename = 'result/DataTCGA2/{}/{}_GEMI_fused_matrix.csv'.format(dataset, dataset)
    datas = load_data(filename=filename)
    data = datas.iloc[:, 1:]
    y_pred = SpectralClustering(gamma=0.1, n_clusters=4).fit_predict(data)
    f = y_pred
    temp_f = []
    for f_num in f:
        temp_f.append(f_num + 1)
    f = temp_f
    # f = f.tolist()
    survivaldata = survivaldata.T
    survivaldata.insert(loc=3, column='groups', value=f)
    survivaltPath = 'result/DataTCGA2/{}/{}_GEMI_survival.csv'.format(dataset, dataset)
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
    proSurvival.to_csv(survivaltPath)
    # plot finial(spectral clistering) figure
    robjects.r('''
           BRCA_survival=read.csv("result/DataTCGA2/{}/{}_GEMI_survival.csv")
           vector1 <- BRCA_survival$groups
           column.names <- BRCA_survival$sample
           groups<-array(vector1,dimnames=list(column.names))

           coxFit <- coxph(Surv(time = Survival, event = Death) ~ as.factor(groups), data = BRCA_survival, ties="exact")
           mfit <- survfit(Surv(Survival, Death == 1) ~ as.factor(groups), data = BRCA_survival)
           plot(mfit, col=levels(factor(groups)), main = paste("Survival curves for ", "{}", ", level N (spectral clustering)", sep=""), xlab = "Days", ylab="Survival", lwd=2)
           legend("top", legend = paste("Cox p-value:", round(summary(coxFit)$sctest[3],digits = 5), sep=""))
           legend("topright", fill=levels(factor(groups)), legend=paste("Group ",levels(factor(groups)), ": ", table(groups)[levels(factor(groups))], sep=""))
           dev.off()
       '''.format(dataset, dataset, dataset))
    Rp_value = robjects.r('summary(coxFit)$sctest[3]')
    p_values3[dataset] = Rp_value[0]
    print("Calinski-Harabasz Score", metrics.calinski_harabasz_score(data, y_pred))
    plotRes(data, y_pred, cluster_num)
    # print("P values = ....{}".format(p_values))
    p_values_list.append("GEMI")
    p_values_list.append(p_values3)
    p_values_list.append("\n")
    print("P values list = ....{}".format(p_values_list))
'''
P values list = ....
'GEME', {'KIRC': 0.003661476134237417, 'LUSC': 0.42973806425159233, 'BRCA': 0.12314832385752228, 'LAML': 0.870707668141491, 'GBM': 0.39926645324935556, 'COAD': 0.6910796009923889},
'MEMI', {'KIRC': 0.3092359568387139, 'LUSC': 0.0016369029370659886, 'BRCA': 0.5254288562015846, 'LAML': 0.3645718938729171, 'GBM': 0.7228523329801897, 'COAD': 0.4679427874717487}, '\n',
'GEMI', {'KIRC': 0.05760250258899532, 'LUSC': 0.0543567013437559, 'BRCA': 0.3213426033449744, 'LAML': 0.03521272597521347, 'GBM': 0.570503034069115, 'COAD': 0.5876707861934682}, '\n',
'GEME', {'KIRC': 0.003661476134237417, 'LUSC': 0.42973806425159233, 'BRCA': 0.12314832385752228, 'LAML': 0.870707668141491, 'GBM': 0.39926645324935556, 'COAD': 0.6910796009923889}, '\n',
'MEMI', {'KIRC': 0.3092359568387139, 'LUSC': 0.0016369029370659886, 'BRCA': 0.5254288562015846, 'LAML': 0.3645718938729171, 'GBM': 0.7228523329801897, 'COAD': 0.4679427874717487}, '\n', 
'GEMI', {'KIRC': 0.05760250258899532, 'LUSC': 0.0543567013437559, 'BRCA': 0.3213426033449744, 'LAML': 0.03521272597521347, 'GBM': 0.570503034069115, 'COAD': 0.5876707861934682}, '\n', 
'GEME', {'KIRC': 0.003661476134237417, 'LUSC': 0.42973806425159233, 'BRCA': 0.12314832385752228, 'LAML': 0.870707668141491, 'GBM': 0.39926645324935556, 'COAD': 0.6910796009923889}, '\n',
'MEMI', {'KIRC': 0.3092359568387139, 'LUSC': 0.0016369029370659886, 'BRCA': 0.5254288562015846, 'LAML': 0.3645718938729171, 'GBM': 0.7228523329801897, 'COAD': 0.4679427874717487}, '\n',
'GEMI', {'KIRC': 0.05760250258899532, 'LUSC': 0.0543567013437559, 'BRCA': 0.3213426033449744, 'LAML': 0.03521272597521347, 'GBM': 0.570503034069115, 'COAD': 0.5876707861934682}, '\n', 
'GEME', {'KIRC': 0.003661476134237417, 'LUSC': 0.42973806425159233, 'BRCA': 0.12314832385752228, 'LAML': 0.870707668141491, 'GBM': 0.39926645324935556, 'COAD': 0.6910796009923889}, '\n',
'MEMI', {'KIRC': 0.3092359568387139, '    LUSC': 0.0016369029370659886, 'BRCA': 0.5254288562015846, 'LAML': 0.3645718938729171, 'GBM': 0.7228523329801897, 'COAD': 0.4679427874717487}, '\n', 
'GEMI', {'KIRC': 0.05760250258899532, 'LUSC': 0.0543567013437559, 'BRCA': 0.3213426033449744, 'LAML': 0.03521272597521347, 'GBM': 0.570503034069115, 'COAD': 0.5876707861934682}, '\n', 
'GEME', {'KIRC': 0.003661476134237417, 'LUSC': 0.42973806425159233, 'BRCA': 0.12314832385752228, 'LAML': 0.870707668141491, 'GBM': 0.39926645324935556, 'COAD': 0.6910796009923889}, '\n',
'MEMI', {'KIRC': 0.3092359568387139, 'LUSC': 0.0016369029370659886, 'BRCA': 0.5254288562015846, 'LAML': 0.3645718938729171, 'GBM': 0.7228523329801897, 'COAD': 0.4679427874717487}, '\n', '
'GEMI', {'KIRC': 0.05760250258899532, 'LUSC': 0.0543567013437559, 'BRCA': 0.3213426033449744, 'LAML': 0.03521272597521347, 'GBM': 0.570503034069115, 'COAD': 0.5876707861934682}, '\n',
'GEME', {'KIRC': 0.003661476134237417, 'LUSC': 0.42973806425159233, 'BRCA': 0.12314832385752228, 'LAML': 0.870707668141491, 'GBM': 0.39926645324935556, 'COAD': 0.6910796009923889}, '\n', 
'MEMI', {'KIRC': 0.3092359568387139, 'LUSC': 0.0016369029370659886, 'BRCA': 0.5254288562015846, 'LAML': 0.3645718938729171, 'GBM': 0.7228523329801897, 'COAD': 0.4679427874717487}, 
'GEMI', {'KIRC': 0.05760250258899532, 'LUSC': 0.0543567013437559, 'BRCA': 0.3213426033449744, 'LAML': 0.03521272597521347, 'GBM': 0.570503034069115, 'COAD': 0.5876707861934682}, '\n']
'''
    # break
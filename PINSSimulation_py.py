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
import snf_plus_altered_sim_all
robjects.r('rm(list=ls())')

numpy2ri.activate()  # From numpy to rpy2

# import R packages
rpackages.importr('base')
rpackages.importr('utils')
rpackages.importr('stats')
rpackages.importr('PINS')
rpackages.importr('survival')
rpackages.importr('flexclust')
rpackages.importr('ConsensusClusterPlus')
rpackages.importr('iClusterPlus')
rpackages.importr('SNFtool')


def snf_plus_altered_sim_1(*aff,pd_DataGE, K=20, t=20, alpha=1.0):
    aff = snf.compute._check_SNF_inputs(aff)
    Wk = [0] * len(aff)
    Wsum = np.zeros(aff[0].shape)

    # get number of modalities informing each subject x subject affinity
    n_aff = len(aff) - np.sum([np.isnan(a) for a in aff], axis=0)

    for n, mat in enumerate(aff):
        # normalize affinity matrix based on strength of edges
        mat = mat / np.nansum(mat, axis=1, keepdims=True)
        aff[n] = sklearn.utils.validation.check_symmetric(mat,
                                                          raise_warning=False)  # sklearn.utils.validation.check_symmetric
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
                        # print(i,j,n)
                        # print(temp_DF.iloc[i, j])
                        # print(pd_DataGE.iloc[i, j])
                        # print(type(temp_DF.iloc[i, j]))
                        # print(type(pd_DataGE.iloc[i, j]))
                        temp_DF.iloc[i, j] = temp_DF.iloc[i, j] / pd_DataGE.iloc[i, j]
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
            aw = np.nan_to_num(np.nansum(aff, axis=0))
            # propagate `Wsum` through masked affinity matrix (`nzW`)
            aff0 = nzW @ (Wsum - aw) @ nzW.T  # TODO: / by 0
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
    
    PINSPath="result/PINSResult/"
    CCPath="result/CCResult/"
    SNFPath="result/SNFResult/"
    iClusterPlusPath="result/iClusterPlusResult/"
    pdfPath="result/Plots/Figures/"
    
    #standard uniform with 1 cluster
    nrow=100;ncol=1000
''')
p_values = {'diyige': 1, 'METABRIC_validation': 1}
datasets = robjects.r('datasets=c("METABRIC_discovery","METABRIC_validation")')
SC_values = []
AH_values = []
Scores = []  # Silhouette Coefficient, if it only has 1 class, it will be 0
Scores1 = []  # AH Coefficient, if it only has 1 class, it will be 0
m = 'sqeuclidean'
K = 20
mu = 0.5
robjects.r('''
    set.seed(1)
    dataU <- matrix(runif(nrow*ncol, 0, 1), nrow=nrow, ncol=ncol)
    # resultU = PerturbationClustering(data = dataU)
    resultU <- PerturbationClustering(dataU)
    resultFile=paste(resultPath, "PINS_Uniform1.RData", sep="")
    save(dataU, resultU, file=resultFile)
    
    
    origGE=resultU[["origS"]][[resultU[["k"]]]]
    # PWGE = dataTypeResult[[1]]$origS[[dataTypeResult[[1]]$k]]
    pertGE=resultU[["pertS"]][[resultU[["k"]]]]
    
''')
dataU = robjects.r['dataU']
origGE = robjects.r['origGE']
pertGE = robjects.r['pertGE']

m = 'sqeuclidean'
K = 20
mu = 0.5

# mydatGE = robjects.r['mydatGE']
# mydatGE = pd.DataFrame(mydatGE)
groups_y = []
for cluster_num in range(2, 11):
    pertGE = robjects.r['pertGE']
    pd_pertGE = pd.DataFrame(pertGE)
    affinity_nets = snf.make_affinity(pd_pertGE.iloc[:, 1:].values.astype(np.float), pd_pertGE.iloc[:, 1:].values.astype(np.float), pd_pertGE.iloc[:, 1:].values.astype(np.float), metric=m, K=K, mu=mu)
    fused_net = snf_plus_altered_sim_all.snf_plus_altered_sim(affinity_nets, pd_DataGE=pd_pertGE, pd_DataME=pd_pertGE, pd_DataMI=pd_pertGE, K=K)
    print('snf done--!')
    print('Save fused adjacency matrix...')
    DataGEList = pd_pertGE.columns.tolist()
    # del DataGEList[0]
    fused_df = pd.DataFrame(fused_net)
    np.fill_diagonal(fused_df.values, 0)
    print('spectral clustering...........')
    y_pred = SpectralClustering(gamma=0.5, n_clusters=cluster_num).fit_predict(fused_df)
    groups_y.append(y_pred)
    Scores.append(metrics.silhouette_score(fused_df, y_pred, metric='euclidean'))
    Scores1.append(metrics.calinski_harabasz_score(fused_df, y_pred))
SC_values.append(Scores)
AH_values.append(Scores1)
print('Scores', Scores)
print('Scores1', Scores1)
print("Silhouette Coefficient Score", Scores)
temp_Scores = []
for j in range(0, 9):
    temp_Scores.append(math.log(10, np.abs(Scores[j])))
print("len(Scores)", len(Scores))
print("len(temp_Scores)", len(temp_Scores))
print("len(Scores1)", len(Scores1))
print("temp_Scores", temp_Scores)
print("AH Score", Scores1)
i = range(0, 9)
plt.xlabel('k')
plt.ylabel('value')
plt.plot(i, temp_Scores[:9], 'g.-', i, Scores1[:9], 'b.-')
plt.savefig('result/Simulation/Simulation_SCAH.png')









# for cluster_num in range(2,11):
#     print("cluster_num", cluster_num)
#     resultUPerts = robjects.r('resultU[["pertS"]][[{}]]'.format(cluster_num))
#     resultUPerts = pd.DataFrame(resultUPerts)
#     affinity_nets = snf.make_affinity([dataU], metric=m, K=K, mu=mu)
#     fused_net = snf_plus_altered_sim_1(affinity_nets, pd_DataGE=resultUPerts, K=K)
#     print('snf done--!')
#
#     print('Save fused adjacency matrix...')
#     DataGEList = resultUPerts.columns.tolist()
#     # del DataGEList[0]
#     fused_df = pd.DataFrame(fused_net)
#     fused_df.columns = DataGEList
#     fused_df.index = DataGEList
#     fused_df.to_csv('result/Simulation/1_fused_matrix.csv', header=True, index=True)
#     np.fill_diagonal(fused_df.values, 0)
#     print('spectral clustering...........')
#     filename = 'result/Simulation/1_fused_matrix.csv'
#     datas = load_data(filename=filename)
#     print('len datas')
#     print(len(datas.index), len(datas.columns))
#     data = datas
#     groups_y = []
#     for cluster_num in range(2, 11):
#         print("cluster_num", cluster_num)
#         # resultUPerts
#         y_pred = SpectralClustering(gamma=0.5, n_clusters=cluster_num).fit_predict(data)
#         groups_y.append(y_pred)
#         Scores.append(metrics.silhouette_score(data, y_pred, metric='euclidean'))
#         Scores1.append(metrics.calinski_harabasz_score(data, y_pred))
#         # Rp_value = robjects.r('summary(coxFit)$sctest[3]')
#         # p_values['diyige'] = Rp_value[0]
#         SC_values.append(Scores)
#         AH_values.append(Scores1)
#         print('Scores', Scores)
#         print('Scores1', Scores1)
#         print("Silhouette Coefficient Score", Scores)
#         temp_Scores = []
#         for j in range(len(Scores)):
#             temp_Scores.append(math.log(10, np.abs(Scores[j])))
#         print("len(Scores)", len(Scores))
#         print("len(temp_Scores)", len(temp_Scores))
#         print("len(Scores1)", len(Scores1))
#         print("temp_Scores", temp_Scores)
#         print("AH Score", Scores1)
#         i = range(2, 11)
#         plt.xlabel('k')
#         plt.ylabel('value')
#         print("temp_Scores", temp_Scores)
#         print("Scores1", Scores1)
#         plt.plot(i, temp_Scores[:11], 'g.-', i, Scores1[:11], 'b.-')






robjects.r('''
    resultFile=paste(resultPath, "PINS_Uniform1.RData", sep="")
    save(dataU, resultU, file=resultFile)


    #Standard Gaussian with 1 cluster
    nrow=100;ncol=1000
    Kmax=10

    set.seed(1)
    dataG <- matrix(rnorm(nrow*ncol, 0, 1), nrow=nrow, ncol=ncol)

    resultG <- PerturbationClustering(data=dataG)
''')
robjects.r('''
    loop=20
    AUC=matrix(NA, ncol=10, nrow=loop)
    AUC[1,]=resultG$Discrepancy$AUC
    for (i in 2:loop) {
      dataTMP <- matrix(rnorm(nrow*ncol, 0, 1), nrow=nrow, ncol=ncol)
      resultTMP = PerturbationClustering(data=dataTMP)
      AUC[i,]=resultTMP$Discrepancy$AUC
    }
    resultFile= paste(resultPath, "PINS_Gaussian1.RData", sep="")
    save(dataG, resultG, AUC, file=resultFile)

''')
robjects.r('''
    # Gaussian datasets with 2-10 classes
    Kmax=10
    nrow=100;ncol=1000
    for (classes in 2:10) {
      set.seed(1)
      dataG <- matrix(rnorm(nrow*ncol, 0, 1), nrow=nrow, ncol=ncol)
      rownames(dataG)=seq(nrow)

      str=NULL
      for (i in 1:classes) {
        str=c(str, rep(i, nrow/classes))
      }
      if (length(str)<nrow) {str=c(str, rep(classes, nrow-length(str)))}
      group=data.frame(row.names=seq(nrow),Sample=seq(nrow), Group=str)

      for (i in 1:classes) {
        dataG[rownames(group)[group[,2]==i], (100*(i-1)+1):(100*i)]=dataG[rownames(group)[group[,2]==i], (100*(i-1)+1):(100*i)]+2
      }

      resultG=PerturbationClustering(data=dataG)

      resultFile=paste(resultPath, "PINS_Gaussian", classes, ".RData", sep="")
      save(dataG, resultG, group, file=resultFile)
    }




''')
robjects.r('''

    # Testing the time for PINS
    dataG <- matrix(rnorm(200*10000, 0, 1), nrow=200, ncol=10000)
    T=NULL
    X=seq(4,20,by=2)
    for (i in 1:length(X)) {
        T[i]=system.time(PerturbationClustering(data = dataG, Kmax = X[i]))[3]
    }
    save(T,X,dataG,file=paste(PINSPath,"SimulationKmax.RData",sep=""))

''')
robjects.r('''
    pdfFile=paste(pdfPath,"PINS_Kmax_Simulation.pdf", sep="")
    pdf(pdfFile)
    par(tcl=0.3,mgp=c(1.7,0.4,0),mar=c(3,3,2.5,1))
    plot(X, T/60,xaxt='n', xlab="Maximum number of clusters (K)", ylab="Running time (minute)", main="Effect of K on PINS's running time", cex.lab=1.4, cex.axis=1.3, cex.main=1.7, col="red")
    lines(X,T/60, lwd=2, col="blue")
    axis(side=1, at=X, labels=X,cex.axis=1.3)
    dev.off()




''')
robjects.r('''
    ############ Check the sensibility of noise variance
    library(flexclust)
    library(PINS)
    library(ConsensusClusterPlus)
    library(iClusterPlus)
    library(SNFtool)
    nrow=100;ncol=1000
    classes=9

    MValues=c(4,3,2,1,0.9,0.8,0.7,0.6)
    ARI_PINS=NULL
    ARI_SNF=NULL
    sigma=NULL

    CCPath="Projects/Subtyping/PackageAndTesting/CCResult/"
    iClusterPlusPath="Projects/Subtyping/PackageAndTesting/iClusterPlusResult/"
''')
robjects.r('''    
    # CCPath="/wsu/home/ex/ex60/ex6091/Subtyping/PackageAndTesting/CCResult/"
    # iClusterPlusPath="/wsu/home/ex/ex60/ex6091/Subtyping/PackageAndTesting/iClusterPlusResult/"
    
    #PINS
    classes=9
    MValues=c(4,3,2,1,0.9,0.8,0.7,0.6)
    sigma=NULL
    ARI_PINS=NULL
    ARI_SNF=NULL
    for (ind in 1:length(MValues)) {
        set.seed(1)
        M=MValues[ind]     
        dataG <- matrix(rnorm(nrow*ncol, 0, 1), nrow=nrow, ncol=ncol)
        rownames(dataG)=seq(nrow)
        str=NULL
        for (i in 1:classes) {
            str=c(str, rep(i, nrow/classes))
        }
        if (length(str)<nrow) {str=c(str, rep(classes, nrow-length(str)))}
        group=data.frame(row.names=seq(nrow),Sample=seq(nrow), Group=str)
        for (i in 1:classes) {
            dataG[rownames(group)[group[,2]==i], (100*(i-1)+1):(100*i)]=dataG[rownames(group)[group[,2]==i], (100*(i-1)+1):(100*i)] + M
        }
        
        sds=apply(dataG, FUN=var, MARGIN=2)
        sigma[ind]=median(sds)
        
        # PINS
        result=PerturbationClustering(dataG)
        ARI_PINS[ind]=randIndex(group[,2],result$groups, correct = TRUE)
    }
    sigma
    ARI_PINS
    
''')
robjects.r('''        
    # SNF
    for (ind in 1:length(MValues)) {
        set.seed(1)
        M=MValues[ind]
        
        dataG <- matrix(rnorm(nrow*ncol, 0, 1), nrow=nrow, ncol=ncol)
        rownames(dataG)=seq(nrow)
        str=NULL
        for (i in 1:classes) {
            str=c(str, rep(i, nrow/classes))
        }
        if (length(str)<nrow) {str=c(str, rep(classes, nrow-length(str)))}
        group=data.frame(row.names=seq(nrow),Sample=seq(nrow), Group=str)
        for (i in 1:classes) {
            dataG[rownames(group)[group[,2]==i], (100*(i-1)+1):(100*i)]=dataG[rownames(group)[group[,2]==i], (100*(i-1)+1):(100*i)] + M
        }
        # SNF
        K = 20;##number of neighbors, usually (10~30)
        alpha = 0.5; ##hyperparameter, usually (0.3~0.8)
        NIT = 10; ###Number of Iterations, usually (10~20)
        data=standardNormalization(dataG)
        PSMgeneE = dist2(as.matrix(data),as.matrix(data));
        W1 = affinityMatrix(PSMgeneE, K, alpha)
        C = estimateNumberOfClustersGivenGraph(W1, NUMC=2:10)  #number of clusters
        groupSNF = spectralClustering(W1,C[[1]])
        ARI_SNF[ind]=randIndex(group[,2],groupSNF, correct = TRUE)
    }
    ARI_SNF
     
 ''')
robjects.r('''   
    # CC results
    MValues=c(4,3,2,1,0.9,0.8,0.7,0.6)
    classes=9
    CCPath="Projects/Subtyping/PackageAndTesting/CCResult/"
    library(flexclust)
    library(PINS)
    library(ConsensusClusterPlus)
    library(iClusterPlus)
    library(SNFtool)
    for (ind in 1:length(MValues)) {
        set.seed(1)
        M=MValues[ind]
        
        dataG <- matrix(rnorm(nrow*ncol, 0, 1), nrow=nrow, ncol=ncol)
        rownames(dataG)=seq(nrow)
        str=NULL
        for (i in 1:classes) {
            str=c(str, rep(i, nrow/classes))
        }
        if (length(str)<nrow) {str=c(str, rep(classes, nrow-length(str)))}
        group=data.frame(row.names=seq(nrow),Sample=seq(nrow), Group=str)
        for (i in 1:classes) {
            dataG[rownames(group)[group[,2]==i], (100*(i-1)+1):(100*i)]=dataG[rownames(group)[group[,2]==i], (100*(i-1)+1):(100*i)] + M
        }
        
        # CC
        path=paste(CCPath, "Simulation_", M,sep="")
        if (!file.exists(path)) dir.create(path)
        d <- t(dataG)
        d = sweep(d,1, apply(d,1,median,na.rm=T))
        results = ConsensusClusterPlus(d,maxK=10,reps=1000, pItem=0.8,pFeature=1, title=path,clusterAlg="hc",distance="pearson", plot="png", seed=888)
        resultFile=paste(CCPath,"CC_Simulation_", M, ".RData" ,sep="")  
        save(dataG, results, group, file=resultFile)
    }
    
    
    
    k=c(9, 9, 9, 9, 9, 9, 9, 9)
    ARI_CC=NULL
    for (ind in 1:length(MValues)) {
        M=MValues[ind]
        resultFile=paste(CCPath,"CC_Simulation_", M, ".RData" ,sep="") 
        load(resultFile)
        memb=results[[k[ind]]]$consensusClass
        ARI_CC[ind]=randIndex(group$Group,memb)
    }
    ARI_CC
    
     
    
    # iClusterPlus
    for (ind in 1:length(MValues)) {
        set.seed(1)
        M=MValues[ind]
        
        dataG <- matrix(rnorm(nrow*ncol, 0, 1), nrow=nrow, ncol=ncol)
        rownames(dataG)=seq(nrow)
        str=NULL
        for (i in 1:classes) {
            str=c(str, rep(i, nrow/classes))
        }
        if (length(str)<nrow) {str=c(str, rep(classes, nrow-length(str)))}
        group=data.frame(row.names=seq(nrow),Sample=seq(nrow), Group=str)
        for (i in 1:classes) {
            dataG[rownames(group)[group[,2]==i], (100*(i-1)+1):(100*i)]=dataG[rownames(group)[group[,2]==i], (100*(i-1)+1):(100*i)] + M
        }
        
        # iClusterPlus
        cv.fit=alist()
        for (k in 1:9) {
            cv.fit[[k]]=tune.iClusterPlus(cpus=1, dt1=dataG, K=k, type=c("gaussian"))
        }
        resultFile=paste(iClusterPlusPath,"iClusterPlus_Simulation_", M, ".RData" ,sep="")  
        save(dataG, cv.fit, group, file=resultFile)
    }
    
    ARI_iCluster=NULL
    k=c(9,9,9,1,1,1,1,1)
    mc.cores<=1
    n.cores<=1
    for (ind in 1:length(MValues)) {
        mc.cores<=1
        n.cores<=1
        M=MValues[ind]
        resultFile=paste(iClusterPlusPath,"iClusterPlus_Simulation_", M, ".RData" ,sep="")  
        load(resultFile)
        
        nK = length(cv.fit)
        BIC=getBIC(cv.fit)
        devR = getDevR(cv.fit) 
        minBICid = apply(BIC,2,which.min)
        devRatMinBIC = rep(NA,nK)
        for(i in 1:nK){
            mc.cores<=1
            devRatMinBIC[i] = devR[minBICid[i],i]
        } 
        plot(devRatMinBIC)
        
        
        clusters=getClusters(cv.fit)
        rownames(clusters)=rownames(dataG)
        colnames(clusters)=paste("K=",2:(length(cv.fit)+1),sep="")
        memb=clusters[,k[ind]]
        
        ARI_iCluster[ind]=randIndex(memb,group[,2], correct = T)
''')
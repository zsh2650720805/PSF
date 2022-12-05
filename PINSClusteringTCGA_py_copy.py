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
from sklearn.cluster import SpectralClustering
from sklearn import metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
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


def snf_plus_altered_sim(*aff, pd_DataGE, pd_DataME, pd_DataMI, K=20, t=20, alpha=1.0):
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


def full_snf_plus_cluster(*aff, pd_DataGE, pd_DataME, pd_DataMI, K=20, t=20, alpha=1.0):
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
        aff[n] = sklearn.utils.validation.check_symmetric(mat,
                                                          raise_warning=False)  # sklearn.utils.validation.check_symmetric
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
        knn_aff[n] = sklearn.utils.validation.check_symmetric(mat,
                                                              raise_warning=False)  # sklearn.utils.validation.check_symmetric
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
robjects.r('resultPath="result/DataTCGA/"')
Kmax = robjects.r('Kmax=10')  # it should be 10
iter = robjects.r('iter=200')  # it should be 200
noisePercent = robjects.r('noisePercent="med"')
kmIter = robjects.r('kmIter=200')  # it should be 200
datasets = robjects.r('datasets=c("KIRC", "LUSC", "BRCA", "LAML", "GBM", "COAD")')

# start for cycle
p_values = {'KIRC': 1, 'LUSC': 1, 'BRCA': 1, 'LAML': 1, 'GBM': 1, 'COAD': 1}
SC_values = []
AH_values = []
Scores = []  # Silhouette Coefficient, if it only has 1 class, it will be 0
Scores1 = []  # AH Coefficient, if it only has 1 class, it will be 0
for dataset in datasets:
    robjects.r('''
        rdataset="{}"
    '''.format(dataset))
    robjects.r('message(paste("Dataset: ", "{}", sep=""))'.format(dataset))
    robjects.r('set.seed(1)')
    file = robjects.r('file=paste(dataPath,"%s", "/", "%s", "_ProcessedData.RData" ,sep="")' % (dataset, dataset))
    robjects.r('load(%s) ' % (file.r_repr()))
    t1 = robjects.r('t1=Sys.time()')

    # reload multi-omics data and survival that only keep common data
    patients = robjects.r('''
        options (warn = -1)
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
    ''')

    survival = robjects.r('survival=survival[patients,]')
    survival_idx = robjects.r('row.names(survival)')
    survival_col = robjects.r('colnames(survival)')

    # load clinical data
    clinical = robjects.r('''
        clinical <- read.table(file=paste(dataPath, "{}", "/", "{}" ,"_Clinical.txt", sep=""), sep="\t", header=T, row.names=1,stringsAsFactors = F, fill=T)
        clinical <- clinical[-1,];clinical <- clinical[-1,]
        a<-rownames(clinical)
        rownames(clinical)<-paste(substr(a,1,4),substr(a,6,7),substr(a,9,12),sep=".")
        clinical <- clinical[rownames(survival),]
    '''.format(dataset, dataset))

    robjects.r('''
        rowGE<-row.names(mydatGE)
        rowME<-row.names(mydatME)
        rowMI<-row.names(mydatMI)
        colGE<-colnames(mydatGE)
        colME<-colnames(mydatME)
        colMI<-colnames(mydatMI)
    ''')

    # set dataList
    Kmax = robjects.r('Kmax=10')
    dataList = robjects.r('dataList <- list (mydatGE, mydatME, mydatMI)')
    robjects.r('names(dataList) = c("GE", "ME", "MI")')

    # run perturbation
    result = robjects.r('result=SubtypingOmicsData(dataList = dataList, Kmax = Kmax, noisePercent = noisePercent, iter = iter, kmIter = kmIter)')
    robjects.r('''
        agreementCutoff=0.5
        dataTypeResult <- list()
        for (i in 1:length(dataList)) {
            message(paste("Data type: ", i, sep=""))
            dataTypeResult[[i]] <- PerturbationClustering(data=dataList[[i]], Kmax=Kmax, noisePercent="med", iter=iter, kmIter=kmIter)
        }
        origGE=dataTypeResult[[1]]$origS[[dataTypeResult[[1]]$k]]
        PWGE = dataTypeResult[[1]]$origS[[dataTypeResult[[1]]$k]]
        pertGE=dataTypeResult[[1]]$pertS[[dataTypeResult[[1]]$k]]


        origME=dataTypeResult[[2]]$origS[[dataTypeResult[[1]]$k]]
        PWME = dataTypeResult[[2]]$origS[[dataTypeResult[[1]]$k]]
        pertME=dataTypeResult[[2]]$pertS[[dataTypeResult[[1]]$k]]

        origMI=dataTypeResult[[3]]$origS[[dataTypeResult[[1]]$k]]
        PWMI = dataTypeResult[[3]]$origS[[dataTypeResult[[1]]$k]]
        pertMI=dataTypeResult[[3]]$pertS[[dataTypeResult[[1]]$k]]
    ''')
    robjects.r('resultFile=paste(resultPath, "{}", "/", "PINS_", "{}", ".RData" ,sep="")'.format(dataset, dataset))

    robjects.r('pdfFile=paste(resultPath, "{}", "/", "PINS_", "{}", ".pdf" ,sep="")'.format(dataset, dataset))
    robjects.r('pdf(pdfFile)')
    # obtain pertS connective matrices
    origGE = robjects.r['origGE']
    origME = robjects.r['origME']
    origMI = robjects.r['origMI']

    m = 'sqeuclidean'
    K = 20
    mu = 0.5

    # input [511 rows x 19580 columns] mydatGE mydatME mydatMI
    mydatGE = robjects.r['mydatGE']
    mydatME = robjects.r['mydatME']
    mydatMI = robjects.r['mydatMI']
    norm_mydatGE = robjects.r['norm_mydatGE']
    norm_mydatME = robjects.r['norm_mydatME']
    norm_mydatMI = robjects.r['norm_mydatMI']
    norm_mydatGE = pd.DataFrame(norm_mydatGE)
    norm_mydatME = pd.DataFrame(norm_mydatME)
    norm_mydatMI = pd.DataFrame(norm_mydatMI)

    mydatGE = pd.DataFrame(mydatGE)
    GEindex = robjects.r('row.names(mydatGE)')
    MEindex = robjects.r('row.names(mydatME)')
    MIindex = robjects.r('row.names(mydatMI)')
    if dataset == 'KIRC' or dataset == 'LUSC' or dataset == 'BRCA' or dataset == 'GBM' or dataset == 'COAD':
        mydatGE = mydatGE.T
    mydatME = pd.DataFrame(mydatME)
    if dataset == 'LUSC' or dataset == 'BRCA' or dataset == 'GBM':
        mydatME = mydatME.T
    mydatMI = pd.DataFrame(mydatMI)
    if dataset == 'KIRC' or dataset == 'GBM':
        mydatMI = mydatMI.T

    print('mydatGE.index=={}, mydatGE.col=={}'.format(len(mydatGE.index), len(mydatGE.columns)))
    print('mydatME.index=={}, mydatME.col=={}'.format(len(mydatME.index), len(mydatME.columns)))
    print('mydatMI.index=={}, mydatMI.col=={}'.format(len(mydatMI.index), len(mydatMI.columns)))

    if mydatGE.shape[0] != mydatME.shape[0] or mydatGE.shape[0] != mydatMI.shape[0]:
        print('Input files must have same samples.')
        exit(1)

    recordP = []
    groups_y = []
    groups_y_GE = []
    groups_y_ME = []
    groups_y_MI = []
    temp_Pvalues = {"cluster_num": 0, "minPvalues": 1, "f": 1}
    temp_Pvalues_GE = {"cluster_num_GE": 0, "minPvalues_GE": 1, "f_GE": 1}
    temp_Pvalues_ME = {"cluster_num_ME": 0, "minPvalues_ME": 1, "f_ME": 1}
    temp_Pvalues_MI = {"cluster_num_MI": 0, "minPvalues_MI": 1, "f_MI": 1}

    for cluster_num in range(2, 11):
        f_path = "result/DataTCGA/{}/f_{}_{}.csv".format(dataset, dataset, cluster_num)
        robjects.r('''
            pertGE=dataTypeResult[[1]][["pertS"]][[{}]]
            pertME=dataTypeResult[[2]][["pertS"]][[{}]]
            pertMI=dataTypeResult[[3]][["pertS"]][[{}]]
        '''.format(cluster_num, cluster_num, cluster_num))
        pertGE = robjects.r['pertGE']
        pertME = robjects.r['pertME']
        pertMI = robjects.r['pertMI']

        pd_pertGE = pd.DataFrame(pertGE)
        pd_pertME = pd.DataFrame(pertME)
        pd_pertMI = pd.DataFrame(pertMI)
        pd_survival = pd.DataFrame(survival)
        pd_survival.to_csv('Projects/SubTyping/DataTCGA/{}/{}_survival.csv'.format(dataset, dataset))
        pd_pertGE.index = robjects.r['rowGE']
        pd_pertGE.columns = robjects.r['rowGE']
        pd_pertME.index = robjects.r['rowME']
        pd_pertME.columns = robjects.r['rowME']
        pd_pertMI.index = robjects.r['rowMI']
        pd_pertMI.columns = robjects.r['rowMI']

        pd_pertGE.to_csv('result/DataTCGA/{}/{}PertGE.csv'.format(dataset, dataset), header=True,
                         index=True)
        pd_pertME.to_csv('result/DataTCGA/{}/{}PertME.csv'.format(dataset, dataset), header=True,
                         index=True)
        pd_pertMI.to_csv('result/DataTCGA/{}/{}PertMI.csv'.format(dataset, dataset), header=True,
                         index=True)

        # load perturbation result datasets
        DataGE = pd.read_csv('result/DataTCGA/{}/{}PertGE.csv'.format(dataset, dataset))
        DataME = pd.read_csv('result/DataTCGA/{}/{}PertME.csv'.format(dataset, dataset))
        DataMI = pd.read_csv('result/DataTCGA/{}/{}PertMI.csv'.format(dataset, dataset))

        pd_DataGE = pd.DataFrame(pd_pertGE)
        pd_DataME = pd.DataFrame(pd_pertME)
        pd_DataMI = pd.DataFrame(pd_pertMI)

        affinity_mydatGE = snf.make_affinity(norm_mydatGE.iloc[:, 1:].values.astype(np.float), metric=m, K=K, mu=mu)
        affinity_mydatME = snf.make_affinity(norm_mydatME.iloc[:, 1:].values.astype(np.float), metric=m, K=K, mu=mu)
        affinity_mydatMI = snf.make_affinity(norm_mydatMI.iloc[:, 1:].values.astype(np.float), metric=m, K=K, mu=mu)

        affinity_mydatGE = pd.DataFrame(affinity_mydatGE)
        affinity_mydatME = pd.DataFrame(affinity_mydatME)
        affinity_mydatMI = pd.DataFrame(affinity_mydatMI)

        affinity_mydatGE.index = robjects.r('row.names(mydatGE)')
        affinity_mydatME.index = robjects.r('row.names(mydatME)')
        affinity_mydatMI.index = robjects.r('row.names(mydatMI)')
        affinity_mydatGE.columns = robjects.r('row.names(mydatGE)')
        affinity_mydatME.columns = robjects.r('row.names(mydatME)')
        affinity_mydatMI.columns = robjects.r('row.names(mydatMI)')

        affinity_nets = snf.make_affinity(
            [mydatGE.iloc[:, 1:].values.astype(np.float), mydatME.iloc[:, 1:].values.astype(np.float),
             mydatMI.iloc[:, 1:].values.astype(np.float)],
            metric=m, K=K, mu=mu)
        affinity_nets_GE = snf.make_affinity(
            [mydatGE.iloc[:, 1:].values.astype(np.float), mydatGE.iloc[:, 1:].values.astype(np.float),
             mydatGE.iloc[:, 1:].values.astype(np.float)],
            metric=m, K=K, mu=mu)
        affinity_nets_ME = snf.make_affinity(
            [mydatME.iloc[:, 1:].values.astype(np.float), mydatME.iloc[:, 1:].values.astype(np.float),
             mydatME.iloc[:, 1:].values.astype(np.float)],
            metric=m, K=K, mu=mu)
        affinity_nets_MI = snf.make_affinity(
            [mydatMI.iloc[:, 1:].values.astype(np.float), mydatMI.iloc[:, 1:].values.astype(np.float),
             mydatMI.iloc[:, 1:].values.astype(np.float)],
            metric=m, K=K, mu=mu)

        fused_net = snf_plus_altered_sim(affinity_nets, pd_DataGE=pd_DataGE, pd_DataME=pd_DataME, pd_DataMI=pd_DataMI, K=K)
        fused_net_GE = snf_plus_altered_sim(affinity_nets_GE, pd_DataGE=pd_DataGE, pd_DataME=pd_DataGE, pd_DataMI=pd_DataGE, K=K)
        fused_net_ME = snf_plus_altered_sim(affinity_nets_ME, pd_DataGE=pd_DataME, pd_DataME=pd_DataME, pd_DataMI=pd_DataME, K=K)
        fused_net_MI = snf_plus_altered_sim(affinity_nets_MI, pd_DataGE=pd_DataMI, pd_DataME=pd_DataMI, pd_DataMI=pd_DataMI, K=K)
        print('snf done......................')

        fused_net_GE = pd.DataFrame(fused_net_GE)
        fused_net_ME = pd.DataFrame(fused_net_ME)
        fused_net_MI = pd.DataFrame(fused_net_MI)

        print('Save fused adjacency matrix...')
        DataGEList = DataGE.columns.tolist()
        del DataGEList[0]
        fused_df = pd.DataFrame(fused_net)
        fused_df.columns = DataGEList
        fused_df.index = DataGEList
        fused_df.to_csv('result/DataTCGA/{}/{}_fused_matrix.csv'.format(dataset, dataset), header=True, index=True)
        np.fill_diagonal(fused_df.values, 0)
        print("fused_df", "\n", fused_df)
        print('spectral clustering...........')
        print("{} dataset cluster {}".format(dataset, cluster_num))
        origin_survivaldata = pd.DataFrame(survival)

        filename = 'result/DataTCGA/{}/{}_fused_matrix.csv'.format(dataset, dataset)
        datas = load_data(filename=filename)
        data = datas.iloc[:, 1:]
        survivaldata = origin_survivaldata
        y_pred = SpectralClustering(gamma=0.5, n_clusters=cluster_num).fit_predict(data)
        groups_y.append(y_pred)
        Scores.append(metrics.silhouette_score(data, y_pred, metric='euclidean'))
        Scores1.append(metrics.calinski_harabasz_score(data, y_pred))
        f = groups_y[cluster_num - 2]
        temp_f = []
        for f_num in f:
            temp_f.append(f_num + 1)
        f = temp_f
        f = pd.DataFrame(f)
        f.to_csv(f_path, header=True, index=True)


        fused_net_GE = fused_net_GE.iloc[:, 1:]
        fused_net_ME = fused_net_ME.iloc[:, 1:]
        fused_net_MI = fused_net_MI.iloc[:, 1:]

        y_pred_GE = SpectralClustering(gamma=0.5, n_clusters=cluster_num).fit_predict(fused_net_GE)
        y_pred_ME = SpectralClustering(gamma=0.5, n_clusters=cluster_num).fit_predict(fused_net_ME)
        y_pred_MI = SpectralClustering(gamma=0.5, n_clusters=cluster_num).fit_predict(fused_net_MI)

        groups_y_GE.append(y_pred)
        groups_y_ME.append(y_pred)
        groups_y_MI.append(y_pred)

        f_GE = groups_y_GE[cluster_num - 2]
        f_ME = groups_y_ME[cluster_num - 2]
        f_MI = groups_y_MI[cluster_num - 2]

        temp_f_GE = []
        for f_num in f_GE:
            temp_f_GE.append(f_num + 1)
        f_GE = temp_f_GE
        f_GE = pd.DataFrame(f_GE)
        f_GE.to_csv("result/DataTCGA/{}/f_{}_[{}]_GE.csv".format(dataset, dataset, cluster_num), header=True, index=True)
        temp_f_ME = []
        for f_num in f_ME:
            temp_f_ME.append(f_num + 1)
        f_ME = temp_f_ME
        f_ME = pd.DataFrame(f_ME)
        f_ME.to_csv("result/DataTCGA/{}/f_{}_[{}]_ME.csv".format(dataset, dataset, cluster_num), header=True, index=True)
        temp_f_MI = []
        for f_num in f_MI:
            temp_f_MI.append(f_num + 1)
        f_MI = temp_f_MI
        f_MI = pd.DataFrame(f_MI)
        f_MI.to_csv("result/DataTCGA/{}/f_{}_[{}]_MI.csv".format(dataset, dataset, cluster_num), header=True, index=True)

        current_f_path = "result/DataTCGA/{}/f_{}_{}.csv".format(dataset, dataset, cluster_num)
        robjects.r('''
            current_f_path="{}"
            current_rf=read.csv(current_f_path,header = TRUE)
        '''.format(current_f_path))
        robjects.r('''
            for (nidx in 1:length(result[["groups2"]])) {
                result[["groups2"]][nidx]= current_rf$X0[nidx]
            }
        ''')

        robjects.r('''
            groups2=result$groups2
            coxFit <- coxph(Surv(time = Survival, event = Death) ~ as.factor(groups2), data = survival[names(groups2),], ties="exact")
            mfit <- survfit(Surv(Survival, Death == 1) ~ as.factor(groups2), data = survival[names(groups2),])
            plot(mfit, col=levels(factor(groups2)), main = paste("Survival curves for ", rdataset, ", (PNF) cluster", "{}", sep=""), xlab = "Days", ylab="Survival", lwd=2)
            legend("top", legend = paste("Cox p-value:", round(summary(coxFit)$sctest[3],digits = 16), sep=""))
            legend("topright", fill=levels(factor(groups2)), legend=paste("Group ",levels(factor(groups2)), ": ", table(groups2)[levels(factor(groups2))], sep=""))
        '''.format(cluster_num))
        currentP = robjects.r('summary(coxFit)$sctest[3]')
        recordP.append("{}".format(currentP[0]))
        if currentP <= temp_Pvalues["minPvalues"]:
            temp_Pvalues["minPvalues"] = currentP
            temp_Pvalues["cluster_num"] = cluster_num
            temp_Pvalues["f"] = f
        pass

    robjects.r('''
        t2=Sys.time()
        print(t2-t1)
    ''')
    print("recordP=", recordP)
    # replace my result
    best_f_path = "result/DataTCGA/{}/f_{}_{}.csv".format(dataset, dataset, temp_Pvalues["cluster_num"])# 聚类数依据最小P值来选择的
    robjects.r('''
        best_f_path="{}"
        rf=read.csv(best_f_path,header = TRUE)
    '''.format(best_f_path))
    robjects.r('''
        for (nidx in 1:length(result[["groups2"]])) {
            result[["groups2"]][nidx]= rf$X0[nidx]
        }
    ''')

    k_GE = robjects.r('k_GE=min(which(result[["dataTypeResult"]][[1]]$Discrepancy$AUC == max(result[["dataTypeResult"]][[1]]$Discrepancy$AUC[2:Kmax])))')
    best_f_path_GE = "result/DataTCGA/{}/f_{}_{}_GE.csv".format(dataset, dataset, k_GE)
    robjects.r('''
        best_f_path_GE="{}"
    '''.format(best_f_path_GE))
    robjects.r('rf_GE=read.csv(best_f_path_GE,header = TRUE)')
    robjects.r('''
        result[["dataTypeResult"]][[1]][["k"]]=k_GE
        for (nidx in 1:length(result[["dataTypeResult"]][[1]][["groups"]])) {
            result[["dataTypeResult"]][[1]][["groups"]][nidx]= rf_GE$X0[nidx]
        }
    ''')

    k_ME = robjects.r('k_ME=min(which(result[["dataTypeResult"]][[2]]$Discrepancy$AUC == max(result[["dataTypeResult"]][[2]]$Discrepancy$AUC[2:Kmax])))')
    best_f_path_ME = "result/DataTCGA/{}/f_{}_{}_ME.csv".format(dataset, dataset, k_ME)
    robjects.r('''
        best_f_path_ME="{}"
        rf_ME=read.csv(best_f_path_ME,header = TRUE)
    '''.format(best_f_path_ME))
    robjects.r('''
        result[["dataTypeResult"]][[2]][["k"]]=k_ME
        for (nidx in 1:length(result[["dataTypeResult"]][[2]][["groups"]])) {
            result[["dataTypeResult"]][[2]][["groups"]][nidx]= rf_ME$X0[nidx]
        }
    ''')

    k_MI = robjects.r('k_MI=min(which(result[["dataTypeResult"]][[3]]$Discrepancy$AUC == max(result[["dataTypeResult"]][[3]]$Discrepancy$AUC[2:Kmax])))')
    best_f_path_MI = "result/DataTCGA/{}/f_{}_{}_MI.csv".format(dataset, dataset, k_MI)
    robjects.r('''
        best_f_path_MI="{}"
        rf_MI=read.csv(best_f_path_MI,header = TRUE)
    '''.format(best_f_path_MI))
    robjects.r('''
        result[["dataTypeResult"]][[3]][["k"]]=k_MI
        for (nidx in 1:length(result[["dataTypeResult"]][[3]][["groups"]])) {
            result[["dataTypeResult"]][[3]][["groups"]][nidx]= rf_MI$X0[nidx]
        }
    ''')
    robjects.r('save(rdataset, Kmax, dataList, survival, clinical, result, t1, t2, file=resultFile)')
# fen ge fu .......................................................................



    robjects.r('''
        
        groups = result$groups
        groups2=result$groups2
        plot(result$dataTypeResult[[1]]$Discrepancy$AUC, ylab= "Area under the curve", xlab="Cluster number", main=paste("AUC of gene expression for ", "{}", " data", sep=""))
        lines(1:Kmax, result$dataTypeResult[[1]]$Discrepancy$AUC)
        points(result$dataTypeResult[[1]]$k, result$dataTypeResult[[1]]$Discrepancy$AUC[result$dataTypeResult[[1]]$k],col="red")

        plot(result$dataTypeResult[[2]]$Discrepancy$AUC, ylab= "Area under the curve", xlab="Cluster number", main=paste("AUC of methylation for ", "{}", " data", sep=""))
        lines(1:Kmax, result$dataTypeResult[[2]]$Discrepancy$AUC)
        points(result$dataTypeResult[[2]]$k, result$dataTypeResult[[2]]$Discrepancy$AUC[result$dataTypeResult[[2]]$k],col="red")

        plot(result$dataTypeResult[[3]]$Discrepancy$AUC, ylab= "Area under the curve", xlab="Cluster number", main=paste("AUC of microRNA for ", "{}", " data", sep=""))
        lines(1:Kmax, result$dataTypeResult[[3]]$Discrepancy$AUC)
        points(result$dataTypeResult[[3]]$k, result$dataTypeResult[[3]]$Discrepancy$AUC[result$dataTypeResult[[3]]$k],col="red")

        coxFit <- coxph(Surv(time = Survival, event = Death) ~ as.factor(result$dataTypeResult[[1]]$groups), data = survival, ties="exact")
        mfit <- survfit(Surv(Survival, Death == 1) ~ as.factor(result$dataTypeResult[[1]]$groups), data = survival)
        plot(mfit, col=unique(result$dataTypeResult[[1]]$groups), main = paste("Survival curves for gene expression of ","{}", " (PNF)", sep=""), xlab = "Days", ylab="Survival", lwd=2)
        legend("topright", legend = paste("Cox p-value:", round(summary(coxFit)$sctest[3],digits = 5), sep=""))

        coxFit <- coxph(Surv(time = Survival, event = Death) ~ as.factor(result$dataTypeResult[[2]]$groups), data = survival, ties="exact")
        mfit <- survfit(Surv(Survival, Death == 1) ~ as.factor(result$dataTypeResult[[2]]$groups), data = survival)
        plot(mfit, col=unique(result$dataTypeResult[[2]]$groups), main = paste("Survival curves for methylation of ", "{}", " (PNF)", sep=""), xlab = "Days", ylab="Survival", lwd=2)
        legend("topright", legend = paste("Cox p-value:", round(summary(coxFit)$sctest[3],digits = 5), sep=""))

        coxFit <- coxph(Surv(time = Survival, event = Death) ~ as.factor(result$dataTypeResult[[3]]$groups), data = survival, ties="exact")
        mfit <- survfit(Surv(Survival, Death == 1) ~ as.factor(result$dataTypeResult[[3]]$groups), data = survival)
        plot(mfit, col=unique(result$dataTypeResult[[3]]$groups), main = paste("Survival curves for microRNA of ", "{}", " (PNF)", sep=""), xlab = "Days", ylab="Survival", lwd=2)
        legend("topright", legend = paste("Cox p-value:", round(summary(coxFit)$sctest[3],digits = 5), sep=""))

        ageCol <- abs(as.numeric(clinical$"birth_days_to"))/365
        names(ageCol) <- rownames(clinical)
    '''.format(dataset, dataset, dataset, dataset, dataset, dataset, dataset, dataset, dataset, dataset))

    robjects.r('''
        age <- list()
        for (j in levels(factor(groups2))) {
            age[[j]] <- ageCol[names(groups2[groups2==j])]
        }
    ''')
    robjects.r('''
        boxplot(age, main=paste("Age distribution, ", "{}", sep=""), xlab="Groups", ylab="Age")
    '''.format(dataset))

    robjects.r('''
          coxFit <- coxph(Surv(time = Survival, event = Death) ~ as.factor(result$groups2), data = survival, ties="exact")
          mfit <- survfit(Surv(Survival, Death == 1) ~ as.factor(groups2), data = survival)
          plot(mfit, col=unique(levels(factor(result$groups2))), main = paste("Survival curves for ", rdataset, ", (PNF)", sep=""), xlab = "Days", ylab="Survival", lwd=2)
          legend("top", legend = paste("Cox p-value:", round(summary(coxFit)$sctest[3],digits = 16), sep=""))
          legend("topright", fill=unique(levels(factor(groups2))), legend=paste("Group ",levels(factor(groups2)), ": ", table(groups2)[levels(factor(groups2))], sep=""))

    ''')
    print('temp_Pvalues["cluster_num"]=', temp_Pvalues["cluster_num"])
    robjects.r('dev.off()')

    Rp_value = robjects.r('summary(coxFit)$sctest[3]')
    p_values[dataset] = Rp_value[0]
    SC_values.append(Scores)
    AH_values.append(Scores1)
    temp_Scores = []
    print("Scores=", Scores)
    print("Scores1=", Scores1)
    print("min(Scores)=", max(Scores))
    print("index max(Scores)=", Scores.index(max(Scores))+2)
    print("max(Scores1)=", max(Scores1))
    print("index max(Scores1)=", Scores1.index(max(Scores1))+2)
    for j in range(0, 9):
        temp_Scores.append(math.log(10, np.abs(Scores[j])))
    i = range(2,11)
    plt.xlabel('k')
    plt.ylabel('value')
    plt.plot(i, temp_Scores[:9], 'g.-', i, Scores1[:9], 'b.-')
    plt.savefig('result/DataTCGA/{}/{}_SCAH.png'.format(dataset, dataset))
    # plotRes(data, y_pred, cluster_num)
    print("P values = ....{}".format(p_values))
    # break

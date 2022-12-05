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


def snf_plus_altered_sim_GEME(*aff,pd_DataGE, pd_DataME, K=20, t=20, alpha=1.0):
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


def snf_plus_altered_sim_MEMI(*aff, pd_DataME, pd_DataMI, K=20, t=20, alpha=1.0):
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
                    if pd_DataME.iloc[i, j] == 0:
                        temp_DF.iloc[i, j] = temp_DF.iloc[i, j] * 2
                    elif pd_DataME.iloc[i, j] == 1:
                        temp_DF.iloc[i, j] = temp_DF.iloc[i, j]
                    else:
                        temp_DF.iloc[i, j] = temp_DF.iloc[i, j] / pd_DataME.iloc[i, j]
            temp_aff[n] = temp_DF.to_numpy()
            Wk[n] = temp_aff[n]
        if n == 1:
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


def snf_plus_altered_sim_GEMI(*aff,pd_DataGE, pd_DataMI, K=20, t=20, alpha=1.0):
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


def PNF(data, Kmax=10, noisePercent="med", iter=200, kmIter=20, noise=None):
    robjects.r('''
        data="{}"
        Kmax={}
        noisePercent="{}"
        iter={}
        kmIter={}
    '''.format(data, Kmax, noisePercent, iter, kmIter, noise))
    robjects.r('''
        noise=NULL
        data <- matrix(rnorm(200*10000, 0, 1), nrow=200, ncol=10000)
        pca = prcomp(data) 
        message("Building original connectivity matrices")
        flush.console()
        origPartition <- getOriginalSimilarity(data = pca$x, clusRange = 2:Kmax)
        origS <- origPartition$origS
        if (is.null(noise)) 
            noise = getNoise(data, noisePercent)
        message("Noise set to ", noise)
        message("Building perturbed connectivity matrices")
        flush.console()
        pertS <- getPerturbedSimilarity(data = pca$x, clusRange = 2:Kmax, iter = iter, noiseSd = noise, kmIter = kmIter)
        Discrepancy <- getPerturbedDiscrepancy(origS = origS, pertS = pertS, 
        clusRange = 2:Kmax)
        ret <- NULL
        ret$k = min(which(Discrepancy$AUC == max(Discrepancy$AUC[2:Kmax])))
        ret$Kmax = 10
        ret$groups <- origPartition$groupings[[ret$k]]
        ret$origS <- origS
        ret$pertS <- pertS
        ret$Discrepancy <- Discrepancy
        message("Done. \n")
        flush.console()
        
        result=ret
    ''')

    data = pd.DataFrame(data)
    groups_y = []
    print("Kmax", Kmax)
    for cluster_num in range(2, int(Kmax)+1):
        f_path = "result/Simulation/f_timecomplexity_{}.csv".format(cluster_num)
        pertS = pd.DataFrame(robjects.r('pertS[[{}]]'.format(cluster_num)))
        m = 'sqeuclidean'
        K = 20
        mu = 0.5

        affinity_nets = snf.make_affinity(
            [data.iloc[:, 1:].values.astype(np.float), data.iloc[:, 1:].values.astype(np.float),
             data.iloc[:, 1:].values.astype(np.float)],
            metric=m, K=K, mu=mu)
        fused_net = snf_plus_altered_sim(affinity_nets, pd_DataGE=pertS, pd_DataME=pertS, pd_DataMI=pertS,
                                         K=K)
        fused_df = pd.DataFrame(fused_net)
        print("fused_df", "\n", fused_df)
        y_pred = SpectralClustering(gamma=0.5, n_clusters=cluster_num).fit_predict(fused_df)
        groups_y.append(y_pred)
        f = groups_y[cluster_num - 2]
        temp_f = []
        for f_num in f:
            temp_f.append(f_num + 1)
        f = temp_f
        f = pd.DataFrame(f)
        f.to_csv(f_path, header=True, index=True)
        robjects.r('''
            f_path="{}"
            # print(c('f_path', f_path))
            current_rf=read.csv(f_path,header = TRUE)
        '''.format(f_path))

        minK = float(
            robjects.r('min(which(result$Discrepancy$AUC == max(result$Discrepancy$AUC[2:result[["Kmax"]]])))'))
        if minK == cluster_num:
            robjects.r('''
                    result[["k"]]=min(which(result$Discrepancy$AUC == max(result$Discrepancy$AUC[2:result[["Kmax"]]])))
                    for (nidx in 1:length(result[["groups"]])) {
                        result[["groups"]][nidx]= current_rf$X0[nidx]
                    }
                ''')
    robjects.r('''
        result
    ''')
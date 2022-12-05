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
robjects.r('rm(list=ls())')
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
datasets = robjects.r('datasets=c("KIRC", "LUSC", "BRCA", "LAML", "GBM", "COAD")')

# start for cycle
for dataset in datasets:
    dataset = "BRCA"
    robjects.r('dataset="{}"'.format(dataset))
    robjects.r('''
        file=paste(resultPath,dataset, "/", "PINS_", dataset, ".RData" ,sep="")
        print(file)
        load(file)
        groups2=data.frame(result$groups2)
        rowname=row.names(groups2)
    ''')
    groups2 = pd.DataFrame(robjects.r('result$groups2'))
    rowname = robjects.r('rowname')
    ajcc_pathologic_tumor_stage = robjects.r('clinical$ajcc_pathologic_tumor_stage')
    gender = robjects.r('clinical$gender')
    margin_status = robjects.r('clinical$margin_status')
    er_status_by_ihc = robjects.r('clinical$er_status_by_ihc')
    groups2.insert(loc=1, column='ajcc_pathologic_tumor_stage', value=ajcc_pathologic_tumor_stage)
    groups2.insert(loc=2, column='gender', value=gender)
    groups2.insert(loc=3, column='margin_status', value=margin_status)
    groups2.insert(loc=4, column='er_status_by_ihc', value=er_status_by_ihc)
    groups2.index = rowname
    # groups2.insert(loc=0, column='patientsID', value=rowname)
    groups2.rename(columns={0: 'groups'}, inplace=True)
    groups2.index.name = "patientID"
    # groups2.to_csv('groups2.csv')
    groups2.to_excel('groups2.xls')


    break

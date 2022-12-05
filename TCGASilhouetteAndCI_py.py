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
rpackages.importr('gplots')
rpackages.importr('cluster')
rpackages.importr('iClusterPlus')

# import R scripts
robjects.r('source("R_scripts/BasicFunctions.R")')

# initialize and set some parameters
# dataPath = robjects.r('dataPath="Projects/SubTyping/DataTCGA/"')
# robjects.r('resultPath="result/DataTCGA/"')
Kmax = robjects.r('Kmax=10')  # it should be 10
iter = robjects.r('iter=2')  # it should be 200
noisePercent = robjects.r('noisePercent="med"')
kmIter = robjects.r('kmIter=2')  # it should be 200
datasets = robjects.r('datasets=c("KIRC", "LUSC", "BRCA", "LAML", "GBM", "COAD")')
robjects.r('''
    SNFPath="./Projects/SubTyping/PackageAndTesting/SNFResult/"
    PINSPath="./Projects/SubTyping/PackageAndTesting/PINSResult/"
    CCPath="./Projects/SubTyping/PackageAndTesting/CCResult/"
    # iClusterPlusPath="./Projects/SubTyping/PackageAndTesting/iClusterPlusResult/"
    pdfPath="result//Plots/Figures/"
    
    datasets=c("KIRC", "GBM", "LAML", "LUSC", "BRCA", "COAD")
    datatypes=c("GE", "ME", "MI", "Integration")
    rows=as.vector(t(outer(datasets, datatypes, paste, sep="_")))
''')






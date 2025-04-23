#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 14:42:19 2025

@author: manupc
"""

import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import tikzplotlib
from scipy.stats import shapiro as normaltest
from scipy.stats import ttest_1samp as ttest


mpl.use("pgf")

def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)


file= 'FinalResults.pkl'
with open(file, 'rb') as f:
    results = pickle.load(f)
    
print('Results (success in experiments):')
for current_dataset in results:
    print('Dataset {}: {}/{}'.format(current_dataset, results[current_dataset]['success'], results[current_dataset]['performed']))

shots= {1 :50000, 2 :50000, 3 :50000, 4 :50000, 
        5 :200000, 6 :200000, 7 :200000, 8: 200000,
        9 :2000000, 10: 2000000, 11: 2000000, 12:2000000,
        13:10000000, 14:10000000, 15:10000000, 16:10000000}
size= len(shots)
for i in range(1, size+1):
    shots[i+size]= shots[i]
    
encodings= ['QRAM', 'AMP']
ns= [4, 8, 16, 32]
ms= [2, 3, 4, 5]

# Success rate table
index= 0
tableContent=''
totalSuccessQRAM= 0
totalSuccessAMP= 0
for n in ns:
    for m in ms:
        index+= 1
        nameQRAM= '$D({},{},e)$'.format(n, m)
        nameAMP= '$D({},2^{},e)$'.format(n, m)
        successQRAM= 'na' if index not in results else results[index]['success']
        successAMP= 'na' if index+size not in results else results[index+size]['success']
        meas= shots[index]
        tableContent+= '{} & {} & {} & {} & {}\\\\\n\\hline\n'.format(nameQRAM, successQRAM, nameAMP, successAMP, meas)
        if successQRAM != 'na' and successAMP != 'na':
            totalSuccessQRAM+= successQRAM
            totalSuccessAMP+= successAMP
    
totalSuccessAMP/=(len(ns)*len(ms))
totalSuccessQRAM/=(len(ns)*len(ms))
tableContent+= '{} & {} & {} & {} & \\\\\n\\hline\n'.format('\\textit{\\textbf{Avg.:}}', totalSuccessQRAM, '\\textit{\\textbf{Avg.:}}', totalSuccessAMP)
    
print('\n\n')
print('Content of table of sucess:\n')
print(tableContent)
print('\n\n')    

# Calculate residuals
residuals= {encodings[0] : {}, encodings[1] : {}}
index= 0
meanDistances={encodings[0] : {}, encodings[1] : {}}
for encoding in encodings:
    for n in ns:
        for m in ms:
            name= '$D({},{},{})$'.format(n, str(m) if encoding=='QRAM' else '2^{}'.format(m), encoding)
            index+= 1
            if index in results:
                trueD= np.array(results[index]['true distance'])
                approxD= np.array(results[index]['calculated distance'])
                meanD= np.array(results[index]['mean true distance'])
                diff= trueD-approxD
                residuals[encoding][name]= diff
                meanDistances[encoding][name]= meanD


# Content of table of errors
decimals= 3
tableContent=''
index= 0
for n in ns:
    for m in ms:
        names= []
        meanErrors= []
        meanD= []
        sdErrors= []
        index+= 1
        for encoding in encodings:
            accQRAM= results[index]['success']
            accAMP= results[index+16]['success']
            name= '$D({},{},{})$'.format(n, str(m) if encoding=='QRAM' else '2^{}'.format(m), encoding)
            names.append(name)
            allErrors= [np.nan] if name not in residuals[encoding] else np.abs(residuals[encoding][name])
            meanErrors.append( 'na' if np.any(np.isnan(allErrors)) else np.round(np.mean(allErrors), decimals) )
            sdErrors.append('na' if np.any(np.isnan(allErrors)) else np.round(np.std(allErrors), decimals) )
            meanD.append('na' if np.any(np.isnan(allErrors)) else np.round(np.mean(meanDistances[encoding][name]), decimals))
        for i in range(2):
            if i==1:
                tableContent+=' & '
            names[i]= names[i].replace(encodings[0], 'e').replace(encodings[1], 'e')
            tableContent+= ' {} & {}$\pm${} '.format(names[i], meanErrors[i], sdErrors[i])
        tableContent+=' \\\\\n\\hline\n'


print('\n\n')
print('Content of table of errors:\n')
print(tableContent)
print('\n\n')    


# Normality tests
print('\n\nNormality tests:')
for encoding in residuals:
    for name in residuals[encoding]:
        _, pval= normaltest(residuals[encoding][name])
        passed= pval > 0.05
            
        print('Encoding: {}. name: {}. Normal: {}'.format(encoding, name, passed))



# T-test
print('\n\nT.Test tests:')
for encoding in residuals:
    for name in residuals[encoding]:
        _, pval= ttest(residuals[encoding][name], popmean=0)
        passed= pval > 0.05
            
        print('Encoding: {}. name: {}. Normal: {}'.format(encoding, name, passed))


# Boxplots of residuals
for encoding in encodings:
    f= plt.figure()
    g= sns.boxplot(residuals[encoding])
    g.set_xticklabels(g.get_xticklabels(), rotation=90)
    plt.tight_layout()
    plt.show()
    
    f.savefig('boxplot{}.pgf'.format(encoding))
    tikzplotlib.save('boxplot{}.tikz'.format(encoding))
    
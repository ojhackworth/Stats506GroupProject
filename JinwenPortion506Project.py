# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 12:29:43 2018
Project Draft of Stat 506
@author: Jinwen Cao
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pysal
from scipy.stats import chi2, norm


## load the dataset ###
with open('communities.data','rb') as f:
        file_content = list(item.decode() for item in f)
        
raw_data = []
for item in file_content:
    item_rp = item.rstrip('\r\n')
    item_sp = item_rp.split(',')
    raw_data.append(item_sp)

col_names = range(128)
f = pd.read_csv('communities.data', header = None)
raw_data = f.replace('?', np.nan)

## select the variables needed and drop missing values
# population: 6  pcturban:17 percapinc: 26  pctpopunderpov: 34 
# population: population for community: (numeric - decimal)
# pctUrban: percentage of people living in areas classified as urban (numeric - decimal)
# perCapInc: per capita income (numeric - decimal)
# PctPopUnderPov: percentage of people under the poverty level (numeric - decimal)

# Min Max Mean SD Correl Median Mode Missing
# population 0 1 0.06 0.13 0.37 0.02 0.01 0
# pctUrban 0 1 0.70 0.44 0.08 1 1 0
# perCapInc 0 1 0.35 0.19 -0.35 0.3 0.23 0
# PctPopUnderPov 0 1 0.30 0.23 0.52 0.25 0.08 0
# ViolentCrimesPerPop 0 1 0.24 0.23 1.00 0.15 0.03 0
mydata = raw_data.iloc[:,[6-1, 17-1, 26-1, 34-1, 128-1]]
mydata.columns = ['population', 'pctUrban', 'perCapInc', 'PctPopUnderPov', 'ViolentCrimesPerPop']
data = mydata.dropna()


## define a function for HL test
def hl_test(data, g):
    '''
    Hosmer-Lemeshow test to judge the goodness of fit for binary data

    Input: dataframe
    
    Output: float
    '''
    data_st = data.sort_values('prob')
    data_st['dcl'] = pd.qcut(data_st['prob'], g)
    
    ys = data_st['ViolentCrimesPerPop'].groupby(data_st.dcl).sum()
    yt = data_st['ViolentCrimesPerPop'].groupby(data_st.dcl).count()
    yn = yt - ys
    
    yps = data_st['prob'].groupby(data_st.dcl).sum()
    ypt = data_st['prob'].groupby(data_st.dcl).count()
    ypn = ypt - yps
    
    hltest = ( ((ys - yps)**2 / yps) + ((yn - ypn)**2 / ypn) ).sum()
    pval = 1 - chi2.cdf(hltest, g-2)
    
    df = g-2
    
    print('\n HL-chi2({}): {}, p-value: {}\n'.format(df, hltest, pval))
#    return df, hltest, pval


def logit_p(skm, x):
    '''
     Print the p-value for sklearn logit model

    Input: model, nparray(df of independent variables)
    
    Output: none
    '''
    pb = skm.predict_proba(x)
    n = len(pb)
    m = len(skm.coef_[0]) + 1
    coefs = np.concatenate([skm.intercept_, skm.coef_[0]])
    x_full = np.matrix(np.insert(np.array(x), 0, 1, axis = 1))
    result = np.zeros((m, m))
    for i in range(n):
        result = result + np.dot(np.transpose(x_full[i, :]), x_full[i, :]) * pb[i,1] * pb[i, 0]
    vcov = np.linalg.inv(np.matrix(result))
    se = np.sqrt(np.diag(vcov))
    t =  coefs/se  
    pval = (1 - norm.cdf(abs(t))) * 2
    print(pd.DataFrame(pval, index=['intercept', 'population', 'pctUrban', 
                                    'perCapInc', 'PctPopUnderPov'], 
                       columns=['p-value']))
#    return pval



## create a cutoff for 'ViolentCrimesPerPop'
## recode >0.5 as 1, and <= 0.5 as 0
data['ViolentCrimesPerPop'][ data['ViolentCrimesPerPop'] > 0.5 ] = 1
data['ViolentCrimesPerPop'][ data['ViolentCrimesPerPop'] <= 0.5 ] = 0

X = data.iloc[:,:4]
Y = data.iloc[:, 4]


## fit probit model and apply HL test
pbt = pysal.spreg.Probit(Y.values[:, None], X.values, 
                         name_x=['population', 'pctUrban', 
                                 'perCapInc', 'PctPopUnderPov'], 
                         name_y='ViolentCrimesPerPop', 
                         name_ds='Coummunities and Crime')
data['prob'] = pbt.predy
print(pbt.summary)
p_probit = hl_test(data, 5)


## fit logit model and apply HL test
lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(X, Y)
data['prob'] = lr.predict_proba(X)[:, 1]
print(logit_p(lr, X.values))
p_logit = hl_test(data, 5)

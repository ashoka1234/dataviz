# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 19:02:06 2016

@author: Ashoka
"""
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy import stats


def discrete_scatter_plot(data, x, y, xindexes=None, yindexes=None, s=10):
    if xindexes is None:
        xindexes = np.unique(data[x].dropna())
    if yindexes is None:
        yindexes = np.unique(data[y].dropna())
    sarray = np.zeros([xindexes.size, yindexes.size])
    count_df = pd.DataFrame(sarray, index=xindexes, columns=yindexes)
    for i in range(data.shape[0]):
        ix = xindexes.dtype.type(data.iloc[i][x])
        iy = yindexes.dtype.type(data.iloc[i][y])
        if pd.notnull(ix) and pd.notnull(iy):
            count_df[iy][ix] += 1
    scolumn = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        ix = xindexes.dtype.type(data.iloc[i][x])
        iy = yindexes.dtype.type(data.iloc[i][y])
        if pd.notnull(ix) and pd.notnull(iy):
            scolumn[i] = count_df[iy][ix]
    for i in range(data.shape[0]):
        scolumn[i] = math.sqrt(scolumn[i])
    my_x, my_y, my_s = x, y, s
    data.plot.scatter(x=my_x, y=my_y, s=my_s*scolumn)


def small_discrete_range(data, column, size=15):
    if np.unique(data[column].dropna()).size > size:
        return False
    else:
        return True


def analyse_pair_plot_type(data, predictor, target):
    """ plot each column data to target scatter plots """
    feature_names = data.columns
    if target in feature_names and predictor in feature_names:
        if small_discrete_range(data, target):
            if small_discrete_range(data, predictor):
                return 'discrete_scatter_plot'
        return 'scatter_plot'
    else:
        print('target or predictor not in the data')
        return None


def pairwise_dataviz(data, target):
    feature_names = data.columns.drop(target)
    for predictor in feature_names:
        ptype = analyse_pair_plot_type(data, predictor, target)
        if ptype == 'discrete_scatter_plot':
            discrete_scatter_plot(df, predictor, 'Income')
        elif ptype == 'scatter_plot':
            data.plot.scatter(x=predictor, y=target)

# plotting commands
df = pd.read_csv('marketing-data.csv')
pairwise_dataviz(df, 'Income')
figs = plt.get_fignums()
for i in figs:
    plt.figure(i)
    gfname = 'graphs' + str(i) + '.png'
    plt.savefig(gfname)

# p-value calculations
a = df[df['Sex'] == 1]['Income']
b = df[df['Sex'] == 2]['Income']
stats.ttest_ind(a, b)
x = a[a != 1]
y = b[b != 1]
stats.ttest_ind(x, y)

# histograms
a.plot.hist(bins=9)
b.plot.hist(bins=9)

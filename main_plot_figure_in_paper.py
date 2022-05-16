#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 20:54:36 2020

@author: yifeng
"""

import os
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
import seaborn as sns
import numpy as np

from utils.plots import plot_counts_multi,plot_props_multi


def plot_dgm_loss_from_diff_expriments(run_dir1, run_dir2, run_dir3, group='epoch', postfix='PCBA'):
    df1 = pd.read_csv(os.path.join(run_dir1, 'results/performance/loss_details.csv')) # 1
    df2 = pd.read_csv(os.path.join(run_dir2, 'results/performance/loss_details.csv')) # 0.1
    df3 = pd.read_csv(os.path.join(run_dir3, 'results/performance/loss_details.csv')) # 0.01
    
    if group=='epoch':
        df1 = df1.groupby('epoch').mean()
        df2 = df2.groupby('epoch').mean()
        df3 = df3.groupby('epoch').mean()
    
    
    #fig, axs = plt.subplots(5, 1, sharex=True)
    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(8,4)
    axs[0,0].plot(df1['CE_loss'], color='red', alpha=0.5, label=r'$\beta=1$')
    axs[0,0].plot(df2['CE_loss'], color='blue', alpha=0.5, label=r'$\beta=0.1$')
    axs[0,0].plot(df3['CE_loss'], color='olive', alpha=0.5, label=r'$\beta=0.01$')
    axs[0,0].set_ylabel('CE Loss')
    axs[0,0].set_xlim(0,49)
    axs[0,0].legend(loc='upper right')
    
    axs[0,1].plot(df1['KL_loss'], color='red', alpha=0.5, label=r'$\beta=1$')
    axs[0,1].plot(df2['KL_loss'], color='blue', alpha=0.5, label=r'$\beta=0.1$')
    axs[0,1].plot(df3['KL_loss'], color='olive', alpha=0.5, label=r'$\beta=0.01$')
    axs[0,1].set_ylabel('KL Loss')
    axs[0,1].set_xlim(0,49)
    #axs[0,1].legend(loc='upper right')
    
    axs[1,0].plot(df1['MSE_loss'], color='red', alpha=0.5, label=r'$\beta=1$')
    axs[1,0].plot(df2['MSE_loss'], color='blue', alpha=0.5, label=r'$\beta=0.1$')
    axs[1,0].plot(df3['MSE_loss'], color='olive', alpha=0.5, label=r'$\beta=0.01$')
    axs[1,0].set_ylabel('MSE Loss')
    axs[1,0].set_x_label('Epoch')
    axs[1,0].set_xlim(0,49)
    #axs[1,0].legend(loc='upper right')
    
    axs[1,1].plot(df1['loss'], color='red', alpha=0.5, label=r'$\beta=1$')
    axs[1,1].plot(df2['loss'], color='blue', alpha=0.5, label=r'$\beta=0.1$')
    axs[1,1].plot(df3['loss'], color='olive', alpha=0.5, label=r'$\beta=0.01$')
    axs[1,1].set_ylabel('Loss') #r'CE_Loss + $\beta$ MSE_Loss + $\alpha$ KL_loss'
    axs[1,1].set_x_label('Epoch')
    axs[1,1].set_xlim(0,49)
    #axs[1,1].legend(loc='upper right')
    filename='./fig_dgm_loss_details_'+group+'_'+postfix+'.pdf'
    fig.savefig(filename,bbox_inches='tight')
    

def plot_del_loss_from_diff_expriments(run_dir1, run_dir2, run_dir3, run_dir4, group='epoch', postfix='PCBA'):
    df1 = pd.read_csv(os.path.join(run_dir1, 'results/performance/loss_details.csv')) # 0.1
    df2 = pd.read_csv(os.path.join(run_dir2, 'results/performance/loss_details.csv')) # 0.1 -> 0.4
    df3 = pd.read_csv(os.path.join(run_dir3, 'results/performance/loss_details.csv')) # 0.01
    df4 = pd.read_csv(os.path.join(run_dir4, 'results/performance/loss_details.csv')) # 0.01 -> 0.4
    
    if group=='epoch':
        df1 = df1.groupby('epoch').mean()
        df2 = df2.groupby('epoch').mean()
        df3 = df3.groupby('epoch').mean()
        df4 = df4.groupby('epoch').mean()
    
    #fig, axs = plt.subplots(5, 1, sharex=True)
    fig, axs = plt.subplots(3, 2)
    fig.set_size_inches(10,8)
    axs[0,0].plot(df1['CE_loss'], color='red', alpha=0.5, label=r'$\beta=0.1, \alpha=1$')
    axs[0,0].plot(df2['CE_loss'], color='orange', alpha=0.5, label=r'$\beta=0.1 \rightarrow 0.4, \alpha=1 \rightarrow 4$')
    axs[0,0].plot(df3['CE_loss'], color='blue', alpha=0.5, label=r'$\beta=0.01, \alpha=1$')
    axs[0,0].plot(df4['CE_loss'], color='steelblue', alpha=0.5, label=r'$\beta=0.01 \rightarrow 0.4, \alpha=1 \rightarrow 4$')
    axs[0,0].set_ylabel('CE Loss')
    #axs[0,0].set_xlim(0,49)
    axs[0,0].legend(loc='upper right')
    
    axs[0,1].plot(df1['KL_loss'], color='red', alpha=0.5, label=r'$\beta=0.1, \alpha=1$')
    axs[0,1].plot(df2['KL_loss'], color='orange', alpha=0.5, label=r'$\beta=0.1 \rightarrow 0.4, \alpha=1 \rightarrow 4$')
    axs[0,1].plot(df3['KL_loss'], color='blue', alpha=0.5, label=r'$\beta=0.01, \alpha=1$')
    axs[0,1].plot(df4['KL_loss'], color='steelblue', alpha=0.5, label=r'$\beta=0.01 \rightarrow 0.4, \alpha=1 \rightarrow 4$')
    axs[0,1].set_ylabel('KL Loss')
    #axs[0,1].set_xlim(0,49)
    #axs[0,1].legend(loc='upper right')
    
    axs[1,0].plot(df1['MSE_loss'], color='red', alpha=0.5, label=r'$\beta=0.1, \alpha=1$')
    axs[1,0].plot(df2['MSE_loss'], color='orange', alpha=0.5, label=r'$\beta=0.1 \rightarrow 0.4, \alpha=1 \rightarrow 4$')
    axs[1,0].plot(df3['MSE_loss'], color='blue', alpha=0.5, label=r'$\beta=0.01, \alpha=1$')
    axs[1,0].plot(df4['MSE_loss'], color='steelblue', alpha=0.5, label=r'$\beta=0.01 \rightarrow 0.4, \alpha=1 \rightarrow 4$')
    axs[1,0].set_ylabel('MSE Loss')
    #axs[1,0].set_xlim(0,49)
    #axs[1,0].legend(loc='upper right')
    
    axs[1,1].plot(df1['loss'], color='red', alpha=0.5, label=r'$\beta=0.1, \alpha=1$')
    axs[1,1].plot(df2['loss'], color='orange', alpha=0.5, label=r'$\beta=0.1 \rightarrow 0.4, \alpha=1 \rightarrow 4$')
    axs[1,1].plot(df3['loss'], color='blue', alpha=0.5, label=r'$\beta=0.01, \alpha=1$')
    axs[1,1].plot(df4['loss'], color='steelblue', alpha=0.5, label=r'$\beta=0.01 \rightarrow 0.4, \alpha=1 \rightarrow 4$')
    axs[1,1].set_ylabel('Loss') #r'CE_Loss + $\beta$ MSE_Loss + $\alpha$ KL_loss'
    #axs[1,1].set_xlim(0,49)
    #axs[1,1].legend(loc='upper right')
    
    axs[2,0].plot(df1['beta'], color='red', alpha=0.5, label=r'$\beta=0.1, \alpha=1$')
    axs[2,0].plot(df2['beta'], color='orange', alpha=0.5, label=r'$\beta=0.1 \rightarrow 0.4, \alpha=1 \rightarrow 4$')
    axs[2,0].plot(df3['beta'], color='blue', alpha=0.5, label=r'$\beta=0.01, \alpha=1$')
    axs[2,0].plot(df4['beta'], color='steelblue', alpha=0.5, label=r'$\beta=0.01 \rightarrow 0.4, \alpha=1 \rightarrow 4$')
    axs[2,0].set_ylabel(r'$\beta$')
    axs[2,0].set_x_label('Epoch')
    #axs[2,0].legend(loc='upper right')

    axs[2,1].plot(df1['alpha'], color='red', alpha=0.5, label=r'$\beta=0.1, \alpha=1$')
    axs[2,1].plot(df2['alpha'], color='orange', alpha=0.5, label=r'$\beta=0.1 \rightarrow 0.4, \alpha=1 \rightarrow 4$')
    axs[2,1].plot(df3['alpha'], color='blue', alpha=0.5, label=r'$\beta=0.01, \alpha=1$')
    axs[2,1].plot(df4['alpha'], color='steelblue', alpha=0.5, label=r'$\beta=0.01 \rightarrow 0.4, \alpha=1 \rightarrow 4$')
    axs[2,1].set_ylabel(r'$\alpha$')
    axs[2,1].set_x_label('Epoch')
    #axs[2,1].legend(loc='upper right')
    
    filename='./fig_del_loss_details_'+group+'_'+postfix+'.pdf'
    fig.savefig(filename,bbox_inches='tight')


def plot_figures_compare_dgm(run_dir1, run_dir2, run_dir3, DEL=False, filename='generated_from_random_aggregated'):
    dataset_name = "ZINC" if "ZINC" in run_dir1 else "PCBA"
    if DEL:
        suffix='_del'
    df1 = pd.read_csv(os.path.join(run_dir1, 'results/samples'+suffix+'/'+filename+'.csv'))
    df0 = df1[df1.who==dataset_name]
    df1 = df1[df1.who=='DEL(1)']
    df1.who = r'$\beta$=1'
    
    df2 = pd.read_csv(os.path.join(run_dir2, 'results/samples'+suffix+'/'+filename+'.csv'))
    df2 = df2[df2.who=='DEL(1)']
    df2.who = r'$\beta$=0.1'
    
    df3 = pd.read_csv(os.path.join(run_dir3, 'results/samples'+suffix+'/'+filename+'.csv'))
    df3 = df3[df3.who=='DEL(1)']
    df3.who = r'$\beta$=0.01'
    df = pd.concat( [df0, df1, df2, df3], ignore_index=True)
    run_dir = './'
    plot_counts_multi(df, dataset_name, dataset_name+'_compare_dgm', dirsave=os.path.join(run_dir, 'results_for_publication'))
    plot_props_multi(df, dataset_name, dataset_name+'_compare_dgm', dirsave=os.path.join(run_dir, 'results_for_publication'))


def plot_MOBO(run_dir):
    filename = run_dir + '/results/bo/hvs_random.csv'
    sobol = np.loadtxt(filename, dtype=float, delimiter=',') 
    
    filename = run_dir + '/results/bo/hvs_qehvi.csv'
    qehvi =  np.loadtxt(filename, dtype=float, delimiter=',')
    
    filename = run_dir + '/results/bo/hvs_qparego.csv'
    qparego =  np.loadtxt(filename, dtype=float, delimiter=',') 
    
    fig = plt.figure(figsize=[4,3])
    ax = fig.add_subplot(111)
    ax.plot(sobol, label='Sobol', color='olive', alpha=0.5)
    ax.plot(qparego, label='qParEGO', color='blue', alpha=0.5)
    ax.plot(qehvi, label='qEHVI', color='red', alpha=0.5)
    ax.set_xlabel('Batch')
    ax.set_ylabel('Hypervolume')
    ax.legend(loc='upper left')
    filename = run_dir + '/results/bo/fig_hvs.pdf'
    fig.savefig(filename,bbox_inches='tight')

# DGM loss
run_dir1 = './RUNS/2020-08-15@10:39:34-yifeng-PCBA' # 1
run_dir2 = './RUNS/2020-08-12@23:59:17-yifeng-PCBA' # 2020-08-11@16:54:54-yifeng-PCBA 0.1
run_dir3 = './RUNS/2020-08-12@20:41:16-yifeng-PCBA' # 0.01
plot_dgm_loss_from_diff_expriments(run_dir1, run_dir2, run_dir3, group='epoch', postfix='PCBA')

run_dir1 = './RUNS/2020-08-15@19:58:55-yifeng-ZINC'
run_dir2 = './RUNS/2020-08-14@03:20:13-yifeng-ZINC'
run_dir3 = './RUNS/2020-08-13@22:56:46-yifeng-ZINC'
plot_dgm_loss_from_diff_expriments(run_dir1, run_dir2, run_dir3, group='epoch', postfix='ZINC')

# DEL loss
run_dir1 = './RUNS/2020-08-11@16:54:54-yifeng-PCBA' # 0.1
run_dir2 = './RUNS/2020-08-12@23:59:17-yifeng-PCBA' # 0.1 to 0.4
run_dir3 = './RUNS/2020-08-12@16:52:46-yifeng-PCBA' # 0.01
run_dir4 = './RUNS/2020-08-12@20:41:16-yifeng-PCBA' # 0.01 to 0.4
plot_del_loss_from_diff_expriments(run_dir1, run_dir2, run_dir3, run_dir4, group='epoch', postfix='PCBA')

run_dir1 = './RUNS/2020-08-13@08:17:08-yifeng-ZINC'
run_dir2 = './RUNS/2020-08-14@03:20:13-yifeng-ZINC'
run_dir3 = './RUNS/2020-08-13@18:51:18-yifeng-ZINC'
run_dir4 = './RUNS/2020-08-13@22:56:46-yifeng-ZINC'
plot_del_loss_from_diff_expriments(run_dir1, run_dir2, run_dir3, run_dir4, group='epoch', postfix='ZINC')

# DGM properties
run_dir1 = './RUNS/2020-08-15@10:39:34-yifeng-PCBA' # 1
run_dir2 = './RUNS/2020-08-12@23:59:17-yifeng-PCBA' # 2020-08-11@16:54:54-yifeng-PCBA 0.1
run_dir3 = './RUNS/2020-08-12@20:41:16-yifeng-PCBA' # 0.01
plot_figures_compare_dgm(run_dir1, run_dir2, run_dir3, DEL=True, filename='generated_from_random_aggregated')

run_dir1 = './RUNS/2020-08-15@19:58:55-yifeng-ZINC'
run_dir2 = './RUNS/2020-08-14@03:20:13-yifeng-ZINC'
run_dir3 = './RUNS/2020-08-13@22:56:46-yifeng-ZINC'
plot_figures_compare_dgm(run_dir1, run_dir2, run_dir3, DEL=True, filename='generated_from_random_aggregated')

# MOBO curves
run_dir = './RUNS/2020-08-12@20:41:16-yifeng-PCBA' # 0.01
plot_MOBO(run_dir)
run_dir = './RUNS/2020-08-12@23:59:17-yifeng-PCBA' # 0.1
plot_MOBO(run_dir)
run_dir = './RUNS/2020-08-14@03:20:13-yifeng-ZINC' # 0.1
plot_MOBO(run_dir)
run_dir = './RUNS/2020-08-13@22:56:46-yifeng-ZINC' # 0.01
plot_MOBO(run_dir)
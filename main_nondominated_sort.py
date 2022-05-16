#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 10:44:46 2020

@author: yifeng
"""

import numpy as np
import pandas as pd
from utils.postprocess import mask_novel_molecules

def fast_nondominated_sort( P ):
    """
    P is an numpy array of N by M where N is the number of data points / solutions, and M is the number is scores.
    
    Test code:
    import numpy as np
    import matplotlib.pyplot as plt

P = 100*np.random.rand( 1000,2)
rank = nondominated_sort(P)
M = rank.max()
for m in range(M):
    plt.plot(P[rank==m][:,0], P[rank==m][:,1], ls = '', marker ='o', markersize=4)
    
    plt.show()
    """
    N,M = P.shape
    
    inds_all_num = np.array( range(N) )
    
    Np = np.zeros(N, dtype=int) # number of solutions which dominate solution p
    rank = np.zeros(N, dtype=int)
    Sp = [] # set of solutions that p dominate
    Fs = []

    for n in range(N):
        diffs = P[n] - P 
        inds_le = ((diffs)<=0).all(axis=1)
        inds_l = ((diffs)<0).any(axis=1)
        inds = inds_le & inds_l            
        Sp.append ( inds_all_num[inds] )
        
        # >= & >
        #inds = ~inds_l & ~inds_le
        inds = ~(inds_l | inds_le)
        Np[n] = inds.sum()
    
    F=[]
    F = inds_all_num[Np == 0]
    rank[F] = 0
    
    i=0 # rank
    while len(F)>0:
        Fs.append(np.array(F))
        Q=[]
        for p in F:
            for q in Sp[p]:
                Np[q] = Np[q] - 1
                if Np[q] ==0:
                    rank[q] = i+1
                    Q.append(q)
        i = i + 1
        F = Q
        
    return rank, Fs

#%% compare DEL and BO
# load csv sample data from different files
#run_dir = './RUNS/2020-08-12@20:41:16-yifeng-PCBA'
#run_dir = './RUNS/2020-08-12@23:59:17-yifeng-PCBA'
#run_dir = './RUNS/2020-08-14@03:20:13-yifeng-ZINC'
run_dir = './RUNS/2020-08-13@22:56:46-yifeng-ZINC'
dataname='ZINC' # PCBA, ZINC
filename = run_dir + '/results/samples_del/new_pop_aggregated.csv'


del1 = pd.read_csv(filename)
data = del1[del1.who==dataname]
del1 = del1[(del1.loc[:,'rank']==0) & (del1.who=='DEL(F)')] # cann't use del1.rank here, because rank is a built-in keyword
stat={}
stat['total_DEL'] = [del1.shape[0]]

# load MOBO data
filename = run_dir + '/results/bo/samples_qehvi.csv'
mobo1 = pd.read_csv(filename)
mobo1 = mobo1[(mobo1.batch>=25) & (mobo1.SAS!=10)]
mobo1['who'] = 'qEHVI'
mobo1.rename(columns={'batch':'rank'}, inplace=True)
#mobo1.rank = 0
novel=mask_novel_molecules(mobo1.smiles.tolist(), data.smiles.tolist())
mobo1['novel']=novel
stat['total_qEHVI'] = [mobo1.shape[0]]

filename = run_dir + '/results/bo/samples_qparego.csv'
mobo2 = pd.read_csv(filename)
mobo2 = mobo2[(mobo2.batch>=25) & (mobo2.SAS!=10)]
mobo2['who'] = 'qParEGO'
mobo2.rename(columns={'batch':'rank'}, inplace=True)
#mobo2.rank = 0
novel=mask_novel_molecules(mobo2.smiles.tolist(), data.smiles.tolist())
mobo2['novel']=novel
stat['total_qParEGO'] = [mobo2.shape[0]]

# merge data
df=pd.concat([del1, mobo1, mobo2], ignore_index=True)

# get rank
P=df.loc[:,['qed', 'SAS', 'logP']]
P['qed'] = -P['qed']
P = P.to_numpy()
new_rank, new_Fs = fast_nondominated_sort( P )
df['new_rank'] = new_rank
df.sort_values(['new_rank'], inplace=True, kind='mergesort')
filename = run_dir + '/results/samples_del/compare_DEL_BO.csv'
df.to_csv(filename)

stat['DEL_in_0'] = [df[(df.new_rank==0) & (df.who=='DEL(F)')].shape[0]]
stat['qEHVI_in_0'] = [df[(df.new_rank==0) & (df.who=='qEHVI')].shape[0]]
stat['qParEGO_in_0'] = [df[(df.new_rank==0) & (df.who=='qParEGO')].shape[0]]
stat['DEL_in_0_percent'] = [stat['DEL_in_0'][0]/stat['total_DEL'][0]]
stat['qEHVI_in_0_percent'] = [stat['qEHVI_in_0'][0]/stat['total_qEHVI'][0]]
stat['qParEGO_in_0_percent'] = [stat['qParEGO_in_0'][0]/stat['total_qParEGO'][0]]
stat=pd.DataFrame(stat)
filename = run_dir + '/results/samples_del/compare_DEL_BO_stat.csv'
stat.to_csv(filename)
#%% compare linear and discrete crossover

# PCBA
run_dir1 = './RUNS/2020-08-12@20:41:16-yifeng-PCBA'
filename = run_dir1 + '/results/samples_del/new_pop_aggregated.csv'
del1 = pd.read_csv(filename)
#data = del1[del1.who=='PCBA']
del1 = del1[(del1.loc[:,'rank']==0) & (del1.who=='DEL(F)')] # cann't use del1.rank here, because rank is a built-in keyword
del1.who = 'linear'

run_dir2 = './RUNS/2020-08-13@04:46:35-yifeng-PCBA'
filename = run_dir2 + '/results/samples_del/new_pop_aggregated.csv'
del2 = pd.read_csv(filename)
#data = del2[del2.who=='PCBA']
del2 = del2[(del2.loc[:,'rank']==0) & (del2.who=='DEL(F)')] 
del2.who = 'discrete'

# merge data
df=pd.concat([del1, del2], ignore_index=True)
# get rank
P=df.loc[:,['qed', 'SAS', 'logP']]
P['qed'] = -P['qed']
P = P.to_numpy()
new_rank, new_Fs = fast_nondominated_sort( P )
df['new_rank'] = new_rank
df.sort_values(['new_rank'], inplace=True, kind='mergesort')
filename = run_dir1 + '/results/samples_del/compare_linear_discrete_crossover.csv'
df.to_csv(filename)

stat={}
stat['total_linear'] = [del1.shape[0]]
stat['total_discrete'] = [del2.shape[0]]
stat['linear_in_0'] = [df[(df.new_rank==0) & (df.who=='linear')].shape[0]]
stat['discrete_in_0'] = [df[(df.new_rank==0) & (df.who=='discrete')].shape[0]]
stat['linear_in_0_perent'] = [ stat['linear_in_0'][0]/stat['total_linear'][0] ] 
stat['discrete_in_0_perent'] = [ stat['discrete_in_0'][0]/stat['total_discrete'][0] ] 
stat=pd.DataFrame(stat)
filename = run_dir1 + '/results/samples_del/compare_linear_discrete_crossover_stat.csv'
stat.to_csv(filename)

# ZINC
run_dir1 = './RUNS/2020-08-13@22:56:46-yifeng-ZINC'
filename = run_dir1 + '/results/samples_del/new_pop_aggregated.csv'
del1 = pd.read_csv(filename)
data = del1[del1.who=='PCBA']
del1 = del1[(del1.loc[:,'rank']==0) & (del1.who=='DEL(F)')] # cann't use del1.rank here, because rank is a built-in keyword
del1.who = 'linear'


run_dir2 = './RUNS/2020-08-14@08:32:03-yifeng-ZINC'
filename = run_dir2 + '/results/samples_del/new_pop_aggregated.csv'
del2 = pd.read_csv(filename)
data = del2[del2.who=='PCBA']
del2 = del2[(del2.loc[:,'rank']==0) & (del2.who=='DEL(F)')] 
del2.who = 'discrete'

# merge data
df=pd.concat([del1, del2], ignore_index=True)
# get rank
P=df.loc[:,['qed', 'SAS', 'logP']]
P['qed'] = -P['qed']
P = P.to_numpy()
new_rank, new_Fs = fast_nondominated_sort( P )
df['new_rank'] = new_rank
df.sort_values(['new_rank'], inplace=True, kind='mergesort')
filename = run_dir1 + '/results/samples_del/compare_linear_discrete_crossover.csv'
df.to_csv(filename)

stat={}
stat['total_linear'] = [del1.shape[0]]
stat['total_discrete'] = [del2.shape[0]]
stat['linear_in_0'] = [df[(df.new_rank==0) & (df.who=='linear')].shape[0]]
stat['discrete_in_0'] = [df[(df.new_rank==0) & (df.who=='discrete')].shape[0]]
stat['linear_in_0_perent'] = [ stat['linear_in_0'][0]/stat['total_linear'][0] ] 
stat['discrete_in_0_perent'] = [ stat['discrete_in_0'][0]/stat['total_discrete'][0] ] 
stat=pd.DataFrame(stat)
filename = run_dir1 + '/results/samples_del/compare_linear_discrete_crossover_stat.csv'
stat.to_csv(filename)

#%% compare beta=0.1, 0.1 to 0.4, 0.01, 0.01 to 0.4
# PCBA
run_dir1 = './RUNS/2020-08-11@16:54:54-yifeng-PCBA' # 0.1
filename = run_dir1 + '/results/samples_del/new_pop_aggregated.csv'
del1 = pd.read_csv(filename)
#data = del1[del1.who=='PCBA']
del1 = del1[(del1.loc[:,'rank']==0) & (del1.who=='DEL(F)')] # cann't use del1.rank here, because rank is a built-in keyword
del1.who = 'beta=0.1'

run_dir2 = './RUNS/2020-08-12@23:59:17-yifeng-PCBA' # 0.1 to 0.4
filename = run_dir2 + '/results/samples_del/new_pop_aggregated.csv'
del2 = pd.read_csv(filename)
#data = del2[del2.who=='PCBA']
del2 = del2[(del2.loc[:,'rank']==0) & (del2.who=='DEL(F)')] 
del2.who = 'beta=0.1_to_0.4'

run_dir3 = './RUNS/2020-08-12@16:52:46-yifeng-PCBA' # 0.01
filename = run_dir3 + '/results/samples_del/new_pop_aggregated.csv'
del3 = pd.read_csv(filename)
#data = del3[del3.who=='PCBA']
del3 = del3[(del3.loc[:,'rank']==0) & (del3.who=='DEL(F)')] 
del3.who = 'beta=0.01'

run_dir4 = './RUNS/2020-08-12@20:41:16-yifeng-PCBA' # 0.01 to 0.4
filename = run_dir4 + '/results/samples_del/new_pop_aggregated.csv'
del4 = pd.read_csv(filename)
#data = del4[del4.who=='PCBA']
del4 = del4[(del4.loc[:,'rank']==0) & (del4.who=='DEL(F)')] 
del4.who = 'beta=0.01_to_0.4'

# merge data
df=pd.concat([del1, del2, del3, del4], ignore_index=True)
# get rank
P=df.loc[:,['qed', 'SAS', 'logP']]
P['qed'] = -P['qed']
P = P.to_numpy()
new_rank, new_Fs = fast_nondominated_sort( P )
df['new_rank'] = new_rank
df.sort_values(['new_rank'], inplace=True, kind='mergesort')
filename = run_dir1 + '/results/samples_del/compare_beta.csv'
df.to_csv(filename)

stat={}
stat['total_beta=0.1'] = [del1.shape[0]]
stat['total_beta=0.1_to_0.4'] = [del2.shape[0]]
stat['total_beta=0.01'] = [del3.shape[0]]
stat['total_beta=0.01_to_0.4'] = [del4.shape[0]]
stat['beta=0.1_in_0'] = [df[(df.new_rank==0) & (df.who=='beta=0.1')].shape[0]]
stat['beta=0.1_to_0.4_in_0'] = [df[(df.new_rank==0) & (df.who=='beta=0.1_to_0.4')].shape[0]]
stat['beta=0.01_in_0'] = [df[(df.new_rank==0) & (df.who=='beta=0.01')].shape[0]]
stat['beta=0.01_to_0.4_in_0'] = [df[(df.new_rank==0) & (df.who=='beta=0.01_to_0.4')].shape[0]]
stat['beta=0.1_in_0_perent'] = [ stat['beta=0.1_in_0'][0]/stat['total_beta=0.1'][0] ] 
stat['beta=0.1_to_0.4_in_0_perent'] = [ stat['beta=0.1_to_0.4_in_0'][0]/stat['total_beta=0.1_to_0.4'][0] ] 
stat['beta=0.01_in_0_perent'] = [ stat['beta=0.01_in_0'][0]/stat['total_beta=0.01'][0] ] 
stat['beta=0.01_to_0.4_in_0_perent'] = [ stat['beta=0.01_to_0.4_in_0'][0]/stat['total_beta=0.01_to_0.4'][0] ] 
stat=pd.DataFrame(stat)
filename = run_dir1 + '/results/samples_del/compare_beta_stat.csv'
stat.to_csv(filename)

#%% compare beta=0.1, 0.1 to 0.4, 0.01, 0.01 to 0.4
# ZINC
run_dir1 = './RUNS/2020-08-13@08:17:08-yifeng-ZINC' # 0.1
filename = run_dir1 + '/results/samples_del/new_pop_aggregated.csv'
del1 = pd.read_csv(filename)
#data = del1[del1.who=='PCBA']
del1 = del1[(del1.loc[:,'rank']==0) & (del1.who=='DEL(F)')] # cann't use del1.rank here, because rank is a built-in keyword
del1.who = 'beta=0.1'

run_dir2 = './RUNS/2020-08-14@03:20:13-yifeng-ZINC' # 0.1 to 0.4
filename = run_dir2 + '/results/samples_del/new_pop_aggregated.csv'
del2 = pd.read_csv(filename)
#data = del2[del2.who=='PCBA']
del2 = del2[(del2.loc[:,'rank']==0) & (del2.who=='DEL(F)')] 
del2.who = 'beta=0.1_to_0.4'

run_dir3 = './RUNS/2020-08-13@18:51:18-yifeng-ZINC' # 0.01
filename = run_dir3 + '/results/samples_del/new_pop_aggregated.csv'
del3 = pd.read_csv(filename)
#data = del3[del3.who=='PCBA']
del3 = del3[(del3.loc[:,'rank']==0) & (del3.who=='DEL(F)')] 
del3.who = 'beta=0.01'

run_dir4 = './RUNS/2020-08-13@22:56:46-yifeng-ZINC' # 0.01 to 0.4
filename = run_dir4 + '/results/samples_del/new_pop_aggregated.csv'
del4 = pd.read_csv(filename)
#data = del4[del4.who=='PCBA']
del4 = del4[(del4.loc[:,'rank']==0) & (del4.who=='DEL(F)')] 
del4.who = 'beta=0.01_to_0.4'

# merge data
df=pd.concat([del1, del2, del3, del4], ignore_index=True)
# get rank
P=df.loc[:,['qed', 'SAS', 'logP']]
P['qed'] = -P['qed']
P = P.to_numpy()
new_rank, new_Fs = fast_nondominated_sort( P )
df['new_rank'] = new_rank
df.sort_values(['new_rank'], inplace=True, kind='mergesort')
filename = run_dir1 + '/results/samples_del/compare_beta.csv'
df.to_csv(filename)

stat={}
stat['total_beta=0.1'] = [del1.shape[0]]
stat['total_beta=0.1_to_0.4'] = [del2.shape[0]]
stat['total_beta=0.01'] = [del3.shape[0]]
stat['total_beta=0.01_to_0.4'] = [del4.shape[0]]
stat['beta=0.1_in_0'] = [df[(df.new_rank==0) & (df.who=='beta=0.1')].shape[0]]
stat['beta=0.1_to_0.4_in_0'] = [df[(df.new_rank==0) & (df.who=='beta=0.1_to_0.4')].shape[0]]
stat['beta=0.01_in_0'] = [df[(df.new_rank==0) & (df.who=='beta=0.01')].shape[0]]
stat['beta=0.01_to_0.4_in_0'] = [df[(df.new_rank==0) & (df.who=='beta=0.01_to_0.4')].shape[0]]
stat['beta=0.1_in_0_perent'] = [ stat['beta=0.1_in_0'][0]/stat['total_beta=0.1'][0] ] 
stat['beta=0.1_to_0.4_in_0_perent'] = [ stat['beta=0.1_to_0.4_in_0'][0]/stat['total_beta=0.1_to_0.4'][0] ] 
stat['beta=0.01_in_0_perent'] = [ stat['beta=0.01_in_0'][0]/stat['total_beta=0.01'][0] ] 
stat['beta=0.01_to_0.4_in_0_perent'] = [ stat['beta=0.01_to_0.4_in_0'][0]/stat['total_beta=0.01_to_0.4'][0] ] 
stat=pd.DataFrame(stat)
filename = run_dir1 + '/results/samples_del/compare_beta_stat.csv'
stat.to_csv(filename)

#%% Compare with MLP vs without MLP
# PCBA
run_dir1 = './RUNS/2020-08-11@16:54:54-yifeng-PCBA' # with MLP, beta=0.1
filename = run_dir1 + '/results/samples_del/new_pop_aggregated.csv'
del1 = pd.read_csv(filename)
#data = del1[del1.who=='PCBA']
del1 = del1[(del1.loc[:,'rank']==0) & (del1.who=='DEL(F)')] # cann't use del1.rank here, because rank is a built-in keyword
del1.who = 'mlp'

run_dir2 = './RUNS/2020-08-12@03:46:00-yifeng-PCBA' # without MLP, beta=0.1
filename = run_dir2 + '/results/samples_del/new_pop_aggregated.csv'
del2 = pd.read_csv(filename)
#data = del2[del2.who=='PCBA']
del2 = del2[(del2.loc[:,'rank']==0) & (del2.who=='DEL(F)')] 
del2.who = 'no_mlp'

# merge data
df=pd.concat([del1, del2], ignore_index=True)
# get rank
P=df.loc[:,['qed', 'SAS', 'logP']]
P['qed'] = -P['qed']
P = P.to_numpy()
new_rank, new_Fs = fast_nondominated_sort( P )
df['new_rank'] = new_rank
df.sort_values(['new_rank'], inplace=True, kind='mergesort')
filename = run_dir1 + '/results/samples_del/compare_mlp_vs_nomlp.csv'
df.to_csv(filename)

stat={}
stat['total_mlp'] = [del1.shape[0]]
stat['total_nomlp'] = [del2.shape[0]]
stat['mlp_in_0'] = [df[(df.new_rank==0) & (df.who=='mlp')].shape[0]]
stat['nomlp_in_0'] = [df[(df.new_rank==0) & (df.who=='nomlp')].shape[0]]
stat['mlp_in_0_perent'] = [ stat['mlp_in_0'][0]/stat['total_mlp'][0] ] 
stat['nomlp_in_0_perent'] = [ stat['nomlp_in_0'][0]/stat['total_nomlp'][0] ] 
stat=pd.DataFrame(stat)
filename = run_dir1 + '/results/samples_del/compare_mlp_vs_nomlp_stat.csv'
stat.to_csv(filename)


# ZINC
run_dir1 = './RUNS/2020-08-13@08:17:08-yifeng-ZINC' # with MLP, beta=0.1
filename = run_dir1 + '/results/samples_del/new_pop_aggregated.csv'
del1 = pd.read_csv(filename)
#data = del1[del1.who=='PCBA']
del1 = del1[(del1.loc[:,'rank']==0) & (del1.who=='DEL(F)')] # cann't use del1.rank here, because rank is a built-in keyword
del1.who = 'mlp'

run_dir2 = './RUNS/2020-08-13@11:25:20-yifeng-ZINC' # without MLP, beta=0.1
filename = run_dir2 + '/results/samples_del/new_pop_aggregated.csv'
del2 = pd.read_csv(filename)
#data = del2[del2.who=='PCBA']
del2 = del2[(del2.loc[:,'rank']==0) & (del2.who=='DEL(F)')] 
del2.who = 'no_mlp'

# merge data
df=pd.concat([del1, del2], ignore_index=True)
# get rank
P=df.loc[:,['qed', 'SAS', 'logP']]
P['qed'] = -P['qed']
P = P.to_numpy()
new_rank, new_Fs = fast_nondominated_sort( P )
df['new_rank'] = new_rank
df.sort_values(['new_rank'], inplace=True, kind='mergesort')
filename = run_dir1 + '/results/samples_del/compare_mlp_vs_nomlp.csv'
df.to_csv(filename)

stat={}
stat['total_mlp'] = [del1.shape[0]]
stat['total_nomlp'] = [del2.shape[0]]
stat['mlp_in_0'] = [df[(df.new_rank==0) & (df.who=='mlp')].shape[0]]
stat['nomlp_in_0'] = [df[(df.new_rank==0) & (df.who=='nomlp')].shape[0]]
stat['mlp_in_0_perent'] = [ stat['mlp_in_0'][0]/stat['total_mlp'][0] ] 
stat['nomlp_in_0_perent'] = [ stat['nomlp_in_0'][0]/stat['total_nomlp'][0] ] 
stat=pd.DataFrame(stat)
filename = run_dir1 + '/results/samples_del/compare_mlp_vs_nomlp_stat.csv'
stat.to_csv(filename)

#%% Compare with finetune vs without finetune
# PCBA
run_dir1 = './RUNS/2020-08-11@16:54:54-yifeng-PCBA' # with finetune, beta=0.1
filename = run_dir1 + '/results/samples_del/new_pop_aggregated.csv'
del1 = pd.read_csv(filename)
#data = del1[del1.who=='PCBA']
del1 = del1[(del1.loc[:,'rank']==0) & (del1.who=='DEL(F)')] # cann't use del1.rank here, because rank is a built-in keyword
del1.who = 'finetune'

run_dir2 = './RUNS/2020-08-12@09:09:28-yifeng-PCBA' # without finetune, beta=0.1
filename = run_dir2 + '/results/samples_del/new_pop_aggregated.csv'
del2 = pd.read_csv(filename)
#data = del2[del2.who=='PCBA']
del2 = del2[(del2.loc[:,'rank']==0) & (del2.who=='DEL(F)')] 
del2.who = 'no_finetune'

# merge data
df=pd.concat([del1, del2], ignore_index=True)
# get rank
P=df.loc[:,['qed', 'SAS', 'logP']]
P['qed'] = -P['qed']
P = P.to_numpy()
new_rank, new_Fs = fast_nondominated_sort( P )
df['new_rank'] = new_rank
df.sort_values(['new_rank'], inplace=True, kind='mergesort')
filename = run_dir1 + '/results/samples_del/compare_finetune_vs_nofinetune.csv'
df.to_csv(filename)

stat={}
stat['total_finetune'] = [del1.shape[0]]
stat['total_nofinetune'] = [del2.shape[0]]
stat['finetune_in_0'] = [df[(df.new_rank==0) & (df.who=='finetune')].shape[0]]
stat['nofinetune_in_0'] = [df[(df.new_rank==0) & (df.who=='nofinetune')].shape[0]]
stat['finetune_in_0_perent'] = [ stat['finetune_in_0'][0]/stat['total_finetune'][0] ] 
stat['nofinetune_in_0_perent'] = [ stat['nofinetune_in_0'][0]/stat['total_nofinetune'][0] ] 
stat=pd.DataFrame(stat)
filename = run_dir1 + '/results/samples_del/compare_finetune_vs_nofinetune_stat.csv'
stat.to_csv(filename)


# ZINC
run_dir1 = './RUNS/2020-08-13@08:17:08-yifeng-ZINC' # with fineturn, beta=0.1
filename = run_dir1 + '/results/samples_del/new_pop_aggregated.csv'
del1 = pd.read_csv(filename)
#data = del1[del1.who=='PCBA']
del1 = del1[(del1.loc[:,'rank']==0) & (del1.who=='DEL(F)')] # cann't use del1.rank here, because rank is a built-in keyword
del1.who = 'finetune'

run_dir2 = './RUNS/2020-08-13@14:28:47-yifeng-ZINC' # without finetune, beta=0.1
filename = run_dir2 + '/results/samples_del/new_pop_aggregated.csv'
del2 = pd.read_csv(filename)
#data = del2[del2.who=='PCBA']
del2 = del2[(del2.loc[:,'rank']==0) & (del2.who=='DEL(F)')] 
del2.who = 'no_finetune'

# merge data
df=pd.concat([del1, del2], ignore_index=True)
# get rank
P=df.loc[:,['qed', 'SAS', 'logP']]
P['qed'] = -P['qed']
P = P.to_numpy()
new_rank, new_Fs = fast_nondominated_sort( P )
df['new_rank'] = new_rank
df.sort_values(['new_rank'], inplace=True, kind='mergesort')
filename = run_dir1 + '/results/samples_del/compare_finetune_vs_nofinetune.csv'
df.to_csv(filename)

stat={}
stat['total_finetune'] = [del1.shape[0]]
stat['total_nofinetune'] = [del2.shape[0]]
stat['finetune_in_0'] = [df[(df.new_rank==0) & (df.who=='finetune')].shape[0]]
stat['nofinetune_in_0'] = [df[(df.new_rank==0) & (df.who=='nofinetune')].shape[0]]
stat['finetune_in_0_perent'] = [ stat['finetune_in_0'][0]/stat['total_finetune'][0] ] 
stat['nofinetune_in_0_perent'] = [ stat['nofinetune_in_0'][0]/stat['total_nofinetune'][0] ] 
stat=pd.DataFrame(stat)
filename = run_dir1 + '/results/samples_del/compare_finetune_vs_nofinetune_stat.csv'
stat.to_csv(filename)

#%% compare 20K vs 100K
# PCBA
#dataname='PCBA'
#run_dir1 = './RUNS/2020-08-12@23:59:17-yifeng-PCBA' # 20K, beta=0.1 -> 0.4
#run_dir2 = './RUNS/2020-08-31@15:20:56-yifeng-PCBA' # 100K, beta=0.1 -> 0.4

#ZINC
dataname='ZINC'
run_dir1 = './RUNS/2020-08-14@03:20:13-yifeng-ZINC' # 20K, beta=0.1 -> 0.4
run_dir2 = './RUNS/2020-09-02@09:59:06-yifeng-ZINC' # 100K, beta=0.1 -> 0.4

filename = run_dir1 + '/results/samples_del/new_pop_aggregated.csv'
del1 = pd.read_csv(filename)
#data = del1[del1.who=='PCBA']
del1 = del1[(del1.loc[:,'rank']==0) & (del1.who=='DEL(F)')] # cann't use del1.rank here, because rank is a built-in keyword
del1.who = '20K'


filename = run_dir2 + '/results/samples_del/new_pop_aggregated.csv'
del2 = pd.read_csv(filename)
#data = del2[del2.who=='PCBA']
del2 = del2[(del2.loc[:,'rank']==0) & (del2.who=='DEL(F)')] 
del2.who = '100K'

# merge data
df=pd.concat([del1, del2], ignore_index=True)
# get rank
P=df.loc[:,['qed', 'SAS', 'logP']]
P['qed'] = -P['qed']
P = P.to_numpy()
new_rank, new_Fs = fast_nondominated_sort( P )
df['new_rank'] = new_rank
df.sort_values(['new_rank'], inplace=True, kind='mergesort')
filename = run_dir1 + '/results/samples_del/compare_20K_vs_100K.csv'
df.to_csv(filename)

stat={}
stat['total_20K'] = [del1.shape[0]]
stat['total_100K'] = [del2.shape[0]]
stat['20K_in_0'] = [df[(df.new_rank==0) & (df.who=='20K')].shape[0]]
stat['100K_in_0'] = [df[(df.new_rank==0) & (df.who=='100K')].shape[0]]
stat['20K_in_0_perent'] = [ stat['20K_in_0'][0]/stat['total_20K'][0] ] 
stat['100K_in_0_perent'] = [ stat['100K_in_0'][0]/stat['total_100K'][0] ] 
stat=pd.DataFrame(stat)
filename = run_dir1 + '/results/samples_del/compare_20K_vs_100K_stat.csv'
stat.to_csv(filename)

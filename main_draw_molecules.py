#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 22:20:45 2020

@author: yifeng
"""

import pandas as pd
from rdkit import Chem


#run_dir = './RUNS/2020-08-11@16_54_54-yifeng-PCBA' #0.1
#run_dir = './RUNS/2020-08-12@16_52_46-yifeng-PCBA' #0.01
#run_dir = './RUNS/2020-08-12@20_41_16-yifeng-PCBA' #0.01 -> 0.4
#run_dir = './RUNS/2020-08-12@23_59_17-yifeng-PCBA' #0.1 -> 0.4

#dataname = 'PCBA'
#threshold_qed = 0.85
#threshold_SAS = 3
#threshold_logP = 1

#filename = run_dir + '/results/samples_del/new_pop_aggregated.csv'
#data = pd.read_csv(filename)
#
#ind_delf = (data['who']=='DEL(F)') & (data['novel']==1) & (data['qed']>=threshold_qed) & (data['SAS']<=threshold_SAS) & (data['logP']<=threshold_logP)
#data_delf = data.loc[ind_delf]

#ms_delf = [ Chem.MolFromSmiles(s) for s in data_delf['smiles']] # molecule instances
##legends = ['QED:' + str(row['qed'])+'_SAS:'+str(row['SAS'])+'_logP'+str(row['logP']) for ind,row in data_delf.iterrows()]
#legends = ['QED:{0:.4f} SAS:{1:.2f} logP:{2:.2f}'.format(row['qed'],row['SAS'],row['logP']) for ind,row in data_delf.iterrows()]
#img=Chem.Draw.MolsToGridImage(ms_delf,molsPerRow=8,subImgSize=(200,200),legends=legends)    
#img.save(run_dir + '/results/samples_del/fig_mol_delf.png')

#ind_del5 = (data['who']=='DEL(5)') & (data['novel']==1) & (data['qed']>=threshold_qed) & (data['SAS']<=threshold_SAS) & (data['logP']<=threshold_logP)
#data_del5 = data.loc[ind_del5]
#
#ind_train = (data['who']=='PCBA') & (data['qed']>=threshold_qed) & (data['SAS']<=threshold_SAS) & (data['logP']<=threshold_logP)
#data_train = data.loc[ind_train]




def draw_mol(run_dir, dataname, threshold_qed=0.88, threshold_SAS=3, threshold_logP=1, maxMol=8*8):
    
    def mols2img(data_sel, run_dir, imagename):
        ms = [ Chem.MolFromSmiles(s) for s in data_sel['smiles']] # molecule instances
        legends = ['QED:{0:.4f} SAS:{1:.2f} logP:{2:.2f}'.format(row['qed'],row['SAS'],row['logP']) for ind,row in data_sel.iterrows()]
        img=Chem.Draw.MolsToGridImage(ms,molsPerRow=8,subImgSize=(300,300),legends=legends, maxMols=maxMol)
        img.save(run_dir + '/results/samples_del/'+imagename+'.png')
    
    filename = run_dir + '/results/samples_del/new_pop_aggregated.csv'
    data = pd.read_csv(filename)
    ind_delf = (data['who']=='DEL(F)') & (data['novel']==1) & (data['qed']>=threshold_qed) & (data['SAS']<=threshold_SAS) & (data['logP']<=threshold_logP)
    data_delf = data.loc[ind_delf]
    n_delf = data_delf.shape[0]
    if n_delf>0:
        mols2img(data_delf, run_dir, imagename='fig_mol_delf')
    
    ind_del5 = (data['who']=='DEL(5)') & (data['novel']==1) & (data['qed']>=threshold_qed) & (data['SAS']<=threshold_SAS) & (data['logP']<=threshold_logP)
    data_del5 = data.loc[ind_del5]
    n_del5 = data_del5.shape[0]
    if n_del5>0:
        mols2img(data_del5, run_dir, imagename='fig_mol_del5')

    ind_del1 = (data['who']=='DEL(1)') & (data['novel']==1) & (data['qed']>=threshold_qed) & (data['SAS']<=threshold_SAS) & (data['logP']<=threshold_logP)
    data_del1 = data.loc[ind_del1]
    n_del1 = data_del1.shape[0]
    if n_del1>0:
        mols2img(data_del1, run_dir, imagename='fig_mol_del1')
    
    ind_train = (data['who']==dataname) & (data['qed']>=threshold_qed) & (data['SAS']<=threshold_SAS) & (data['logP']<=threshold_logP)
    data_train = data.loc[ind_train]
    n_train = data_train.shape[0]
    if n_train>0:
        mols2img(data_train, run_dir, imagename='fig_mol_train')
    return n_train, n_del1, n_del5, n_delf


index=['ZINC_beta=0.1', 'ZINC_beta=0.01', 'ZINC_beta=0.1to0.4', 'ZINC_beta=0.01to0.4',
       'PCBA_beta=0.1', 'PCBA_beta=0.01', 'PCBA_beta=0.1to0.4', 'PCBA_beta=0.01to0.4']
counts = pd.DataFrame( {'n_train':[0]*8, 'n_del1':[0]*8, 'n_del5':[0]*8, 'n_delf':[0]*8}, index = index )

run_dir = './RUNS/2020-08-11@16_54_54-yifeng-PCBA' #0.1
count=draw_mol(run_dir, 'PCBA')
counts.loc['PCBA_beta=0.1']=count
run_dir = './RUNS/2020-08-12@16_52_46-yifeng-PCBA' #0.01
count=draw_mol(run_dir, 'PCBA')
counts.loc['PCBA_beta=0.01']=count
run_dir = './RUNS/2020-08-12@23_59_17-yifeng-PCBA' #0.1 -> 0.4
count=draw_mol(run_dir, 'PCBA')
counts.loc['PCBA_beta=0.1to0.4']=count
run_dir = './RUNS/2020-08-12@20_41_16-yifeng-PCBA' #0.01 -> 0.4
count=draw_mol(run_dir, 'PCBA')
counts.loc['PCBA_beta=0.01to0.4']=count

run_dir = './RUNS/2020-08-13@08_17_08-yifeng-ZINC' #0.1
count=draw_mol(run_dir, 'ZINC')
counts.loc['ZINC_beta=0.1']=count
run_dir = './RUNS/2020-08-13@18_51_18-yifeng-ZINC' #0.01
count=draw_mol(run_dir, 'ZINC')
counts.loc['ZINC_beta=0.01']=count
run_dir = './RUNS/2020-08-14@03_20_13-yifeng-ZINC' #0.1 -> 0.4
count=draw_mol(run_dir, 'ZINC')
counts.loc['ZINC_beta=0.1to0.4']=count
run_dir = './RUNS/2020-08-13@22_56_46-yifeng-ZINC' #0.01 -> 0.4
count=draw_mol(run_dir, 'ZINC')
counts.loc['ZINC_beta=0.01to0.4']=count
counts.to_csv('tab_counts_good_samples.csv')

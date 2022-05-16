#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 17:02:09 2020

@author: yifeng
"""
import pickle
import math
import time
import numpy as np
import pandas as pd
import copy
from joblib import Parallel, delayed

import torch
from torch.utils.data import DataLoader

from learner.trainer import Trainer, save_ckpt
from learner.dataset import MolTreeDataset, MolTreeFolder
from learner.sampler import Sampler
from fast_jtnn.datautils import tensorize
from fast_jtnn.mol_tree import MolTree


from molecules.properties import add_property
from molecules.structure import (add_atom_counts, add_bond_counts, add_ring_counts)

from utils.config import get_dataset_info
from utils.postprocess import score_samples

#loop
#train

# sample

class DEL_single():
    def __init__(self, config):
        self.config=config
        #self.model=None
        #self.population=None
        # self.dataset = FragmentDataset(self.config) # initialize population
        self.dataset = MolTreeDataset(self.config)
        self.population_size = self.config.get('population_size')#len(self.dataset)
        self.vocab = copy.deepcopy( self.dataset.get_vocab() )
        self.trainer = Trainer(self.config, self.vocab)


    def get_prop_stats(self, data):
        """
        data: data frame usually from self.dataset.data
        NOTE: sign of data['qed'] is flipped.
        """
        # flip sign, because of minimization
        data['qed'] = -data['qed']
        
        ss = dict()
        ss['wqed'] = 1.0/data['qed'].median()
        ss['wSAS'] = 1.0/data['SAS'].median()
        ss['wlogP'] = 1.0/data['logP'].median()
        ss['meanqed'] = data['qed'].mean()
        ss['meanSAS'] = data['SAS'].mean()
        ss['meanlogP'] = data['logP'].mean()
        ss['stdqed'] = data['qed'].std()
        ss['stdSAS'] = data['SAS'].std()
        ss['stdlogP'] = data['logP'].std()
        ss['maxqed'] = data['qed'].max()
        ss['maxSAS'] = data['SAS'].max()
        ss['maxlogP'] = data['logP'].max()
        ss['minqed'] = data['qed'].min()
        ss['minSAS'] = data['SAS'].min()
        ss['minlogP'] = data['logP'].min()
        return ss
    
    
    def normalize_dataset(self, data, method='standardnormal'):
        if method =='standardnormal':
            data['qed'] = (data['qed'] - self.ss['meanqed'])/self.ss['stdqed']
            data['SAS'] = (data['SAS'] - self.ss['meanSAS'])/self.ss['stdSAS']
            data['logP'] = (data['logP'] - self.ss['meanlogP'])/self.ss['stdlogP']
        elif method =='rescale01':
            self.ss['maxqed'] - data['qed'] / (self.ss['maxqed'] - self.ss['minqed'])
            self.ss['maxSAS'] - data['SAS'] / (self.ss['maxSAS'] - self.ss['minSAS'])
            self.ss['maxlogP'] - data['logP'] / (self.ss['maxlogP'] - self.ss['minlogP'])
        elif method =='rescalemedian1':
            data['qed'] = self.ss['wqed'] * data['qed']
            data['SAS'] = self.ss['wSAS'] * data['SAS']
            data['logP'] = self.ss['wlogP'] * data['logP']
    
        
    def get_single_objective(self, data, method='standardnormal', flip_qed_sign=False):
        """
        data: data frame in original scale.
        """
        properties = copy.deepcopy( data.loc[:,['qed', 'SAS', 'logP']])# qed: the larger the better, SAS: the smaller the better, logP: the smaller the better 
        if flip_qed_sign:
            properties['qed'] = -properties['qed'] # to be minimized
        #properties['SAS'] = properties['SAS'] # to be minimized
        self.normalize_dataset(properties, method)
        properties = properties.to_numpy()
        
        properties_single = properties[:,0] + properties[:,1] + properties[:,2]
        return properties, properties_single
    
    def train(self):
        print('!!!!SINGLE OBJECTIVE')
        d1 = {'validity_pop':[0]*self.config.get('num_generations'), 
              'novelty':[0]*self.config.get('num_generations'), 
              'diversity':[0]*self.config.get('num_generations'),
              'validity_sampler':[0]*self.config.get('num_generations')}
        vnds_pop = pd.DataFrame(d1)
        vnds_dgm = pd.DataFrame(d1)
        
        # initialize the model
        time_random_sampling = 0
        start_total = time.time()
        for g in range(self.config.get('num_generations')):
            print('Generation: {0}'.format(g))
            # train DGM
            if g == 0:
                # keep initial optimizer and scheduler info for letter use
                optim_lr = self.config.get('optim_lr')
                sched_step_size = self.config.get('sched_step_size')
                #alpha = self.config.get('alpha')
                a_beta = self.config.get('a_beta')
                a_alpha = self.config.get('a_alpha')
                dataset_rawtrain = copy.deepcopy(self.dataset) # keep a copy of the original training set
                dataset = copy.deepcopy(self.dataset) # keep a copy of the original training set

                # -------------normalization
                # rescale properties to have median 1
                #self.dataset.data['qed'] = self.wqed * self.dataset.data['qed']
                #self.dataset.data['SAS'] = self.wSAS * self.dataset.data['SAS']
                #self.dataset.data['logP'] = self.wlogP * self.dataset.data['logP']
                data = self.dataset.get_smile_data() # pd DataFrame with properties
                data['qed'] = -data['qed']  # flip sign, because of minimization
                self.ss = self.get_prop_stats(data)
                self.normalize_dataset(data) # normalize properties

                start_epoch = 0
                num_epochs = self.config.get('init_num_epochs')
                self.trainer.config.set('num_epochs', num_epochs)
                start_model_training = time.time()
                # self.trainer.train(self.dataset.get_loader(), start_epoch)
                # start training
                self.trainer.train(start_epoch)
                save_ckpt(self.trainer, num_epochs, filename="pretrain.pt")
                end_model_training = time.time()
                time_model_training = end_model_training - start_model_training
            else:
                ##self.dataset = FragmentDataset(self.config, kind='given', data=new_pop) # new population/data
                self.dataset.change_data(self.new_pop)
                # ------------normalization
                dataset = copy.deepcopy(self.dataset) # keep a copy of the unscaled training set
                data = self.dataset.get_smile_data() # pd DataFrame with properties
                data['qed'] = -data['qed']  # flip sign, because of minimization
                self.normalize_dataset(data) # normalize properties

                start_epoch = start_epoch + num_epochs
                num_epochs = self.config.get('subsequent_num_epochs')
                self.trainer.config.set('num_epochs', num_epochs)
                self.trainer.config.set('start_epoch', start_epoch)
                # reset learning rate and scheduler setting
                self.config.set('optim_lr', optim_lr )
                self.config.set('sched_step_size', math.ceil(sched_step_size/2) )
                if self.config.get('increase_beta')!=1:
                    if g==1:
                        self.config.set('offset_epoch', self.config.get('init_num_epochs')) # offset epoch for computing beta
                    else:
                        self.config.set('k_beta', 0) # let beta constant when g>1
                    self.config.set('a_beta', a_beta*self.config.get('increase_beta'))
                    #self.config.set('alpha',alpha*4)
                if self.config.get('increase_alpha')!=1:
                    if g==1:
                        self.config.set('offset_epoch', self.config.get('init_num_epochs')) # offset epoch for computing beta
                    else:
                        self.config.set('k_alpha', 0) # let beta constant when g>1
                    self.config.set('a_alpha', a_alpha*self.config.get('increase_alpha'))
                # comment this out if not finetuning DGM
                if self.config.get('no_finetune'):
                    print('DGM is not finetuned.')
                elif g < self.config.get('num_generations'):
                    # self.trainer.train(self.dataset.get_loader(), start_epoch)
                    self.trainer.train(start_epoch)
            
            # get current training data
            samples = self.dataset.get_smile_data() # pd DataFrame with properties
            # properties = samples.loc[:,['qed', 'SAS', 'logP']] # qed: the larger the better, SAS: the smaller the better, logP: the smaller the better 
            # properties['qed'] = -properties['qed'] # to be minimized
            # #properties['SAS'] = properties['SAS'] # to be minimized
            # properties = properties.to_numpy()

            # ----------get SO properties
            properties,properties_single = self.get_single_objective(samples, flip_qed_sign=True)
            
            
            # create a sampler for sampling from random noice
            self.sampler=Sampler(self.config, self.vocab, self.trainer.model) 
            
            if (self.population_size<=1000 and g<self.config.get('num_generations')) or (self.population_size>1000 and g in [0,math.ceil(self.config.get('num_generations')/2)-1,self.config.get('num_generations')-1]):
                # randomly generate samples with N(0,1) + decoder
                #if g in [0, self.config.get('num_generations')-1]:
                start_random_sampling = time.time()
                if g<self.config.get('num_generations')-1:
                    postfix=str(g)
                else:
                    postfix='final'
                num_samples=self.population_size
                if num_samples>30000:
                    num_samples=30000
                num_samples = 2000 # to save time
                # get jtVAE smaples      
                samples_random, valids_random=self.sampler.sample(num_samples=self.population_size, save_results=False, 
                                                   postfix=postfix, folder_name='samples_del')

                _,vnd_dgm = score_samples(samples_random, dataset.get_smile_data())
                vnd_dgm.append( np.array(valids_random).mean() )
                vnds_dgm.loc[g] = vnd_dgm
                
                samples_random = self.get_properties(samples_random)
                # adjust column order
                fieldnames = samples.columns.values.tolist()
                samples_random = samples_random.loc[:,fieldnames] # same order of fields as samples
                # save
                prefix = 'generated_from_random'
                folder_name='samples_del'
                filename = self.config.path(folder_name) / f"{prefix}_{postfix}.csv"
                samples_random.to_csv(filename)
                end_random_sampling = time.time()
                time_random_sampling = time_random_sampling + end_random_sampling - start_random_sampling
            
            # back to gpu
            if self.config.get('use_gpu'):
                self.trainer.model = self.trainer.model.cuda()
            
            
            # get latent representation of data
            print('Getting latent representations of current population...')
            z,mu,logvar = self.get_z()
            z = z.cpu().numpy()
            mu = mu.cpu().numpy()
            logvar = logvar.cpu().numpy()
            print(z)
            # evolutionary operations
            if g ==0:
            # if g<self.config.get('num_generations'):
                # p_min = np.min(properties, axis=0)
                # p_max = np.max(properties, axis=0)
                # if population =0, use all training samples, otherwise, use specified number.
                if self.population_size>0:
                    #samples = samples[0:self.population_size]
                    #properties = properties[0:self.population_size]
                    #z = z[0:self.population_size]
                    #smile_list = [x.strip("\r\n ") for x in open(self.config.get('processed_smile'))]
                    ind = np.random.choice(len(samples), self.population_size, replace=False )
                    # ind = np.random.choice(100, 10, replace=False )
                    print('mu---', mu.shape, 'z----', z.shape, 'ind---', ind)
                    samples = samples.loc[ind, :]
                    properties = properties[ind,:]
                    properties_single = properties_single[ind]
                    z = z[ind,:]
                    mu = mu[ind,:]
                    logvar = logvar[ind,:]
                # save z and original samples for t-SNE visualization
                self.save_z(g, z, mu, logvar, samples, prefix='traindata')
            
            #if g == self.config.get('num_generations'):
            #    break
            
            print('get ranks of individuals...')
            rank=np.argsort(properties_single)
            print('Evolutionary operations: selection, recombination, and mutation ...')
            z = self.evol_ops_single(z, rank, prob_ts = self.config.get('prob_ts'), 
                                     crossover_method =self.config.get('crossover'), 
                                     mutation_rate = self.config.get('mutation'))

            # print('Fast non-dominated sort to get fronts ...')
            # rank,Fs = self.fast_nondominated_sort(properties)
            # print('Crowding distance sort for each front ...')
            # _,dists_vec=self.crowding_distance_all_fronts(properties, Fs, p_min, p_max, n_jobs=-1)
            
            #pm=properties.mean().to_numpy() # properties mean
            
            # 应该是在modify z
            # z = self.evol_ops(z, rank, dists_vec, prob_ts = self.config.get('prob_ts'), crossover_method =self.config.get('crossover'), mutation_rate = self.config.get('mutation'))
            
            # eliminate sampling from modified z

            print('Generate new samples from modified z ...')
            print('z shape:', z.shape)
            # generate new samples
            #self.sampler=Sampler(self.config, self.vocab, self.trainer.model)
            # loader for z
            # zloader = DataLoader(z, batch_size=self.config.get('batch_size'))
            zloader = DataLoader(z, 1)
           
            

            # sample from batchs of z
            new_samples = [] # a list of tuples
            valids= []
            for zbatch in zloader:
                generated_sample = self.sampler.sample_from_z( zbatch, save_results=False, seed=None)
                new_samples.append(generated_sample) # list
                generated_valid = self.sampler.generate_valid(generated_sample)
                valids.append(generated_valid) # list
            z = z[valids] # remove invalid z
            # obtain fitness score / properties for generated samples
            new_samples = self.get_properties(new_samples) # samples now is a pd DataFrame
            
            # # obtain validity, novelty, and diversity of population data
            _,vnd_pop = score_samples(new_samples, dataset.get_smile_data())
            vnd_pop.append( np.array(valids).mean() )
            vnds_pop.loc[g] = vnd_pop
            
            # remove duplicates
            #new_samples = new_samples.drop_duplicates(subset = ["smiles"], ignore_index=True)
            new_samples = new_samples.drop_duplicates(subset = ["smiles"]).reset_index(drop=True)
            # print(new_samples.columns) 
            # #new_properties = new_samples.loc[:,['qed', 'SAS', 'logP']]
            # #new_properties = new_properties.to_numpy()
            
            print('Producing new generation of data ...')
            # # merge new samples with old population
            fieldnames = samples.columns.values.tolist()
            new_samples = new_samples.loc[:,fieldnames] # same order of fields as samples

            
            print(samples.shape)
            print(new_samples.shape)

            combined_samples = pd.concat( [samples, new_samples], ignore_index=True) # dataframe
            # 以防出错先沿用combined_samples
            # combined_samples = samples
            # remove duplicates
            combined_samples = combined_samples.drop_duplicates(subset = ["smiles"]).reset_index(drop=True)
            
            # combined properties
            #combined_properties = np.vstack( (properties, new_properties) ) # numpy array
            # combined_properties = combined_samples.loc[:,['qed', 'SAS', 'logP']]
            # combined_properties['qed'] = -combined_properties['qed'] # to be minimized
            # #combined_properties['SAS'] = combined_properties['SAS'] # to be minimized
            # combined_properties = combined_properties.to_numpy()
            combined_properties,combined_propertiess_single = self.get_single_objective(combined_samples, flip_qed_sign=True)

            # sort all samples
            rank=np.argsort(combined_propertiess_single)
            
            combined_size = len(rank)
            if combined_size < self.population_size:
                self.population_size = combined_size
            
            # take top samples
            rank=rank[0:self.population_size]
            
            # make new data of size self.population_size
            self.new_pop = combined_samples.loc[rank,:].reset_index(drop=True)
            print(self.new_pop)
            print(self.new_pop.shape)
            
            self.new_rank = np.arange(len(rank))
            self.new_combined_properties_single = combined_propertiess_single[rank]
                
            if g == self.config.get('num_generations')-1 or (g < self.config.get('num_generations')-1 and self.config.get('save_pops')):
                # self.save_population(g)
                self.save_population_single(g)
            # save latent representations
            if g in [0,math.ceil(self.config.get('num_generations')/2)-1,self.config.get('num_generations')-1]:
                # back to gpu
                if self.config.get('use_gpu'):
                    self.trainer.model = self.trainer.model.cuda()
                z,mu,logvar = self.get_z(self.new_pop)
                z = z.detach().numpy()
                mu = mu.cpu().detach().numpy()
                logvar = logvar.cpu().detach().numpy()
                self.save_z(g, z, mu, logvar, self.new_pop)
            
        # save running time    
        end_total = time.time()
        time_total = end_total - start_total
        elapsed_total = time.strftime("%H:%M:%S", time.gmtime(time_total))
        elapsed_random_sampling = time.strftime("%H:%M:%S", time.gmtime(time_random_sampling))
        elapsed_model_training = time.strftime("%H:%M:%S", time.gmtime(time_model_training))
        time_save = {'time_second': [time_total, time_random_sampling, time_model_training],
                     'time_hms': [elapsed_total, elapsed_random_sampling, elapsed_model_training] }
        time_save = pd.DataFrame(time_save, index=['total','random_sampling','model_training'])
        filename = self.config.path('performance') / f"running_time.csv"
        time_save.to_csv(filename)
        # save validity, novelty, and diversity
        filename = self.config.path('performance') / f"vnds_pop.csv"
        vnds_pop.to_csv(filename)
        filename = self.config.path('performance') / f"vnds_dgm.csv"
        vnds_dgm.to_csv(filename)
        
        print('Done DEL training.')
        return self.new_pop, self.new_rank
            
            
    def encode_z(self, inputs, lengths):
        with torch.no_grad():
            embeddings = self.trainer.model.embedder(inputs)
            z, mu, logvar = self.trainer.model.encoder(inputs, embeddings, lengths)
        return z, mu, logvar


    def get_z(self, data=None):
        if data is not None:
            dataset = copy.deepcopy(self.dataset)
            dataset.change_data(data)
            #dataset = FragmentDataset(config, kind='given', data)
        else:
            dataset = self.dataset
        
        df = self.dataset.get_smile_data()
        latent_size = int(self.config.get('latent_size') / 2)
        zs = torch.zeros(df.shape[0], latent_size)
        mus = torch.zeros(df.shape[0], latent_size)
        logvars = torch.zeros(df.shape[0], latent_size)
        # loader=dataset.get_loader() # load data

        # get latent space representation from JtVAE
        loader = MolTreeFolder(self.config, self.config.get('train_path'), self.vocab, self.config.get('batch_size'), num_workers=4)

        idx = 0

        for batch in loader:
            self.trainer.model.zero_grad()
            with torch.no_grad():
                loss, kl_div, wacc, tacc, sacc, z, mu, logvar, predicted = self.trainer.model(batch, self.config.get('beta'))
            start = idx*self.config.get('batch_size')
            end = start + z.shape[0]
            zs[start:end,:]=z
            mus[start:end, :]=mu
            logvars[start:end,:]=logvar
            idx += 1

        return zs,mus,logvars


    def save_z(self, g, z, mu=None, logvar=None, samples=None, prefix='pop'):
        """
        Save z for visualization purpose.
        g: scalar integer, the generation index.
        """
        if g==0:
            g='init'
        elif g == self.config.get('num_generations')-1:
            g='final'
        else:
            g=str(g)
            
        filename = self.config.path('samples_del') / f"{prefix}_z_{g}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump((z,mu,logvar,samples), file=f)
        
    
    def load_z(self, g, prefix='pop'):
        if g >= self.config.get('num_generations')-1:
            g='final'
        filename = self.config.path('samples_del') / f"{prefix}_z_{g}.pkl"
        with open(filename, 'rb') as f:
            z,mu,logvar,samples=pickle.load(f)
        return z,mu,logvar,samples


    def evol_ops(self, z, rank, dists_vec, prob_ts=0.95, crossover_method='linear', mutation_rate=0.01):
        """
        Selection, cross-over, mutation operations on z.
        """
        # selection
        N = rank.shape[0]
        selected_inds = self.tournament_selection_N(N, rank, dists_vec, prob_ts=prob_ts, k=2, n_jobs=-1)
        selected_points = z[selected_inds]
        
        new_data = []
        for n in range(0,N,2):
            ind1 = n
            ind2 = n+1
            if n==N-1: # N is odd
                ind1 = n-1
                ind2 = n
            new_point1,new_point2=self.crossover(point1=selected_points[ind1], point2=selected_points[ind2], method=crossover_method)
            self.mutate(new_point1, mutation_rate=mutation_rate)
            self.mutate(new_point2, mutation_rate=mutation_rate)
            
            new_data.append(new_point1)
            if n != N-1:
                new_data.append(new_point2)
            
        new_data = np.array(new_data)
        
        return new_data
    
    def evol_ops_single(self, z, rank, prob_ts=0.95, crossover_method='linear', mutation_rate=0.01):
        """
        Selection, cross-over, mutation operations on z.
        Single objective.
        """
        # selection
        N = rank.shape[0]
        selected_inds = self.tournament_selection_N_single(N, rank, prob_ts=prob_ts, k=2, n_jobs=-1)
        selected_points = z[selected_inds]
        
        new_data = []
        for n in range(0,N,2):
            ind1 = n
            ind2 = n+1
            if n==N-1: # N is odd
                ind1 = n-1
                ind2 = n
            ##new_point1,new_point2=self.crossover(point1=selected_points[ind1], point2=selected_points[ind2], method=crossover_method)
            new_point1=selected_points[ind1]
            new_point2=selected_points[ind2]
            new_point1,new_point2=self.crossover(point1=new_point1, point2=new_point2, method=crossover_method)
            self.mutate(new_point1, mutation_rate=mutation_rate)
            self.mutate(new_point2, mutation_rate=mutation_rate)
            
            new_data.append(new_point1)
            if n != N-1:
                new_data.append(new_point2)
            
        new_data = np.array(new_data)
        
        return new_data
    
    
    def tournament_selection(self, rank, dists_vec, prob_ts, k=2):
        N = len(rank)
        inds_num = np.array(range(N))
        # randomly selecting k points
        candidates=np.random.choice(inds_num, size=k, replace=False)
        
        # rank candidates
        rank_cand = rank[candidates] # prefer small rank
        # crowding distances
        dist_cand = -dists_vec[candidates] # perfer large distance
        # order these candidates
        order=np.lexsort( (dist_cand,rank_cand) )
        candidates_ordered = candidates[order]
        # assign probability
        probs = prob_ts*(1-prob_ts)**np.array(range(k)) 
        #inds_k = np.array( range(k) )
        #inds_k = inds_k[order]
        probs_cum=np.cumsum(probs)
        r=np.random.rand()
        sel_i = 0
        for i in range(k):
            sel_i = k-1 # initialize to the last
            if r<probs_cum[i]:
                sel_i = i
                break
        selected = candidates_ordered[sel_i]
        return selected


    def tournament_selection_single(self, rank, prob_ts, k=2):
        """
        Single-objective.
        """
        N = len(rank)
        inds_num = np.array(range(N))
        # randomly selecting k points
        candidates=np.random.choice(inds_num, size=k, replace=False)
        
        # rank candidates
        rank_cand = rank[candidates] # prefer small rank

        # order these candidates
        order=np.argsort( rank_cand )
        candidates_ordered = candidates[order]
        
        # assign probability
        probs = prob_ts*(1-prob_ts)**np.array(range(k)) 
        #inds_k = np.array( range(k) )
        #inds_k = inds_k[order]
        probs_cum=np.cumsum(probs)
        r=np.random.rand()
        sel_i = 0
        for i in range(k):
            sel_i = k-1 # initialize to the last
            if r<probs_cum[i]:
                sel_i = i
                break
        selected = candidates_ordered[sel_i]
        return selected

    
#    def tournament_selection_N(self, num, rank, dists_vec, prob_ts, k=2, n_jobs=-1):
#        """
#        Select num points.
#        k: scalar, number of points to be randomly selected from the population.
#        """
#        
#        pjob = Parallel(n_jobs=n_jobs, verbose=0)
#        selected_inds = pjob( delayed(self.tournament_selection)(rank, dists_vec, prob_ts, k) for n in range(num) )
#        return selected_inds
    
    def tournament_selection_N(self, num, rank, dists_vec, prob_ts, k=2, n_jobs=-1):
        """
        Select num points.
        k: scalar, number of points to be randomly selected from the population.
        """
        selected_inds = [ self.tournament_selection(rank, dists_vec, prob_ts, k) for n in range(num) ]
        return selected_inds
    
    
    def tournament_selection_N_single(self, num, rank, prob_ts, k=2, n_jobs=-1):
        """
        Select num points.
        k: scalar, number of points to be randomly selected from the population.
        Single objective.
        """
        selected_inds = [ self.tournament_selection_single(rank, prob_ts, k) for n in range(num) ]
        return selected_inds
    
    
    def crossover(self, point1, point2, method='linear'):
        """
        point1, point2: vector of size K, two data points.
        """
        K = point1.size
        
        if method == 'linear':
            d=0.25
            alpha = np.random.rand()
            alpha = -d + (1+2*d) * alpha 
            new_point1 = point1 + alpha*(point2 - point1)
            alpha = np.random.rand()
            alpha = -d + (1+2*d) * alpha 
            new_point2 = point1 + alpha*(point2 - point1)
        elif method == 'discrete':
            alpha=np.random.randint(K)
            new_point1 = np.zeros(K, dtype=np.float32)
            new_point1[:alpha]=point1[:alpha]
            new_point1[alpha:]=point2[alpha:]
            #alpha=np.random.randint(K)
            new_point2 = np.zeros(K, dtype=np.float32)
            new_point2[:alpha]=point2[:alpha]
            new_point2[alpha:]=point1[alpha:]
            #temp = np.copy( point2[:alpha] )
            #point2[:alpha] = point1[:alpha]
            #point1[:alpha] = temp
            #temp = np.copy( point2[alpha:] )
            #point2[alpha:] = point1[alpha:]
            #point1[alpha:] = temp
            
        return new_point1,new_point2
        
    
    def mutate(self, point, mutation_rate=0.01):
        p = np.random.rand()
        #mutation
        if p<mutation_rate:
            pos = np.random.randint(point.size)
            point[pos] = point[pos] + np.random.randn()
    
    
    def save_population(self, g):
        """
        g: integer, generation index.
        """
        if g == self.config.get('num_generations')-1:
            g='final'
        new_rank = pd.DataFrame(self.new_rank, columns=['rank']) 
        data_to_save = pd.concat( [self.new_pop, new_rank], axis=1 )
        data_to_save = data_to_save.sort_values('rank')
        filename = self.config.path('samples_del') / f"new_pop_{g}.csv"
        data_to_save.to_csv(filename)

    def save_population_single(self, g):
        """
        g: integer, generation index.
        Single-objective.
        """
        if g == self.config.get('num_generations')-1:
            g='final'
        new_rank = pd.DataFrame(self.new_rank, columns=['rank']) 
        new_prop_single = pd.DataFrame(self.new_combined_properties_single, columns=['score']) 
        data_to_save = pd.concat( [self.new_pop, new_rank, new_prop_single], axis=1 )
        data_to_save = data_to_save.sort_values('score')
        filename = self.config.path('samples_del') / f"new_pop_{g}.csv"
        data_to_save.to_csv(filename)


    def generate_sample(self):
        pass
    
    
    def get_properties(self, samples, n_jobs=-1):
        info = get_dataset_info(self.config.get('dataset'))
        
        # columns = ["smiles", "fragments", "n_fragments"]

        # JtVAE 不需要 fregment 列
        columns = ["smiles"]
        samples = pd.DataFrame(samples, columns=columns)
        
        samples = add_atom_counts(samples, info, n_jobs)
        samples = add_bond_counts(samples, info, n_jobs)
        samples = add_ring_counts(samples, info, n_jobs)

        # add same properties as in training/test dataset 
        for prop in info['properties']:
            samples = add_property(samples, prop, n_jobs)
        
        #samples.to_csv(config.path('samples') / 'aggregated.csv')
        return samples
    
    
    def fast_nondominated_sort( self, P ):
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
                    # print('sort----,', len(Sp) , len(F))
                    Np[q] = Np[q] - 1
                    if Np[q] ==0:
                        rank[q] = i+1
                        Q.append(q)
            i = i + 1
            F = Q
            
        return rank, Fs
        
        
    def crowding_distance_assignment( self, I, f_min, f_max ):
        """
        I: Numpy array of N by M. It can be the property matrix for one front. 
        """
        
        N,M = I.shape
        dists= np.zeros( N, dtype=float )
        for m in range(M): # for each property
            inds = np.argsort( I[:,m] )
            dists[inds[0]] = np.inf
            dists[inds[-1]] = np.inf
            dists[inds[1:-1]] = dists[inds[1:-1]] + (I[inds[2:],m] - I[inds[0:-2],m])/(f_max[m] - f_min[m])
            
        return dists
    
    
#    def crowding_distance_all_fronts( self, P, Fs, f_min, f_max, n_jobs=-1 ):
#        """
#        P: properties.
#        Fs: fronts.
#        f_min: min values of propreties
#        f_max: max value of properties
#        """
#        pjob = Parallel(n_jobs=n_jobs, verbose=0)
#        dists_all = pjob(delayed(self.crowding_distance_assignment)(P[F], f_min, f_max) for F in Fs)
#        dists_vec = np.zeros(P.shape[0])
#        for F,D in zip(Fs,dists_all):
#            dists_vec[F] = D
#        return dists_all, dists_vec
       
 
    def crowding_distance_all_fronts( self, P, Fs, f_min, f_max, n_jobs=-1 ):
        """
        P: properties.
        Fs: fronts.
        f_min: min values of propreties
        f_max: max value of properties
        """
        
        dists_all =[ self.crowding_distance_assignment(P[F], f_min, f_max) for F in Fs ]
        dists_vec = np.zeros(P.shape[0])
        for F,D in zip(Fs,dists_all):
            dists_vec[F] = D
        return dists_all, dists_vec
    

        
    


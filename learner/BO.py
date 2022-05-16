#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 17:02:09 2020

@author: yifeng
"""

# https://botorch.org/tutorials/multi_objective_bo
# https://botorch.org/tutorials/vae_mnist
# https://github.com/pytorch/botorch/blob/master/tutorials/multi_objective_bo.ipynb

import pickle
import math
import time
import numpy as np
import pandas as pd
import copy
from joblib import Parallel, delayed

from learner.trainer import Trainer, load_ckpt
from learner.dataset import FragmentDataset
from learner.sampler import Sampler

from molecules.properties import add_property
from molecules.structure import (add_atom_counts, add_bond_counts, add_ring_counts)

from utils.config import get_dataset_info


import torch
#from torch.utils.data import DataLoader
#tkwargs = {
#    "dtype": torch.double,
#    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
#}

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cpu"),
}

from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.utils.transforms import normalize, unnormalize
from botorch.utils.sampling import draw_sobol_normal_samples

from botorch.optim.optimize import optimize_acqf, optimize_acqf_list
from botorch.acquisition.objective import GenericMCObjective
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.multi_objective.box_decomposition import NondominatedPartitioning
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.utils.sampling import sample_simplex

from botorch import fit_gpytorch_model
from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.hypervolume import Hypervolume


import warnings
warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)


class BO():
    def __init__(self, config):
        self.config=config
        self.dataset = FragmentDataset(self.config) # initialize population
        self.population_size = self.config.get('population_size')#len(self.dataset)
        self.vocab = copy.deepcopy( self.dataset.get_vocab() )
        self.trainer = Trainer(self.config, self.vocab)
    
    def train(self):
        # initialize the model
        start_total = time.time()
        
        # train DGM
        #start_epoch = 0
        #num_epochs = self.config.get('num_epochs')
        #self.trainer.config.set('num_epochs', num_epochs)
        start_model_training = time.time()
        # load trained model
        load_ckpt(self.trainer, self.vocab )
        #self.trainer.train(self.dataset.get_loader(), start_epoch)
        end_model_training = time.time()
        time_model_training = end_model_training - start_model_training

        # BO
        model_dgm = self.trainer.model
        trainer = self.trainer
        sampler_dgm=Sampler(self.config, self.vocab, model_dgm)
        config = self.config
        dataset = self.dataset
        
        d = model_dgm.config.get('latent_size')
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        dtype = torch.double
        bounds = torch.tensor([[-3.0] * d, [3.0] * d], device=device, dtype=dtype)
        standard_bounds = torch.zeros(2, d, **tkwargs)
        standard_bounds[1] = 1
        
        BATCH_SIZE = config.get('batch_size_bo')
        #standard_bounds = torch.zeros(2, problem.dim, **tkwargs)
        #standard_bounds[1] = 1
        num_outcomes = 3
        
        maxlogP = dataset.data['logP'].max()
        ref_point = [0, -10, -maxlogP]
        
        N_TRIALS = config.get('n_trials') # 3, update
        N_BATCH = config.get('n_batch') #25 # number of batches
        MC_SAMPLES = config.get('mc_samples') #32 #128
        verbose = True
        
        #def problem(z, sampler_dgm=sampler_dgm, config=config):
        def problem(z):
            """
            z: torch tensor.
            """
            z=z.float()
            #print('value of z in problem:')
            #print(z)
            samples, valids = sampler_dgm.sample_from_z( z, save_results=False, seed=None)
            if config.get('use_gpu'):
                trainer.model = trainer.model.cuda()
            ## DO MORE HERE, SET invalid points to a fix value
            props = torch.zeros(z.shape[0], 3, device=device, dtype=dtype)
            #print(props)
            # invalid samples will have bad values
            props[:,0] = 0
            props[:,1] = -10
            props[:,2] = -maxlogP
            #print(props)
            #samples = get_props(samples, config)
            samples = get_props(samples)
            props_valid = samples.loc[:,['qed', 'SAS', 'logP']]
            props_valid['SAS'] = -props_valid['SAS'] # to be maximized
            props_valid['logP'] = -props_valid['logP'] # to be maximized
            props_valid = props_valid.to_numpy(np.double)
            props_valid = torch.from_numpy(props_valid)#.to(device)
            #print(props_valid.shape)
            
            #print(valids)
            #print(props_valid)
            #props_valid = props_valid.to(device)
            #print(props_valid.data)
            props[valids] = props_valid# .to(device)
            #props = props.to(device)
            return props
    
        
        def decode_from_z(z):
            """
            z: torch tensor.
            """
            z=z.float()
            #print('value of z in problem:')
            #print(z)
            samples, valids = sampler_dgm.sample_from_z( z, save_results=False, seed=None)
            if config.get('use_gpu'):
                trainer.model = trainer.model.cuda()
            
            #print(props)
            #samples = get_props(samples, config)
            samples = get_props(samples)
            fieldnames=dataset.data.columns.values.tolist()
            samples = samples.loc[:,fieldnames]
            print('samples shape:')
            print(samples.shape)
            print(valids.shape)
            print(z.shape)
            # create empty daa frame for all samples
            samples_all = pd.DataFrame(np.nan, index=range(len(valids)), columns=fieldnames)
            print(samples_all.shape)
            samples_all['qed'] = 0
            samples_all['SAS'] = 10
            samples_all['logP'] = 20
            samples_all.loc[valids] = samples.values
            return samples_all
        
        
        #def get_props(samples, config, n_jobs=-1):
        def get_props(samples, n_jobs=-1):
            info = get_dataset_info(config.get('dataset'))
            
            columns = ["smiles", "fragments", "n_fragments"]
            samples = pd.DataFrame(samples, columns=columns)
            
            samples = add_atom_counts(samples, info, n_jobs)
            samples = add_bond_counts(samples, info, n_jobs)
            samples = add_ring_counts(samples, info, n_jobs)
            
            # add same properties as in training/test dataset 
            for prop in info['properties']:
                samples = add_property(samples, prop, n_jobs)
            
            return samples
        
        
        #def encode_z(model_dgm, inputs, lengths):
        def encode_z(inputs, lengths):
            with torch.no_grad():
                embeddings = trainer.model.embedder(inputs)
                z, mu, logvar = trainer.model.encoder(inputs, embeddings, lengths)
            return z, mu, logvar
        
        
        #def get_z(model_dgm, dataset, config):
        def get_z(dataset, config):
            zs = torch.zeros(len(dataset), config.get('latent_size'), dtype=dtype, device=device)
            mus = torch.zeros(len(dataset), config.get('latent_size'), dtype=dtype, device=device)
            logvars = torch.zeros(len(dataset), config.get('latent_size'), dtype=dtype, device=device)
            loader=dataset.get_loader() # load data
            for idx, (src, tgt, lengths, properties) in enumerate(loader):
                if config.get('use_gpu'):
                    src = src.cuda()
                    tgt = tgt.cuda()
                    properties = properties.cuda()
                #z, mu, logvar = encode_z(model_dgm, src,lengths)
                z, mu, logvar = encode_z(src,lengths)
                start = idx*config.get('batch_size')
                end = start + z.shape[0]
                zs[start:end,:]=z
                mus[start:end,:]=mu
                logvars[start:end,:]=logvar
            return zs,mus,logvars
        
        
        # get initial data
        def get_initial_data_from_trainset(n):
            samples = dataset.data # pd DataFrame with properties
            properties = samples.loc[:,['qed', 'SAS', 'logP']] # qed: the larger the better, SAS: the smaller the better, logP: the smaller the better 
            properties['SAS'] = -properties['SAS'] # to be maximized
            properties['logP'] = -properties['logP'] # to be maximized
            properties = properties.to_numpy()
            properties = torch.tensor(properties, device=device, dtype=dtype)
            zs_all,_,_ = get_z(dataset, config)
            #ind = np.random.choice(zs_all.shape[0],size=8,replace=False)
            ind=torch.randperm(zs_all.shape[0])[:n]
            zs = zs_all[ind]
            props = properties[ind]
            zs = zs.double()
            samples_selected = samples.loc[ind.numpy()]
            return zs,props,samples_selected
        
        
        def generate_initial_data(n):
            zs=draw_sobol_normal_samples(d, n)
            props=problem(zs)
            return zs,props
            
        
        def initialize_model(train_x, train_obj):
            # define models for objective and constraint
            model_gp = SingleTaskGP(train_x, train_obj)
            mll = ExactMarginalLogLikelihood(model_gp.likelihood, model_gp)
            return mll, model_gp
        
        
        def optimize_qehvi_and_get_observation(model_gp, train_obj, sampler):
            """Optimizes the qEHVI acquisition function, and returns a new candidate and observation."""
            # partition non-dominated space into disjoint rectangles
            partitioning = NondominatedPartitioning(num_outcomes=num_outcomes, Y=train_obj)
            acq_func = qExpectedHypervolumeImprovement(
                model=model_gp,
                ref_point=torch.tensor(ref_point, dtype=dtype, device=device),  # use known reference point 
                partitioning=partitioning,
                sampler=sampler
            )
            # optimize
            candidates, _ = optimize_acqf(
                acq_function=acq_func,
                bounds=standard_bounds,
                q=BATCH_SIZE,
                num_restarts=20,
                raw_samples=1024,  # used for intialization heuristic
                options={"batch_limit": 5, "maxiter": 200},
                sequential=True
            )
            # observe new values
            new_x =  unnormalize(candidates.detach(), bounds=bounds) # ?????
            #new_x =  candidates.detach()
            new_obj = problem(new_x)
            new_obj = new_obj.to(device)
            return new_x, new_obj
        
        
        def update_random_observations(best_random):
            """Simulates a random policy by taking a the current list of best values observed randomly,
            drawing a new random point, observing its value, and updating the list.
            """
            rand_x = torch.rand(BATCH_SIZE, d)
            #rand_x = unnormalize(torch.rand(BATCH_SIZE, d), bounds)
            next_random_best = problem(rand_x).max().item()
            best_random.append(max(best_random[-1], next_random_best))       
            return best_random
        
        
        def optimize_qparego_and_get_observation(model_gp, train_obj, sampler):
            """Samples a set of random weights for each candidate in the batch, performs sequential greedy optimization 
            of the qParEGO acquisition function, and returns a new candidate and observation."""
            acq_func_list = []
            for _ in range(BATCH_SIZE):
                weights = sample_simplex(num_outcomes, **tkwargs).squeeze()
                objective = GenericMCObjective(get_chebyshev_scalarization(weights=weights, Y=train_obj))
                acq_func = qExpectedImprovement(  # pyre-ignore: [28]
                    model=model_gp,
                    objective=objective,
                    best_f=objective(train_obj).max().item(),
                    sampler=sampler
                )
                acq_func_list.append(acq_func)
            # optimize
            candidates, _ = optimize_acqf_list(
                acq_function_list=acq_func_list,
                bounds=standard_bounds,
                num_restarts=20,
                raw_samples=1024,  # used for intialization heuristic
                options={"batch_limit": 5, "maxiter": 200}
            )
            # observe new values 
            new_x =  unnormalize(candidates.detach(), bounds=bounds) # ??????????
            #new_x =  candidates.detach() # ??????????
            new_obj = problem(new_x)
            new_obj = new_obj.to(device)
            return new_x, new_obj
        

        
        hvs_qparego_all, hvs_qehvi_all, hvs_random_all = [], [], []
        
        hv = Hypervolume(ref_point=torch.tensor(ref_point, dtype=dtype, device=device))
        
        # average over multiple trials
        for trial in range(1, N_TRIALS + 1):
            torch.manual_seed(trial)
            
            print(f"\nTrial {trial:>2} of {N_TRIALS} ", end="")
            hvs_qparego, hvs_qehvi, hvs_random = [], [], []
            
            # call helper functions to generate initial training data and initialize model
            #train_x_qparego = zs_init
            #train_obj_qparego = properties_init
            train_x_qparego, train_obj_qparego, samples_selected = get_initial_data_from_trainset(config.get('num_initial_samples'))
            
            train_x_qparego = normalize(train_x_qparego, bounds=bounds)
            mll_qparego, model_qparego = initialize_model(train_x_qparego, train_obj_qparego)
            train_x_qparego = unnormalize(train_x_qparego, bounds=bounds)
            
            train_x_qehvi, train_obj_qehvi = train_x_qparego, train_obj_qparego
            train_x_random, train_obj_random = train_x_qparego, train_obj_qparego
            
            # compute hypervolume 
            train_x_qehvi = normalize(train_x_qehvi, bounds=bounds)
            mll_qehvi, model_qehvi = initialize_model(train_x_qehvi, train_obj_qehvi)
            train_x_qehvi = unnormalize(train_x_qehvi, bounds=bounds)
            
            # compute pareto front
            pareto_mask = is_non_dominated(train_obj_qparego)
            pareto_y = train_obj_qparego[pareto_mask]
            # compute hypervolume
            
            volume = hv.compute(pareto_y)
            
            hvs_qparego.append(volume)
            hvs_qehvi.append(volume)
            hvs_random.append(volume)
            
            # run N_BATCH rounds of BayesOpt after the initial random batch
            for iteration in range(1, N_BATCH + 1):    
                
                t0 = time.time()
                
                # fit the models
                fit_gpytorch_model(mll_qparego)
                fit_gpytorch_model(mll_qehvi)
                
                # define the qEI and qNEI acquisition modules using a QMC sampler
                qparego_sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES)
                qehvi_sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES)
                
                # optimize acquisition functions and get new observations
                new_x_qparego, new_obj_qparego = optimize_qparego_and_get_observation(
                    model_qparego, train_obj_qparego, qparego_sampler
                )
                new_x_qehvi, new_obj_qehvi = optimize_qehvi_and_get_observation(
                    model_qehvi, train_obj_qehvi, qehvi_sampler
                )
                new_x_random, new_obj_random = generate_initial_data(n=BATCH_SIZE)
                        
                # update training points
                train_x_qparego = torch.cat([train_x_qparego, new_x_qparego])
                train_obj_qparego = torch.cat([train_obj_qparego, new_obj_qparego])
        
                train_x_qehvi = torch.cat([train_x_qehvi, new_x_qehvi])
                train_obj_qehvi = torch.cat([train_obj_qehvi, new_obj_qehvi])
                
                new_x_random = new_x_random.double()
                train_x_random = torch.cat([train_x_random, new_x_random])
                train_obj_random = torch.cat([train_obj_random, new_obj_random])
                
        
                # update progress
                for hvs_list, train_obj in zip(
                    (hvs_random, hvs_qparego, hvs_qehvi), 
                    (train_obj_random, train_obj_qparego, train_obj_qehvi),
                ):
                    # compute pareto front
                    pareto_mask = is_non_dominated(train_obj)
                    pareto_y = train_obj[pareto_mask]
                    # compute hypervolume
                    volume = hv.compute(pareto_y)
                    hvs_list.append(volume)
        
                # reinitialize the models so they are ready for fitting on next iteration
                # Note: we find improved performance from not warm starting the model hyperparameters
                # using the hyperparameters from the previous iteration
                
                train_x_qparego = normalize(train_x_qparego, bounds=bounds)
                mll_qparego, model_qparego = initialize_model(train_x_qparego, train_obj_qparego)
                train_x_qparego = unnormalize(train_x_qparego, bounds=bounds)
                
                train_x_qehvi = normalize(train_x_qehvi, bounds=bounds)
                mll_qehvi, model_qehvi = initialize_model(train_x_qehvi, train_obj_qehvi)
                train_x_qehvi = unnormalize(train_x_qehvi, bounds=bounds)
                
                #mll_qparego, model_qparego = initialize_model(train_x_qparego, train_obj_qparego)
                #mll_qehvi, model_qehvi = initialize_model(train_x_qehvi, train_obj_qehvi)
                
                t1 = time.time()
                
                if verbose:
                    print(
                        f"\nBatch {iteration:>2}: Hypervolume (random, qParEGO, qEHVI) = "
                        f"({hvs_random[-1]:>4.6f}, {hvs_qparego[-1]:>4.6f}, {hvs_qehvi[-1]:>4.6f}), "
                        f"time = {t1-t0:>4.4f}.", end=""
                    )
                else:
                    print(".", end="")
           
            hvs_qparego_all.append(hvs_qparego)
            hvs_qehvi_all.append(hvs_qehvi)
            hvs_random_all.append(hvs_random)
        
        end_total = time.time()
        time_total = end_total - start_total
        elapsed_total = time.strftime("%H:%M:%S", time.gmtime(time_total))
        elapsed_model_training = time.strftime("%H:%M:%S", time.gmtime(time_model_training))
        time_save = {'time_second': [time_total, time_model_training],
                     'time_hms': [elapsed_total, elapsed_model_training] }
        time_save = pd.DataFrame(time_save, index=['total','model_training'])
        filename = self.config.path('bo') / f"running_time_bo.csv"
        time_save.to_csv(filename)
        
        
        
        # get molecule data for each set of solutions
        batch_number = torch.cat([torch.zeros(config.get('num_initial_samples'), dtype=torch.int64), torch.arange(1, N_BATCH+1).repeat(BATCH_SIZE, 1).t().reshape(-1)]).numpy()
        train_samples_qehvi = decode_from_z(train_x_qehvi)
        train_samples_qehvi[0:config.get('num_initial_samples')] = samples_selected.values
        train_samples_qehvi['batch'] = batch_number
        train_samples_qparego = decode_from_z(train_x_qparego)
        train_samples_qparego[0:config.get('num_initial_samples')] = samples_selected.values
        train_samples_qparego['batch'] = batch_number
        train_samples_random = decode_from_z(train_x_random)
        train_samples_random[0:config.get('num_initial_samples')] = samples_selected.values
        train_samples_random['batch'] = batch_number
        # save samples
        filename = self.config.path('bo') / f"samples_qehvi.csv"
        train_samples_qehvi.to_csv(filename)
        filename = self.config.path('bo') / f"samples_qparego.csv"
        train_samples_qparego.to_csv(filename)
        filename = self.config.path('bo') / f"samples_random.csv"
        train_samples_random.to_csv(filename)
        # save z
        filename = self.config.path('bo') / f"zs.pkl"
        with open(filename, 'wb') as f:
            pickle.dump((train_x_qehvi,train_x_qparego,train_x_random), file=f)
        
        # save hvs
        filename = self.config.path('bo') / f"hvs_qehvi.csv"
        np.savetxt(filename, hvs_qehvi_all, fmt='%4.6f', delimiter=',')
        filename = self.config.path('bo') / f"hvs_qparego.csv"
        np.savetxt(filename, hvs_qparego_all, fmt='%4.6f', delimiter=',')
        filename = self.config.path('bo') / f"hvs_random.csv"
        np.savetxt(filename, hvs_random_all, fmt='%4.6f', delimiter=',')

            
        print(train_x_qehvi.shape)
        print(train_x_qparego.shape)
        print(train_x_random.shape)     
            
            
            

    

        
    

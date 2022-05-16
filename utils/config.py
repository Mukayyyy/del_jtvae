import os
import socket
import torch
import random
import numpy as np
from datetime import datetime
from pathlib import Path
import shutil

from .filesystem import (
    load_pickle, save_pickle, commit, save_json, load_json)


PROJ_DIR = Path('.')
DATA_DIR = PROJ_DIR / 'DATA'
RUNS_DIR = PROJ_DIR / 'RUNS'


DEFAULTS = {
    # general
    'title': 'Molecule Generator',
    'description': 'DEL Based Molecule Generator',
    'log_dir': RUNS_DIR.as_posix(),
    'random_seed': 42,
    'use_gpu': False,
    'processed_smile': 'fast_jtnn/zinc/all.txt',
     #'processed_smile': 'fast_jtnn/zinc/ZBD_smile.txt',
    'train_path': 'fast_jtnn/zinc-processed-all',
    #'train_path': 'fast_jtnn/zbd-deleted-processed',
    'vocab_path': 'fast_jtnn/zinc/vocab.txt',
    #'vocab_path': 'fast_jtnn/zinc/zbd_vocab.txt',
    'BCC_path': 'predictive_model/BCC_smiles.txt',
    'protein_path': 'DATA/protein_files/6rqu.pdbqt',
    'save_dir': 'vae_model',
    'load_epoch': 0,
    # data
    'batch_size': 32,
    'shuffle': True,
    'use_mask': True,
    # model
    'embed_size': 256,
    'embed_window': 3,
    'mask_freq': 10,
    'num_clusters': 30,
    # 'hidden_size': 64,
    # 'hidden_layers': 2,
    'hidden_size': 450,
    'hidden_layers': 1,
    'dropout': 0.3,
    'latent_size': 56,
    'depthT': 20,
    'depthG': 3,
    # learning
    # 'num_epochs': 10,
    'num_epochs': 10,
    'optim_lr': 1e-3,
    'use_scheduler': True,
    'sched_step_size': 2,
    'sched_gamma': 0.9,
    'clip_norm': 50.0,
    'beta': 0.0,
    'step_beta': 0.002,
    'max_beta': 1.0,
    'warmup': 40000,
    'anneal_rate': 0.9,
    'anneal_iter': 40000,
    'kl_anneal_iter': 1000,
    'print_iter': 50,
    'save_iter': 5000,
    # sampler
    'load_last': False,
    'validate_after': 0.3,
    'validation_samples': 300,
    'num_samples': 100,
    'max_length': 10,
    'temperature': 0.8,
    'reproduce': False,
    'sampling_seed': None,
    # predictor
    'predictor_num_layers': 3,
    'predictor_hidden_size': 64,
    'predictor_output_size':3,
    'a_alpha': 1, # for calculating alpha
    'l_alpha': 0, # for calculating alpha
    'u_alpha': 10, # for calculating alpha
    'k_alpha': 0, # for calculating alpha
    'a_beta': 1, # for calculating beta
    'l_beta': 0, # for calculating beta
    'u_beta': 10, # for calculating beta
    'k_beta': 1, # for calculating beta
    'start_epoch': 0,
    'offset_epoch': 0,
    # del
    'num_generations': 100,
    'population_size': 10000,
    'init_num_epochs': 70,
    'subsequent_num_epochs': 10,
    'prob_ts': 0.95,
    'crossover': 'linear',
    'mutation': 0.01,
    'save_pops': False,
    'no_finetune': False,
    'single_objective': False,
    'epsilon': 0.001, # learning rate to updata z in latent space
    'increase_alpha': 1,
    'increase_beta': 1,
    # BO
    'num_initial_samples': 2000,
    'batch_size_bo': 64,
    'n_batch': 10,
    'mc_samples': 128,
    'n_trials': 1
}


def set_random_seed(seed=None):
    if seed is None:
        seed = random.randint(0, 2**32-1)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed


def get_run_info(name):
    start_time = datetime.now().strftime('%Y-%m-%d@%X')
    host_name = socket.gethostname()
    run_name = f'{start_time}-{host_name}-{name}'
    return run_name, host_name, start_time


def get_dataset_info(name):
    path = PROJ_DIR / 'utils/data' / f'{name}.json'
    return load_json(path)


def get_text_summary(params):
    start_time = params.get('start_time')
    tag = (f"Experiment params: {params.get('title')}\n")

    text = f"<h3>{tag}</h3>\n"
    text += f"{params.get('description')}\n"

    text += '<pre>'
    text += f"Start Time: {start_time}\n"
    text += f"Host Name: {params.get('host_name')}\n"
    text += f'CWD: {os.getcwd()}\n'
    text += f'PID: {os.getpid()}\n'
    text += f"Commit Hash: {params.get('commit_hash')}\n"
    text += f"Random Seed: {params.get('random_seed')}\n"
    text += '</pre>\n<pre>'

    skip_keys = ['title', 'description', 'random_seed', 'run_name']
    for key, val in params.items():
        if key in skip_keys:
            continue
        text += f'{key}: {val}\n'
    text += '</pre>'

    return tag, text


def create_folder_structure(root, run_name, data_path):
    paths = {'data': data_path}

    paths['run'] = root / run_name
    if not os.path.exists(paths['run']):
        os.makedirs(paths['run'])
        
    paths['code'] = paths['run'] / 'code'
    if not os.path.exists(paths['code']):
        os.makedirs(paths['code'])

    paths['ckpt'] = paths['run'] / 'ckpt'
    if not os.path.exists(paths['ckpt']):
        os.makedirs(paths['ckpt'])

    paths['config'] = paths['run'] / 'config'
    if not os.path.exists(paths['config']):
        os.makedirs(paths['config'])

    paths['tb'] = paths['run'] / 'tb'
    if not os.path.exists(paths['tb']):
        os.makedirs(paths['tb'])

    paths['results'] = paths['run'] / 'results'
    if not os.path.exists(paths['results']):
        os.makedirs(paths['results'])

    paths['samples'] = paths['results'] / 'samples'
    if not os.path.exists(paths['samples']):
        os.makedirs(paths['samples'])
        
    paths['samples_del'] = paths['results'] / 'samples_del'
    if not os.path.exists(paths['samples_del']):
        os.makedirs(paths['samples_del'])

    paths['performance'] = paths['results'] / 'performance'
    if not os.path.exists(paths['performance']):
        os.makedirs(paths['performance'])
    
    paths['bo'] = paths['results'] / 'bo'
    if not os.path.exists(paths['bo']):
        os.makedirs(paths['bo'])

    return paths


class Config:
    FILENAME = 'config.pkl'
    JSON_FILENAME = 'params.json'

    @classmethod
    def load(cls, run_dir, **opts):
        path = Path(run_dir) / 'config' / cls.FILENAME
        config = load_pickle(path)
        config.update(**opts)
        return config

    def __init__(self, dataset, **opts):
        run_dir, host_name, start_time = get_run_info(dataset)
        data_path = DATA_DIR / dataset / 'PROCESSED'
        params = DEFAULTS.copy()
        params.update({
            'dataset': dataset,
            'data_path': data_path.as_posix(),
            'run_dir': run_dir,
            'host_name': host_name,
            'start_time': start_time
        })
        paths = create_folder_structure(RUNS_DIR, run_dir, data_path)

        for opt in opts:
            if opt not in params:
                continue
            params[opt] = opts[opt]

        _ = set_random_seed(params['random_seed'])

        self._PARAMS = params
        self._PATHS = paths

        self.copy_code()
        self.save()

    def get(self, attr):
        if attr in self._PARAMS:
            return self._PARAMS[attr]
        raise ValueError(f'{self} does not contain attribute {attr}.')

    def set(self, attr, value):
        if attr in self._PARAMS:
            self._PARAMS[attr] = value
        else:
            raise ValueError(f'{self} does not contain attribute {attr}.')

    def params(self):
        return self._PARAMS

    def path(self, name):
        return self._PATHS[name]

    def save(self):
        # commit if you can
        try:
            commit_hash = commit(self.get('title'), self.get('start_time'))
        except Exception:
            commit_hash = "<automatic commit disabled>"
        self._PARAMS['commit_hash'] = commit_hash

        path = self.path('config') / self.JSON_FILENAME
        save_json(self.params(), path)

        path = self.path('config') / self.FILENAME
        save_pickle(self, path)
        
    def copy_code(self):
        shutil.copytree(PROJ_DIR / 'learner', self._PATHS['code'] / 'learner')
        shutil.copytree(PROJ_DIR / 'molecules', self._PATHS['code'] / 'molecules')
        shutil.copytree(PROJ_DIR / 'scripts', self._PATHS['code'] / 'scripts')
        shutil.copytree(PROJ_DIR / 'utils', self._PATHS['code'] / 'utils')
        shutil.copy(PROJ_DIR / 'manage.py', self._PATHS['code'])
        print('Current code is copied to ' + str(self._PATHS['code']))


    def update(self, **params):
        for param in params:
            #if param not in self._PARAMS:
            #    continue
            self._PARAMS[param] = params[param]

    def write_summary(self, writer):
        tag, text = get_text_summary(self.params())
        writer.add_text(tag, text, 0)

    def __repr__(self):
        return str(self._PARAMS)

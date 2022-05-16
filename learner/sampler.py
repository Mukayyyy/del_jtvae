import time
import numpy as np
import pandas as pd
import itertools
from datetime import datetime

import torch
#from torch.autograd import Variable
from torch.nn import functional as F

from fast_jtnn import *
from molecules.conversion import (
    mols_from_smiles, mols_to_smiles, mol_to_smiles)
from molecules.fragmentation import reconstruct

from utils.config import set_random_seed


def remove_consecutive(fragments):
    return [i for i, _ in itertools.groupby(fragments)]


def generate_molecules(samples, vocab):
    """
    Convert generated samples from indices to molecules and fragments.
    """
    result = []
    num_samples = samples.shape[0]
    valid = np.zeros(num_samples, dtype=bool)

    for idx in range(num_samples):
        frag_smiles = vocab.translate(samples[idx, :])
        frag_smiles = remove_consecutive(frag_smiles)

        if len(frag_smiles) <= 1:
            continue

        try:
            frag_mols = mols_from_smiles(frag_smiles)
            mol, frags = reconstruct(frag_mols)

            if mol is not None:
                smiles = mol_to_smiles(mol)
                num_frags = len(frags)
                frags = " ".join(mols_to_smiles(frags))
                result.append((smiles, frags, num_frags))
                valid[idx] = True # mark is valid
        except Exception:
            continue

    return result, valid


def dump_samples(config, samples, prefix='sampled', postfix='', folder_name='samples'):
    """
    Save generated samples into a CSV file.
    """
    columns = ["smiles", "fragments", "n_fragments"]
    df = pd.DataFrame(samples, columns=columns)
    date = datetime.now().strftime('%Y-%m-%d@%X')
    if postfix == '':
        postfix = date
    #filename = config.path(folder_name) / f"{prefix}{date}_{len(samples)}.csv"
    filename = config.path(folder_name) / f"{prefix}_{postfix}.csv"
    df.to_csv(filename)


# vocab is defined in skipgram.py

class Sampler:
    def __init__(self, config, vocab, model):
        self.config = config
        self.vocab = vocab
        self.model = model

    def sample(self, num_samples, save_results=True, seed=None, postfix='', folder_name='samples'):
        # self.model = self.model.cpu()
        # self.model.eval()
        self.model = self.model.cuda()
        vocab = self.vocab

        hidden_layers = 1
        hidden_size = self.model.hidden_size

        def row_filter(row):
            return (row == vocab.EOS).any()
        
        count = 0
        total_time = 0
        batch_size = 100
        samples, sampled = [], 0 # samples is a list
        valids = []

        max_length = self.config.get('max_length')
        temperature = self.config.get('temperature')

        seed = set_random_seed(seed)
        self.config.set('sampling_seed', seed)
        print("Sampling seed:", self.config.get('sampling_seed'))

        with torch.no_grad():
            while len(samples) < num_samples:
                start = time.time()

                # sample vector from latent space
                # z = self.model.sample_normal(batch_size, use_gpu=False)

                # # get the initial state
                # state = self.model.latent2rnn(z)
                
                # #state = state.view(hidden_layers, batch_size, hidden_size) # may be not necessary
                
                # #state = torch.tanh(state) # I added this
                # state = state.view(batch_size, hidden_layers, hidden_size) # batch x num_layers x hidden_size
                # state = state.transpose(1,0) # now state is num_layer x batch x hidden_state
                # state= state.contiguous()

                # # all idx of batch
                # sequence_idx = torch.arange(0, batch_size).long()

                # # all idx of batch which are still generating
                # running = torch.arange(0, batch_size).long() 
                # sequence_mask = torch.ones(batch_size, dtype=torch.bool) # 1d boolean tensor of length bath_size, with True values

                # # idx of still generating sequences
                # # with respect to current loop
                # running_seqs = torch.arange(0, batch_size).long() # 0, 1, 2, ..., batch_size
                # lengths = [1] * batch_size

                # generated = torch.Tensor(batch_size, max_length).long() # batch x L, to save all generated indices at each time step
                # generated.fill_(vocab.PAD) # 0s

                # inputs = torch.Tensor(batch_size).long() # inputs is a 1d tensor of size batch_size with SOS
                # inputs.fill_(vocab.SOS).long() # 1s
                
                # step = 0

                # while(step < max_length and len(running_seqs) > 0):
                #     # inputs = inputs.unsqueeze(1) # shape (batch_size,1)
                #     # emb = self.model.embedder(inputs)
                #     # scores, state = self.model.decoder(emb, state, lengths)
                #     # scores = scores.squeeze(1) # scores (batch_size, 1, output_size) -> (batch_size, output_size)

                #     probs = F.softmax(scores / temperature, dim=1) # change scores to probabilities, batch_size x output_size
                #     inputs = torch.argmax(probs, 1).reshape(1, -1) # prepare for the next inputs, reshape from batch_size to 1 x batch_size
                #     #??? the reshpa above is odd, will check in run time

                #     # save next input
                #     # add the newly generated indices (1 x batch_size) to the step-th column of generated, rows in running only updated
                #     generated = self.update(generated, inputs, running, step) # generated: batch_size x L, inputs: 1 x batch, step: time step
                #     # generate tensor is like (suppose L, max_length, is 5)
                #     #[[9,8,4,0,0] # sample onoging,   PAD=0,SOS=1,EOS=2
                #     # [6,10,2,0,0] # SOS=2, stopped sample
                #     # [9,3,7,0,0]] # sample stopped, reached full length
                                       
                #     # update global running sequence
                #     sequence_mask[running] = (inputs != vocab.EOS) # if sequence_mask[b]=True, the b-th batch is still running
                #     running = sequence_idx.masked_select(sequence_mask) # get only running indices, such [2,4,7]

                #     # update local running sequences
                #     running_mask = (inputs != vocab.EOS)
                #     running_seqs = running_seqs.masked_select(running_mask)

                #     # prune input and hidden state according to local update
                #     run_length = len(running_seqs)
                #     if run_length > 0:
                #         inputs = inputs.squeeze(0) # input now is shape batch_size
                #         inputs = inputs[running_seqs] # input now is shape running_seqs <= batch_size
                #         state = state[:, running_seqs] # get states for running samples
                #         running_seqs = torch.arange(0, run_length).long() # running_seq now is 0,1,..., run_length
                #         # difference between running and running_seqs:
                #         # running contains indices from original batch, such as [2,4,7]
                #         # running_seq contains sequential numbers, such as [0,1,2]

                #     lengths = [1] * run_length # lengths for decoder is 1, because it generate one by one
                #     step += 1
                    
                # # after generation
                # new_samples = generated.numpy() # 2d array of bacth_size x L 
                # # print(new_samples)
                # mask = np.apply_along_axis(row_filter, 1, new_samples) # apply row_filter for each row
                # # mask is like [False,True,False,True] where True means the corresponding row is a valid sample which has an end.
                # # samples is a list of tuples, each element is a (smiles, frags, num_frags)
                # samples_batch,valid_batch = generate_molecules(new_samples[mask], vocab) # add valid samples to list
                # samples += samples_batch
                # valids += list(valid_batch)

                # generete samples using JtVAE sampler
                sample = self.model.sample_prior() # generate on at a time
                # print('generated sample:', sample)
                # samples += sample
                samples.append(sample)
                valid = self.generate_valid(sample)
                valids.append(valid)

                end = time.time() - start
                total_time += end
                
                # if new samples are added, set count to 0, otherwise, increase count, if tried many times but failed to grow, give up.
                if len(samples) > sampled:
                    sampled = len(samples)
                    count = 0 
                else: 
                    count += 1

                if len(samples) % 1000 < 50:
                    elapsed = time.strftime("%H:%M:%S", time.gmtime(end))
                    print(f'Sampled {len(samples)} molecules. '
                          f'Time elapsed: {elapsed}')

                if count >= 10000:
                    break 

        if save_results:
            dump_samples(self.config, samples, prefix='generated_from_random', postfix=postfix, folder_name=folder_name )

        elapsed = time.strftime("%H:%M:%S", time.gmtime(total_time))
        print(f'Done. Total time elapsed: {elapsed}.')

        set_random_seed(self.config.get('random_seed'))
        # print('sample list:', samples)
        # valids 暂时不用 返回空list
        return samples, valids
    
    
    def update(self, save_to, sample, running_seqs, step):
        # select only still running
        running_latest = save_to[running_seqs]
        # update token at step position
        running_latest[:, step] = sample.data
        # save back
        save_to[running_seqs] = running_latest

        return save_to

    def generate_valid(self, sample):
        valid = 0
        if sample is not None:
            valid = True

        return valid
    
    
    # def encode_z(self, inputs, lengths):
    #     with torch.no_grad():
    #         embeddings = self.model.embedder(inputs)
    #         z, mu, logvar = self.model.encoder(inputs, embeddings, lengths)
    #     return z, mu, logvar
    
    def sample_from_z(self, z, save_results=True, seed=None):
        """
        Given latent representation z, generate samples.
        z: tensor, batch (or num_samples) x latent_size.
        """
        self.model = self.model.cuda()
        self.model.eval()
        vocab = self.vocab

        hidden_layers = 1
        hidden_size = self.model.hidden_size

        def row_filter(row):
            return (row == vocab.EOS).any()
        
        total_time = 0
        samples = [] # samples is a list

        max_length = self.config.get('max_length')
        temperature = self.config.get('temperature')
        
        batch_size = z.shape[0] # number of samples to be generated

        seed = set_random_seed()
        self.config.set('sampling_seed', seed)
        print("Sampling seed:", self.config.get('sampling_seed'))

        with torch.no_grad():
            #while len(samples) < num_samples:
            start = time.time()

            # generete samples using JtVAE sampler
            sample = self.model.sample_prior_z(z) # generate on at a time

        #     # sample vector from latent space
        #     #z = self.model.encoder.sample_normal(batch_size)

        #     # get the initial state
        #     state = self.model.latent2rnn(z)
            
        #     #state = state.view(hidden_layers, batch_size, hidden_size) # may be not necessary
            
        #     #state = torch.tanh(state) # I added this
        #     state = state.view(batch_size, hidden_layers, hidden_size) # batch x num_layers x hidden_size
        #     state = state.transpose(1,0) # now state is num_layer x batch x hidden_state
        #     state= state.contiguous()

        #     # all idx of batch
        #     sequence_idx = torch.arange(0, batch_size).long()

        #     # all idx of batch which are still generating
        #     running = torch.arange(0, batch_size).long() 
        #     sequence_mask = torch.ones(batch_size, dtype=torch.bool) # 1d boolean tensor of length bath_size, with True values

        #     # idx of still generating sequences
        #     # with respect to current loop
        #     running_seqs = torch.arange(0, batch_size).long() # 0, 1, 2, ..., batch_size, the size of this vector will shrink along the progress
        #     lengths = [1] * batch_size

        #     generated = torch.Tensor(batch_size, max_length).long() # batch x L, to save all generated indices at each time step
        #     generated.fill_(vocab.PAD) # 0s

        #     inputs = torch.Tensor(batch_size).long() # inputs is a 1d tensor of size batch_size with SOS
        #     inputs.fill_(vocab.SOS).long() # 1s
            
        #     step = 0

        #     while(step < max_length and len(running_seqs) > 0):
        #         inputs = inputs.unsqueeze(1) # shape (<=batch_size,1)
        #         emb = self.model.embedder(inputs)
        #         scores, state = self.model.decoder(emb, state, lengths)
        #         scores = scores.squeeze(1) # scores (batch_size, 1, output_size) -> (batch_size, output_size)

        #         probs = F.softmax(scores / temperature, dim=1) # change scores to probabilities, batch_size x output_size
        #         inputs = torch.argmax(probs, 1).reshape(1, -1) # prepare for the next inputs, reshape from batch_size to 1 x batch_size
        #         #??? the reshape above is odd, will check in run time
        #         # I think it is not necessary to reshape inputs as either way is good for the update() function below.

        #         # save next input
        #         # add the newly generated indices (1 x batch_size) to the step-th column of generated, rows in running only updated
        #         generated = self.update(generated, inputs, running, step) # generated: batch_size x L, inputs: 1 x batch, step: time step
        #         # generate tensor is like (suppose L, max_length, is 5)
        #         #[[9,8,4,0,0] # sample onoging,   PAD=0,SOS=1,EOS=2
        #         # [6,10,2,0,0] # SOS=2, stopped sample
        #         # [9,3,7,0,0]] # sample stopped, reached full length
                                   
        #         # update global running sequence
        #         sequence_mask[running] = (inputs != vocab.EOS) # if sequence_mask[b]=True, the b-th batch is still running
        #         running = sequence_idx.masked_select(sequence_mask) # update running, get only running indices, such [2,4,7]
                
        #         # update local running sequences
        #         running_mask = (inputs != vocab.EOS)
        #         running_seqs = running_seqs.masked_select(running_mask) # indices of active samples
                
        #         # prune input and hidden state according to local update
        #         run_length = len(running_seqs)
        #         if run_length > 0:
        #             inputs = inputs.squeeze(0) # input now is shape batch_size
        #             inputs = inputs[running_seqs] # input now is shape running_seqs <= batch_size
        #             state = state[:, running_seqs] # get states for running samples
        #             running_seqs = torch.arange(0, run_length).long() # running_seqs now is 0,1,..., run_length
        #             # difference between running and running_seqs:
        #             # running contains indices from original batch, such as [2,4,7]
        #             # running_seqs contains sequential numbers, such as [0,1,2]

        #         lengths = [1] * run_length # lengths for decoder is 1, because it generate one by one
        #         step += 1
                
        # # after generation
        # new_samples = generated.numpy() # 2d array of bacth_size x L 
        # # print(new_samples)
        # mask = np.apply_along_axis(row_filter, 1, new_samples) # apply row_filter for each row
        # # mask is like [False,True,False,True] where True means the corresponding row is a valid sample which has an end.
        # # samples is a list of tuples, each element is a (smiles, frags, num_frags)
        # samples, valid = generate_molecules(new_samples[mask], vocab) # add valid samples to list
        # mask[ mask ] = valid

        end = time.time() - start
        total_time += end

        if save_results:
            dump_samples(self.config, samples, prefix='generated_from_z')

        elapsed = time.strftime("%H:%M:%S", time.gmtime(total_time))
        print(f'Done. Total time elapsed: {elapsed}.')

        set_random_seed(self.config.get('random_seed'))

        return sample

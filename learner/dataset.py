import time
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader

from fast_jtnn import JTNNEncoder, MPN, JTMPN, Vocab
# from .fast_jtnn import *
from utils.filesystem import load_dataset
# from .skipgram import Vocab
import pickle
import os, random


class MolTreeFolder(object):

    def __init__(self, config, data_folder, vocab, batch_size, num_workers=4, shuffle=True, assm=True, replicate=None):
        self.config = config
        self.data_folder = data_folder
        #all_data_files = [fn for fn in os.listdir(data_folder)]
        #self.data_files = all_data_files[0: 10]
        self.data_files = [fn for fn in os.listdir(data_folder)]
        self.batch_size = batch_size
        self.vocab = vocab
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.assm = assm
        self.size = 0
        self.smile_list = [x.strip("\r\n ") for x in open(self.config.get('processed_smile'))]

        if replicate is not None: #expand is int
            self.data_files = self.data_files * replicate

    def __iter__(self):
        for fn in self.data_files:
            fn = os.path.join(self.data_folder, fn)
            with open(fn, 'rb') as f:
                data = pickle.load(f)

            if self.shuffle: 
                random.shuffle(data) #shuffle data before batch

            batches = [data[i : i + self.batch_size] for i in range(0, len(data), self.batch_size)]
            if len(batches) < 1: 
                batches = []

            elif len(batches[-1]) < self.batch_size:
                batches.pop()

            dataset = MolTreeDataset(self.config, batches, self.vocab, self.assm)
            self.size += len(batches)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.num_workers, collate_fn=lambda x:x[0])

            for b in dataloader:
                yield b
            del data, batches, dataset, dataloader

    def get_len(self):
        return self.size

    def get_smile_list(self):
        return self.smile_list

class MolTreeDataset(Dataset):

    def __init__(self, config, data = None, vocab = None, assm=True):
        # if kind != 'given':
        #     self.data = load_dataset(config, kind=kind)
        #     #data = data[0:20000]
        # if not data:
        self.data = data
        self.config = config
        self.vocab = vocab
        self.assm = assm
        self.smile_list = [x.strip("\r\n ") for x in open(self.config.get('processed_smile'))]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return tensorize(self.data[idx], self.vocab, assm=self.assm)

    def get_processed_data(self):
        smile_list = [x.strip("\r\n ") for x in open(self.config.get('processed_smile'))]
        return smile_list

    def get_smile_data(self):
        data = load_dataset(self.config, kind='train')
        return data.reset_index(drop=True)

    def get_vocab(self):
        vocab = [x.strip("\r\n ") for x in open(self.config.get('vocab_path'))] 
        vocab = Vocab(vocab)
        return vocab

    def get_smile_list(self):
        return self.smile_list
    
    # 待定
    def change_data(self, data):
        self.data = data
        self.size = self.data.shape[0]

def tensorize(tree_batch, vocab, assm=True):
    set_batch_nodeID(tree_batch, vocab)
    smiles_batch = [tree.smiles for tree in tree_batch]
    jtenc_holder,mess_dict = JTNNEncoder.tensorize(tree_batch)
    jtenc_holder = jtenc_holder
    mpn_holder = MPN.tensorize(smiles_batch)

    if assm is False:
        return tree_batch, jtenc_holder, mpn_holder

    cands = []
    batch_idx = []
    for i,mol_tree in enumerate(tree_batch):
        for node in mol_tree.nodes:
            #Leaf node's attachment is determined by neighboring node's attachment
            if node.is_leaf or len(node.cands) == 1: continue
            cands.extend( [(cand, mol_tree.nodes, node) for cand in node.cands] )
            batch_idx.extend([i] * len(node.cands))

    jtmpn_holder = JTMPN.tensorize(cands, mess_dict)
    batch_idx = torch.LongTensor(batch_idx)

    return tree_batch, jtenc_holder, mpn_holder, (jtmpn_holder,batch_idx)

def set_batch_nodeID(mol_batch, vocab):
    tot = 0
    for mol_tree in mol_batch:
        for node in mol_tree.nodes:
            node.idx = tot
            node.wid = vocab.get_index(node.smiles)
            tot += 1


# DEL
class DataCollator:
    def __init__(self, vocab):
        self.vocab = vocab

    def merge(self, sequences):
        '''
        Sentence to indices, pad with zeros
        '''
        #print('before:')
        #print(sequences)
        #sequences = sorted(sequences, key=len, reverse=True)# YL: choose not to sort it, because I need properties in same order
        #print('after:')
        #print(sequences)
        lengths = [len(seq) for seq in sequences]
        padded_seqs = np.full((len(sequences), max(lengths)), self.vocab.PAD)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return torch.LongTensor(padded_seqs), lengths

    def __call__(self, data):
        # seperate source and target sequences
        # if batch is 3
        #data: [([1, 6127, 2701], [6127, 2701, 2], array([0.8318088 , 2.02817336, 3.34902   ])), 
        #       ([1, 31928, 269, 1036, 44], [31928, 269, 1036, 44, 2], array([0.86020293, 3.05513613, 3.9534    ])), 
        #       ([1, 850, 4212, 769], [850, 4212, 769, 2], array([0.51382926, 1.96542485, 2.44458   ]))]
        
        src_seqs, tgt_seqs, properties = zip(*data) # 
        
        # now src_seqs, tgt_seqs, properties are in separate lists.
        #print('data:')
        #print(data)
        #print('\n------------------------------------------------------------------\n')
        #print('src_seqs:')
        #print(src_seqs)
        #print('tgt_seqs:')
        #print(tgt_seqs)

        # merge sequences (from tuple of 1D tensor to 2D tensor)
        src_seqs, src_lengths = self.merge(src_seqs)
        tgt_seqs, tgt_lengths = self.merge(tgt_seqs)
        properties = torch.tensor(properties, dtype=torch.float)
        return src_seqs, tgt_seqs, src_lengths, properties

class FragmentDataset(Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, config, kind='train', data=None):
        """
        Reads source and target sequences from a csv file or given variable data_given.
        kind: string, from {'train', 'test', 'given'}
        data: pd array, having the following fields: smiles, fragments, n_fragments, C, F, N, O, Other, 
                                                     SINGLE, DOUBLE, TRIPLE, Tri, Quad, Pent, Hex,
                                                     logP, mr, qed, SAS

        """
        self.config = config
        
        if kind != 'given':
            # 这里拿到的data已经是处理过含有各个column的了
            data = load_dataset(config, kind=kind)
            #data = data[0:20000]
        
        # the following is for test data, because there are n_fragments=0 / 1 
        min_nfrag=2
        data = data[data.n_fragments>=min_nfrag]
        
        self.data = data.reset_index(drop=True)
        self.size = self.data.shape[0]

        #print('data:',self.data.SAS)
        
        self.vocab = None

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        seq = self.data.fragments[index].split(" ")
        seq = self.vocab.append_delimiters(seq)
        src = self.vocab.translate(seq[:-1]) # include '<SOS>' but  do not include '<EOS>'
        tgt = self.vocab.translate(seq[1:]) # do not include '<SOS>', but do not include '<EOS>'
        
        properties = np.array([0,0,0], dtype=float)
        properties[0] = self.data.qed[index]
        properties[1] = self.data.SAS[index]
        properties[2] = self.data.logP[index]
        
        return src, tgt, properties

    def __len__(self):
        return self.size

    ### 需要在jtvae dataset里添加的方法
    def get_loader(self):
        start = time.time()
        collator = DataCollator(self.vocab)
        loader = DataLoader(dataset=self,
                            collate_fn=collator,
                            batch_size=self.config.get('batch_size'),
                            num_workers=24,
                            shuffle=True)
        end = time.time() - start
        elapsed = time.strftime("%H:%M:%S", time.gmtime(end))
        print(f'Data loaded. Size: {self.size}. '
              f'Time elapsed: {elapsed}.')
        return loader

    def get_vocab(self):
        start = time.time()
        if self.vocab is None:
            try:
                self.vocab = Vocab.load(self.config)
            except Exception:
                self.vocab = Vocab(self.config, self.data)

        end = time.time() - start
        elapsed = time.strftime("%H:%M:%S", time.gmtime(end))
        print(f'Vocab created/loaded. '
              f'Size: {self.vocab.get_size()}. '
              f'Effective size: {self.vocab.get_effective_size()}. '
              f'Time elapsed: {elapsed}.')

        return self.vocab
    
    def change_data(self, data):
        self.data = data
        self.size = self.data.shape[0]

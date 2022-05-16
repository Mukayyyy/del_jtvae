import time
import numpy as np
import pandas as pd
import math, random, sys

import torch
import torch.nn as nn
#from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from tensorboardX import SummaryWriter

from fast_jtnn import *
from .dataset import MolTreeFolder as MolTreeDataLoader
from .dataset import MolTreeDataset
from .model import Loss, Frag2Mol
from .sampler import Sampler
from utils.filesystem import load_dataset
from utils.postprocess import score_samples


SCORES = ["validity", "novelty", "uniqueness"]


def save_ckpt(trainer, epoch, filename):
    """
    Save checkpoint a certain epoch.
    """
    path = trainer.config.path('ckpt') / filename
    torch.save({
        'epoch': epoch,
        'best_loss': trainer.best_loss,
        'losses': trainer.losses,
        'best_score': trainer.best_score,
        'scores': trainer.scores,
        'model': trainer.model.state_dict(),
        'optimizer': trainer.optimizer.state_dict(),
        'scheduler': trainer.scheduler.state_dict(),
        # 'criterion': trainer.criterion.state_dict()
    }, path)


def load_ckpt(trainer, last=False, pretrained=True):
    """
    Load checkpoint.
    """
    if pretrained:
        filename = 'pretrain.pt'
    else:
        filename = 'last.pt' if last is True else 'best_valid.pt'
    path = trainer.config.path('ckpt') / filename

    if trainer.config.get('use_gpu') is False:
        checkpoint = torch.load(
            path, map_location=lambda storage, loc: storage)
    else:
        checkpoint = torch.load(path)

    print(f"loading {filename} at epoch {checkpoint['epoch']+1}...")

    trainer.model.load_state_dict(checkpoint['model'])
    trainer.optimizer.load_state_dict(checkpoint['optimizer'])
    trainer.scheduler.load_state_dict(checkpoint['scheduler'])
    # trainer.criterion.load_state_dict(checkpoint['criterion'])
    trainer.best_loss = checkpoint['best_loss']
    trainer.losses = checkpoint['losses']
    trainer.best_score = checkpoint['best_score']
    trainer.scores = checkpoint['scores']
    return checkpoint['epoch']


# def get_optimizer(config, model):
#     return Adam(model.parameters(), lr=config.get('optim_lr'))


# def get_scheduler(config, optimizer):
#     return StepLR(optimizer,
#                   step_size=config.get('sched_step_size'),
#                   gamma=config.get('sched_gamma'))


def dump(config, losses, scores, loss_details):
    """
    Save losses and scores to CSV.
    """
    df = pd.DataFrame(losses, columns=["loss"])
    filename = config.path('performance') / "loss.csv"
    df.to_csv(filename)
    
    # YL's add loss, beta, KL_div, Word, Topo, Assm, PNorm, GNorm
    #df = pd.DataFrame(loss_details, columns=['epoch', 'idx', 'loss', 'CE_loss', 'MSE_loss', 'KL_loss', 'alpha', 'beta'])

    # 保存loss 和其他参数

    # filename = config.path('performance') / "loss_details.csv"
    # df.to_csv(filename)
    df = pd.DataFrame(loss_details, columns=['epoch', 'idx', 'totalStep', 'loss', 'beta', 'KL', 'Word', 'Topo', 'Assm', 'PNorm', 'GNorm'])
    filename = config.path('performance') / "loss_details.csv"
    df.to_csv(filename)
    
    if scores != []:
        df = pd.DataFrame(scores, columns=SCORES)
        filename = config.path('performance') / "scores.csv"
        df.to_csv(filename)


class TBLogger:
    def __init__(self, config):
        self.config = config
        self.writer = SummaryWriter(config.path('tb').as_posix())
        config.write_summary(self.writer)

    def log(self, name, value, epoch):
        self.writer.add_scalar(name, value, epoch)


class Trainer:
    @classmethod
    def load(cls, config, vocab, last):
        trainer = Trainer(config, vocab)
        epoch = load_ckpt(trainer, last=last)
        return trainer, epoch

    def __init__(self, config, vocab):
        self.config = config
        self.vocab = vocab
        # vocab = [x.strip("\r\n ") for x in open(args.vocab)] 
        # vocab = Vocab(vocab)

        self.model = JTNNVAE(self.vocab, self.config)
        print("JtVAE model", self.model)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.get('optim_lr')) #lr in jtvae args
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, config.get('anneal_rate'))
        self.scheduler.step()
        # self.criterion = Loss(config, pad=vocab.PAD)  # loss直接由model输出

        if self.config.get('use_gpu'):
            self.model = self.model.cuda()

        self.losses = []
        self.best_loss = np.float('inf')
        self.scores = []
        self.best_score = - np.float('inf')
        self.loss_details=None


    def _valid_epoch(self, epoch, loader):
        use_gpu = self.config.get('use_gpu')
        self.config.set('use_gpu', False)

        num_samples = self.config.get('validation_samples')
        trainer, _ = Trainer.load(self.config, self.vocab, last=True)
        sampler = Sampler(self.config, self.vocab, trainer.model)
        samples = sampler.sample(num_samples, save_results=False)
        dataset = load_dataset(self.config, kind="test") # load test dataset
        _, scores = score_samples(samples, dataset)

        self.config.set('use_gpu', use_gpu)
        return scores

    def log_epoch(self, start_time, epoch, epoch_loss, epoch_scores):
        end = time.time() - start_time
        elapsed = time.strftime("%H:%M:%S", time.gmtime(end))

        print(f'epoch {epoch:06d} - '
              f'loss {epoch_loss:6.4f} - ',
              end=' ')

        if epoch_scores is not None:
            for (name, score) in zip(SCORES, epoch_scores):
                print(f'{name} {score:6.4f} - ', end='')

        print(f'elapsed {elapsed}')

    def train(self, start_epoch):

        ### jtvae
        total_step = self.config.get('load_epoch')
        beta = self.config.get('a_beta')
        meters = np.zeros(4)


        for param in self.model.parameters():
            if param.dim() == 1:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_normal_(param)

        if total_step > 0:
            self.model.load_state_dict(torch.load(self.config.get('save_dir') + "/model.iter-" + str(total_step))) # model内置函数

        print("Model #Params: %dK" % (sum([x.nelement() for x in self.model.parameters()]) / 1000,))

        param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
        grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))
        
        self.scheduler.step()

        num_epochs = self.config.get('num_epochs')

        logger = TBLogger(self.config)

        if start_epoch>0:
            if self.config.get('use_gpu'):
                self.model = self.model.cuda()
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.get('optim_lr')) #lr in jtvae arg
            self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, self.config.get('anneal_rate'))

        for epoch in range(start_epoch, start_epoch + num_epochs):

            ### DataLoader  => FragmentDataset.get_loader()
            idx = 0
            start = time.time()

            loader = MolTreeDataLoader(self.config, self.config.get('train_path'), self.vocab, self.config.get('batch_size'))
            # d = MolTreeDataset(self.config)
            epoch_loss = 0
            # smile_list = [x.strip("\r\nsmile_list ") for x in open(self.config.get('processed_smile'))]
            epoch_loss_details = np.zeros( (len(loader.get_smile_list()), 11) ) #epoch, idx, totalSteps, loss, beta, KL_div, Word, Topo, Assm, PNorm, GNorm
            # epoch_loss_details = np.zeros((12000, 11))

            for batch in loader:
                idx += 1
                total_step += 1
                try:
                    self.model.zero_grad()
                    loss, kl_div, wacc, tacc, sacc, z, mu, logvar, predicted_prop = self.model(batch, beta)

                    if self.config.get('use_gpu'):
                        # properties = properties.cuda()
                        predicted_prop = predicted_prop.cuda()
                    
                    loss.backward()
                    epoch_loss += loss.item()
                    epoch_loss_details[total_step,:]= [total_step, epoch, idx, loss.item(), beta, kl_div, wacc * 100, tacc * 100, sacc * 100, param_norm(self.model), grad_norm(self.model)]
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('clip_norm'))
                    self.optimizer.step()
                except Exception as e:
                    print(e)
                    continue

                meters = meters + np.array([kl_div, wacc * 100, tacc * 100, sacc * 100])

                # 一系列需要写进参数config文件的值
                if total_step % self.config.get('print_iter') == 0:
                    meters /= self.config.get('print_iter')
                    print("[%d] Loss: %.3f, Beta: %.3f, KL: %.2f, Word: %.2f, Topo: %.2f, Assm: %.2f, PNorm: %.2f, GNorm: %.2f" % (total_step, loss.item(), beta, meters[0], meters[1], meters[2], meters[3], param_norm(self.model), grad_norm(self.model)))
                    sys.stdout.flush()
                    meters *= 0

                if total_step % self.config.get('save_iter') == 0:
                    # 可能会存两份
                    torch.save(self.model.state_dict(), self.config.get('save_dir') + "/model.iter-" + str(total_step))

                if total_step % self.config.get('anneal_iter') == 0:
                    self.scheduler.step()
                    print("learning rate: %.6f" % self.scheduler.get_lr()[0])

                # if total_step % self.config.get('kl_anneal_iter') == 0 and total_step >= self.config.get('warmup'):
                    # beta = min(self.config.get('max_beta'), beta + self.config.get('step_beta'))
        
            # start = time.time()
            epoch_loss = epoch_loss/idx
            self.losses.append(epoch_loss)

            # 暂且取消loss detail, 因为jtvae只有一个loss值

            #save losses, alpha, and beta values
            if epoch == 0:
                self.loss_details = epoch_loss_details
            else:
                self.loss_details = np.vstack( (self.loss_details,epoch_loss_details) )
            
            logger.log('loss', epoch_loss, epoch)
            save_ckpt(self, epoch, filename="last.pt")
            
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                save_ckpt(self, epoch, filename=f'best_loss.pt')

            epoch_scores = None

            if epoch_loss < self.config.get('validate_after'):
                epoch_scores = self._valid_epoch(epoch, loader)
                self.scores.append(epoch_scores)

                if epoch_scores[2] >= self.best_score:
                    self.best_score = epoch_scores[2]
                    save_ckpt(self, epoch, filename=f'best_valid.pt')

                logger.log('validity', epoch_scores[0], epoch)
                logger.log('novelty', epoch_scores[1], epoch)
                logger.log('uniqueness', epoch_scores[2], epoch)

            self.log_epoch(start, epoch, epoch_loss, epoch_scores)
    
        dump(self.config, self.losses, self.scores, self.loss_details)

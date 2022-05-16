from learner.dataset import FragmentDataset
from learner.sampler import Sampler
from learner.trainer import Trainer, save_ckpt
from utils.config import Config
from utils.parser import command_parser
from utils.plots import plot_paper_figures, plot_figures, plot_loss, plot_pareto_fronts
from utils.preprocess import preprocess_dataset
from utils.postprocess import postprocess_samples, postprocess_samples_del, postprocess_data_del, score_samples, dump_scores 
from utils.filesystem import load_dataset

from learner.DEL import DEL
from learner.DEL_SO import DEL_single
from learner.BO import BO

from rdkit import rdBase
rdBase.DisableLog('rdApp.*')


def train_model(config):
    dataset = FragmentDataset(config)
    vocab = dataset.get_vocab()
    trainer = Trainer(config, vocab)
    trainer.train(dataset.get_loader(), 0)


# can use this function in model evolution
def resume_model(config):
    dataset = FragmentDataset(config)
    vocab = dataset.get_vocab()
    load_last = config.get('load_last')
    trainer, epoch = Trainer.load(config, vocab, last=load_last)
    trainer.train(dataset.get_loader(), epoch + 1)


def sample_model(config):
    dataset = FragmentDataset(config)
    vocab = dataset.get_vocab()
    load_last = config.get('load_last')
    trainer, epoch = Trainer.load(config, vocab, last=load_last)
    sampler = Sampler(config, vocab, trainer.model)
    seed = config.get('sampling_seed') if config.get('reproduce') else None
    samples = sampler.sample(config.get('num_samples'), seed=seed)
    dataset = load_dataset(config, kind="test")
    _, scores = score_samples(samples, dataset)
    is_max = dump_scores(config, scores, epoch)
    if is_max:
        save_ckpt(trainer, epoch, filename=f"best.pt")
    config.save()


def sample_from_z_model(config):
    """
    Reconstruction.
    """
    dataset = FragmentDataset(config, kind='test')
    vocab = dataset.get_vocab()
    load_last = config.get('load_last') # boolean to indicate whether load last
    trainer, epoch = Trainer.load(config, vocab, last=load_last)
    sampler = Sampler(config, vocab, trainer.model)
    seed = config.get('sampling_seed') if config.get('reproduce') else None
    
    loader=dataset.get_loader() # load data
    for idx, (src, tgt, lengths, properties) in enumerate(loader):
        z, mu, logvar = sampler.encode_z(src,lengths)
        samples = sampler.sample_from_z(z, seed=seed)
    
    # concatenate samples
    pass

    dataset = load_dataset(config, kind="test")
    _, scores = score_samples(samples, dataset)
    is_max = dump_scores(config, scores, epoch)
    if is_max:
        save_ckpt(trainer, epoch, filename=f"best.pt")
    config.save()
    

if __name__ == "__main__":
    parser = command_parser() # defined in parser.py
    args = vars(parser.parse_args())
    command = args.pop('command')

    if command == 'preprocess':
        dataset = args.pop('dataset')
        n_jobs = args.pop('n_jobs')
        preprocess_dataset(dataset, n_jobs)

    elif command == 'train':
        config = Config(args.pop('dataset'), **args)
        train_model(config)

    elif command == 'resume':
        run_dir = args.pop('run_dir')
        config = Config.load(run_dir, **args)
        resume_model(config)

    elif command == 'sample':
        args.update(use_gpu=False)
        run_dir = args.pop('run_dir')
        config = Config.load(run_dir, **args)
        sample_model(config)

    elif command == 'postprocess':
        run_dir = args.pop('run_dir')
        config = Config.load(run_dir, **args)
        postprocess_samples(config, **args)
    
    elif command == 'plot':
        run_dir = args.pop('run_dir')
        plot_paper_figures(run_dir)
        plot_loss(run_dir, group='batch')
        plot_loss(run_dir, group='epoch')
        
    elif command == 'DEL':
        config = Config(args.pop('dataset'), **args)
        print(config)
        if not config.get('single_objective'):
            del1 = DEL(config)
        else:
            del1 = DEL_single(config)
        del1.train()
        #del1.save_population(config.get('num_generations'))
        
        
    elif command == 'plot_del':
        run_dir = args.pop('run_dir')
        config = Config.load(run_dir, **args)
        postprocess_data_del(config, use_train=False, prefix='new_pop', postfixes=['0','4','final'])
        plot_figures(run_dir, DEL=True, filename='new_pop_aggregated')
        postprocess_data_del(config, use_train=False, prefix='generated_from_random', postfixes=['0','4','final'])
        plot_figures(run_dir, DEL=True, filename='generated_from_random_aggregated')
        plot_loss(run_dir, group='batch')
        plot_loss(run_dir, group='epoch')
        plot_pareto_fronts(run_dir, fronts=[0,1,2,3,4], with_bo=False)
        plot_pareto_fronts(run_dir, fronts=[0,1,2,3,4], with_bo=True)
        
        
    elif command == 'BO':
        run_dir = args.pop('run_dir')
        config = Config.load(run_dir, **args)
        bo1 = BO(config)
        bo1.train()

        
        

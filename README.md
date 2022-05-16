This README file contains only old info about the original FragVAE.

# Codes for JTVAE are from the paper "Junction Tree Variational Autoencoder for Molecular Graph Generation" (Wengong Jin)

Link to the paper: [arXiv](https://arxiv.org/abs/1802.04364)

### Installation

Run:

`source scripts/install.sh`

This will take care of installing all required dependencies.
If you have trouble during the installation, try running each line of the `scripts/install.sh` file separately (one by one) in your shell.

The only required dependency is the latest Conda package manager, which you can download with the Anaconda Python distribution [here](https://www.anaconda.com/distribution/).

After that, you are all set up.


### Preprocessing for JTVAE

#### 1. Deriving Vocabulary 
To perform tree decomposition over a set of molecules, run
```
python ../fast_jtnn/mol_tree.py < ../DATA/ZINC/PROCESSED/train.txt
```
This gives you the vocabulary of cluster labels over the dataset `train.txt`. 

#### 2. Preprocess the data
```
python ../fast_jtnn/preprocess.py --train ../DATA/ZINC/PROCESSED/train.txt --split 100 --jobs 16
cd fast_jtnn
mkdir moses-processed
mv tensor* moses-processed
```
This script will preprocess the training data (subgraph enumeration & tree decomposition), and save results into a list of files. We suggest you to use small value for `--split` if you are working with smaller datasets.


### Preprocessing

First, you need to download the data and do some preprocessing. To do this, run:

`python manage.py preprocess --dataset <DATASET_NAME>`

where `<DATASET_NAME>` must be `ZINC` or `PCBA`. At the moment, we support only these two.

Use `python manage.py preprocess --help` to see other useful options for preprocessing.

This will download the necessary files in the `DATA` folder, and will preprocess them as described in the paper.




### Training

After preprocessing, you can train the model running:

`python manage.py train --dataset <DATASET_NAME>`

where `<DATASET_NAME>` is defined as described above.

If you wish to train using a GPU, add the `--use_gpu` option.

Check out `python manage.py train --help` to see all the other hyperparameters you can change.

Training the model will create folder `RUNS` with the following structure:

```
RUNS
└── <date>@<time>-<hostname>-<dataset>
    ├── ckpt
    │   ├── best_loss.pt
    │   ├── best_valid.pt
    │   └── last.pt
    ├── config
    │   ├── config.pkl
    │   ├── emb_<embedding_dim>.dat
    │   ├── params.json
    │   └── vocab.pkl
    ├── results
    │   ├── performance
    │   │   ├── loss.csv
    │   │   └── scores.csv
    │   └── samples
    └── tb
        └── events.out.tfevents.<tensorboard_id>.<hostname>
```


the `<date>@<time>-<hostname>-<dataset>` folder is a snapshot of your experiment, which will contain all the data collected during training.

You can monitor the progress of training using tensorboardX, just run

`tensorboard --logdir RUNS`

during training and check the `localhost:6006` page in your favorite browser.


### Sampling

After the model is trained, you can sample from it using

`python manage.py sample --run <RUN_PATH>`

where `<RUN_PATH>` is the path to the run directory of the experiment, which will be something like `RUNS/<date>@<time>-<hostname>-<dataset>` (`<date>`, `<time>`, `<hostname>`, `<dataset>` are placeholders of the actual data).

Check out `python manage.py sample --help` to see all the sampling options.

You will find your samples in the `results/samples` folder on your experiment run directory.


### Postprocessing

After you have sampled the model, you wish to conduct some common postprocessing operations such as calculate statistics on the samples, aggregate multiple sample files and the test data in one big file for plotting, etc.

Then, you need to run:

`python manage.py postprocess --run <RUN_PATH>`

where `<RUN_PATH>` is obtained as described above.

Check out `python manage.py postprocess --help` to see all available options.


### Plotting

If you wish to obtain similar figures as the ones in the paper on your samples, just run:

`python manage.py plot --run <RUN_PATH>`

where `<RUN_PATH>` is defined as described above.

### Samples

You can find the 20k SMILES samples used in the paper for the analysis in the SAMPLES folder.

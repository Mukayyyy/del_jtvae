### Source code for the paper "Multi-Objective Drug Design Based on Graph-Fragment Molecular Representation and Deep Evolutionary Learning"

Muhetaer Mukaidaisi, and Yifeng Li

An integration of the graph-fragment based generative model JTVAE and the DEL framework

### JTVAE model is from the paper "Junction Tree Variational Autoencoder for Molecular Graph Generation"  by Wengong Jin

Link to the paper: [arXiv](https://arxiv.org/abs/1802.04364)

### DEL framework is proposed in the paper "Deep Evolutionary Learning for Molecular Design"

Link to the paper: [IEEE CIM](https://ieeexplore.ieee.org/document/9756593)

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

### Running 

After preprocessing, you may run the framework altogether (training + samplling) by running the default script:

`bash runscript5_3.sh`

where runscript5_3.sh contains the 

### Training

You can also train the model solely running:

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

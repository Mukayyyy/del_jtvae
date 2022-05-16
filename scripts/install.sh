# create venv with all required dependencies
#conda create --name aistats2020 -y
conda create --name fregment -y

# activate venv
#conda activate aistats2020
conda activate fregment

# install pip
conda install pip -y

conda install python=3.7 -y

# install conda packages
conda install scipy pandas gensim joblib sh matplotlib seaborn scikit-learn -y

# install pip packages
pip install tensorboardX

# install pytorch
conda install pytorch -c pytorch -y

# install rdkit
conda install rdkit -c rdkit -y

# install qvina dependencies
conda install -c conda-forge qvina openbabel numpy psutil

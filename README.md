# K-Mer Fingerprinting with RNN to predict MICs for K. Pneumoniae
![Process flow and RNN architecture](architecture.drawio.png?raw=true "Process flow and RNN architecture")

# Setup
1. Make sure that you have the requiset NVIDIA software installed for CUDANN 11.0. 

2. Then, you can set up the conda environment using 
```
conda create --name kmer_fingerprint_rnn --file conda_requirements.txt
conda activate kmer_fingerprint_rnn
```

3. Finally, you should install pip requirements with 
```
pip install -r pip_requirements.txt
```

# Classify
Before you can classify anything, make sure you follow the instructions in the setup section. If you would like to pass in a Fasta file to have all drug models predict MICs for it, cd into the script directory and run `python classify.py [path to FASTA file]` (ex: `python classify.py ../data/isolate_1.fasta`). This will output tab delimited values to the console in the format:
```
drug    softmax output  predicted MIC
Meropenem       [3.9857876e-04 5.5150859e-02 2.4652623e-01 1.5810619e-01 6.7635857e-02
 2.5493633e-02 3.7189886e-02 4.9746264e-02 4.7098272e-02 4.6077225e-02
 7.2678879e-02 5.2639838e-02 6.3260801e-02 7.6727271e-02 1.1610194e-04
 1.3132016e-04 1.1498636e-04 4.1232881e-04 4.9548986e-04]       0.03
```

# Training

## Processing data
When you train on a new dataset, the FASTAS should be placed in `data/fastas`. The first time training is run the code will process and split the FASTA files into training/testing datasets and put the processed fingerprints in respective CSVs in `data/fastas`. If you would like to change this location, change `DATA_DIR` in `script/train.py`.

## No Seed set
A note on training. You may get slightly different results than we do as there is not a set seed set up. However, the results should be within a margin of error of what is shown. If you would like to tune the hyperparameters to your data, you should delete `./script/untitled_project/`. This will delete all tuning progress, and the next time training is done, tuning will be performed.

## How to train
If you would like to train new models, make sure you follow the setup section before continuing. It is expected that you can go inside the training file to manually set parameters (ex: whether to train or not) rather than pass them in as arguments. This is done so the user can understand the code before training as it is not complex.

To train, cd into the script directory and run `python train.py`.

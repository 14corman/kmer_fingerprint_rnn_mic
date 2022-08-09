"""
This file contains all functions necessary to convert a Fasta
file of contigs into an embeded vector representation.

Base for code was taken from the following files on Github:
- https://github.com/lelugom/wgs_classifier/blob/master/wgs_classifier.py
- https://github.com/lelugom/wgs_classifier/blob/master/wgs_dataset.py

@author Cory Kromer-Edwards

Created: 24July2022 8:00 AM
"""

import re
import gzip
import math
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import glob
import os
import random
from tqdm import tqdm

ALPHABET = ['A', 'T', 'C', 'G']
SCALE_CSV = "metadata/scaler.csv"
FASTA_PARSE_REGEX = re.compile('\>.+?\n([\w\n]+)')
PREPARED_TRAIN_DATASET_FILE = "prepared_train_dataset.csv"
PREPARED_TEST_DATASET_FILE = "prepared_test_dataset.csv"

class Contigs2Vec(object):
  def __init__(self, ks=[3, 4, 5]):
    self.optional_ks = ks
    self.kmers_dicts = dict()
    self.scaler = StandardScaler()
    
    self._create_kmer_dict()

    scaler_df = pd.read_csv(SCALE_CSV)
    self.scaler.scale_ = np.asarray(scaler_df["scale"].values, np.float32)
    self.scaler.mean_  = np.asarray(scaler_df["mean"].values, np.float32)
    self.scaler.var_   = np.asarray(scaler_df["var"].values, np.float32)

  
  def prepare_single_fasta(self, fasta_file):
    """Parse a Fasta file to be used as input to a model.

    Args:
        fasta_file (str): Path to Fasta file location

    Returns:
        numpy array: representation of contigs in Fasta file
    """
    sequences = self._parse_fasta(fasta_file)
    histograms = self._sequences_to_histograms(sequences)
    print(histograms)
    scaled_hist = self.scaler.transform([histograms])[0]
    representations = self._histogram_to_representation(scaled_hist)
    representations = np.asarray([representations], dtype=np.float32)
    return representations

  
  def prepare_data_for_training(self, dir_path):
    """Take in a directory that has fasta files holding contigs.
    Prep all the sequences to be input into the the model.
    This also trains the scaler and saves it to a file.

    Args:
        dir_path (str): Directory holding Fasta files

    Returns:
        (dict, dict): Training dataset dict and testing dataset dict
    """
    fasta_matrices = dict()

    if not dir_path.endswith("/"):
      dir_path = dir_path + "/"

    if os.path.exists(dir_path + PREPARED_TRAIN_DATASET_FILE):
      max_k_size = self._get_kmers_count(max(self.optional_ks))
      num_ks = len(self.optional_ks)
      training_df = pd.read_csv(dir_path + PREPARED_TRAIN_DATASET_FILE, index_col="file_name")
      training_dict = training_df.to_dict('index')
      for file in training_dict.keys():
         histograms = np.asarray(list(training_dict[file].values()))
         representation = np.reshape(histograms, (1, num_ks, max_k_size))
         training_dict[file] = representation

      test_df = pd.read_csv(dir_path + PREPARED_TEST_DATASET_FILE, index_col="file_name")
      test_dict = test_df.to_dict('index')
      for file in test_dict.keys():
         histograms = np.asarray(list(test_dict[file].values()))
         representation = np.reshape(histograms, (1, num_ks, max_k_size))
         test_dict[file] = representation

    else:
      fasta_files = glob.glob(dir_path + "*.fasta") + glob.glob(dir_path + "*.fasta.gz")

      # For debugging purposes
      # fasta_files = fasta_files[:5]

      for fasta_file in tqdm(fasta_files, total=len(fasta_files)):
        file_name = os.path.basename(fasta_file)      
        sequences = self._parse_fasta(fasta_file)
        histograms = self._sequences_to_histograms(sequences)
        fasta_matrices[file_name] = histograms

      fasta_keys = list(fasta_matrices.keys())
      training_indices = random.sample(list(range(0, len(fasta_keys))), int(len(fasta_keys) * 0.9))
      train_keys = [fasta_keys[i] for i in training_indices]
      test_keys = [fasta_keys[i] for i in range(len(fasta_keys)) if i not in training_indices]

      training_data = []
      for key in train_keys:
        training_data.append(fasta_matrices[key])
      
      self.scaler = StandardScaler().fit(training_data)
      self._save_scaler()

      fasta_matrices.update( (k, self.scaler.transform([fasta_matrices[k]])[0]) for k in fasta_matrices)    
      fasta_matrices.update( (k, self._histogram_to_representation(fasta_matrices[k])) for k in fasta_matrices)
      fasta_matrices.update( (k, np.asarray([fasta_matrices[k]], dtype=np.float32)) for k in fasta_matrices)

      training_dict = {k: fasta_matrices[k] for k in train_keys}
      test_dict = {k: fasta_matrices[k] for k in test_keys}

      training_array = [np.array(fasta_matrices[k]).flatten() for k in train_keys]
      test_array = [np.array(fasta_matrices[k]).flatten() for k in test_keys]

      training_df = pd.DataFrame(training_array, index=train_keys)
      test_df = pd.DataFrame(test_array, index=test_keys)

      training_df.index.rename('file_name', inplace=True)
      test_df.index.rename('file_name', inplace=True)

      training_df.to_csv(dir_path + PREPARED_TRAIN_DATASET_FILE)
      test_df.to_csv(dir_path + PREPARED_TEST_DATASET_FILE)
    
    # print(list(test_dict.values())[0])
    # print(list(test_dict.values())[0].shape)
    return training_dict, test_dict


  def convert_fasta_dir(self, dir_path):
    """Go over each Fasta file in a directory and convert the contigs
    in those files to matrix embeddings.
    

    Args:
        dir_path (str): Directory holding Fasta files

    Returns:
        dict: Dictionary where key -> file name, value -> embedding matrix
    """
    fasta_matrices = dict()

    if not dir_path.endswith("/"):
      dir_path = dir_path + "/"

    fasta_files = glob.glob(dir_path + "*.fasta") + glob.glob(dir_path + "*.fasta.gz")

    for fasta_file in fasta_files:
      file_name = os.path.basename(fasta_file)
      fasta_matrices[file_name] = self.convert_fasta(fasta_file)

    return fasta_matrices


  def convert_fasta(self, file_path):
    """Convert a Fasta file (may be compressed) into an embeding matrix representation.

    Args:
        file_path (str): File path to Fasta file

    Returns:
        numpy array: Embedded matrix representation of all contigs in Fasta file
    """
    sequences = self._parse_fasta(file_path)
    histograms = self._sequences_to_histograms(sequences)
    histograms = self.scaler.transform([histograms])[0]
    embeded_representation = self._histogram_to_representation(histograms)
    embeded_representation = np.asarray([embeded_representation], dtype=np.float32)
    return embeded_representation


  def _get_kmers_count(self, k):
    """Get the number of K-Mers that are possible for a given k.

    Args:
        k (int): Size of K-Mers

    Returns:
        int: Number of possible K-Mers that can be made of DNA for k size
    """
    return int(math.pow(len(ALPHABET), k))


  def _create_kmer_dict(self):
    """
    Create a dictionary of K-Mers for each K-size.
    Each integer is the index of that K-Mer for the 
    value of k.

    {
      3: {
        "AAA": 0,
        "AAT": 1,
        "AAC": 2,
        "AAG": 3,
        "ATA": 4,
        ...
      },
      4: {
        "AAAA": 0,
        "AAAT": 1,
        ...
      },
      ...
    }
    """
    for k in self.optional_ks:
      kmers = {}
      
      for i in range(0, self._get_kmers_count(k)):
        kmer = ''
        mask = 3

        # This is essentially a dynamic for loop that 
        # can build sequences of size k using 4 letters
        # (i & mask) -> bitwise and only looking at 2 bits (11, 1100, 110000, ..)
        # >> (2 * j) -> Cut off trailing bits after mask (result will 00, 01, 10, or 11)
        # mask << 2 -> Move the 2 '1' bits left by 2 (000011, 001100, 110000, ..)
        for j in range(0, k):
          kmer += ALPHABET[(i & mask) >> (2 * j)]
          mask = mask << 2

        kmers[kmer] = i
        
      self.kmers_dicts[k] = kmers


  def _parse_fasta(self, file_path):
    """Convert a Fasta file (may be compressed) into an embeding matrix representation.

    Args:
        file_path (str): File path to Fasta file

    Returns:
        [str]: List of sequences from fasta
    """
    sequences = []

    if file_path.endswith(".gz"):
      with gzip.open(file_path, 'rt') as fasta_file:
        try:
          file_contents = fasta_file.read()
          sequences = re.findall(FASTA_PARSE_REGEX, file_contents)
        except:
          raise(f"Error while reading fasta sequence from gziped file: {file_path}")
    else:
      with open(file_path, 'r') as fasta_file:
        file_contents = fasta_file.read()
        sequences = re.findall(FASTA_PARSE_REGEX, file_contents)

    sequences = [seq.replace('\n', '').upper() for seq in sequences]
    return sequences


  def _sequences_to_histograms(self, sequences):
    """Take in a list of sequences and turn them into histogram of K-Mer counts.

    Args:
        sequences ([str]): List of sequences

    Returns:
        list: List of histograms where each histogram is a list of K-Mer counts for a specific k
    """
    histograms = np.array([], dtype=np.float32)
    k_histograms = {}
    for k in self.optional_ks:
      k_histograms[k] = np.zeros(self._get_kmers_count(k), dtype=np.float32)
    
    for sequence in sequences:
      for k in self.optional_ks:
        kmers_dict = self.kmers_dicts[k]
        histogram = k_histograms[k]
        for position in range(0, len(sequence) - min(self.optional_ks) + 1):
          substring = sequence[position : position + k]
          index = kmers_dict.get(substring, None)
          if index is not None:
            histogram[index] += 1
        
    for k in sorted(self.optional_ks):
      histograms = np.append(histograms, k_histograms[k])
    
    return histograms


  def _histogram_to_representation(self, histograms):
    """Turn a list of histograms into a matrix.

    Args:
        histograms (list): List of histograms

    Returns:
        numpy array: Numpy matrix of histograms
    """
    max_kmer_count = self._get_kmers_count(max(self.optional_ks))
    representation = np.zeros((len(self.optional_ks), max_kmer_count))
    
    # k=3 is the minimum considered
    # Start at minimum or 3, whichever is larger
    hists_ptr = 0
    for k in range(3, min(self.optional_ks)):
      hists_ptr += self._get_kmers_count(k)
    
    for i in range(0, len(self.optional_ks)):
      kmers_count = self._get_kmers_count(self.optional_ks[i])
      for j in range(0, kmers_count):
        representation[i][j] = histograms[hists_ptr]
        hists_ptr += 1
    
    return representation

  def _save_scaler(self):
    scaler_df = pd.DataFrame(zip(self.scaler.scale_, self.scaler.mean_, self.scaler.var_), columns=['scale', 'mean', 'var'])
    scaler_df.to_csv(SCALE_CSV, index=False)

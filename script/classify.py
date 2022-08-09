"""
File that can load models to classify on a FASTA file.

@author Cory Kromer-Edwards

Created: 7Aug2022 7:46 PM
"""

import contigs2vec
import tensorflow as tf
from tensorflow.keras import models
import numpy as np
import argparse

# This is required for the GPU to have enough memory to load the CudaDNN files for model.
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

# Where models are saved to be loaded in later on
MODEL_OUT_DIR = "../output/models"

# List of antibiotics that will be made into ML models
DRUGS = ["Aztreonam", "Ceftazidime-avibactam", "Piperacillin-tazobactam", "Ampicillin-sulbactam",
               "Ceftolozane-tazobactam",
               "Cefepime", "Ceftaroline", "Ceftazidime", "Ceftobiprole", "Ceftriaxone", "Imipenem", "Doripenem",
               "Meropenem"]

# List of MICs to use as classes
OPTIONAL_MICS = [
    # 0.00012,
    # 0.00025,
    # 0.0005,
    # 0.001,
    # 0.002,
    # 0.004,
    0.008,
    0.015,
    0.03,
    0.06,
    0.12,
    0.25,
    0.5,
    1.0,
    2.0,
    4.0,
    8.0,
    16.0,
    32.0,
    64.0,
    128.0,
    256.0,
    512.0,
    1024.0,
    2048.0
    # 4096.0,
    # 8192.0,
    # 16384.0
  ]

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('fasta_file')
  args = parser.parse_args()

  contig_worker = contigs2vec.Contigs2Vec()
  representation = contig_worker.prepare_single_fasta(args.fasta_file)

  for drug in DRUGS:
    model = models.load_model(f"{MODEL_OUT_DIR}/{drug}")
    prediction = model(representation)[0]
    mic_index = np.argmax(prediction)
    mic = OPTIONAL_MICS[mic_index]
    print(f"{drug}\t{prediction}\t{mic}")

if __name__ == '__main__':
  main()
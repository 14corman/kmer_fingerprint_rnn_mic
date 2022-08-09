"""
File with functions to train the RNN model.

@author Cory Kromer-Edwards

Created: 24July2022 5:33 PM
"""

import contigs2vec
import tensorflow as tf
import math
import pandas as pd
from tensorflow.keras import layers, models, utils, optimizers, metrics
import keras_tuner as kt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
import seaborn as sns

# Seaborn has five built-in themes to style its plots: darkgrid, whitegrid, dark, white, and ticks. Seaborn defaults to using the darkgrid theme
sns.set_style("ticks")

# In order of relative size they are: paper, notebook, talk, and poster. The notebook style is the default.
sns.set_context("paper")

# This is required for the GPU to have enough memory to load the CudaDNN files for model.
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

# Where contig fasta files are located
DATA_DIR = "../data/fastas"

# Where training PDFs are located for Loss and AUC plots
TRAINING_OUT_DIR = "../output/training"

# Where overall results are put and F1 score plot
RESULT_OUT_DIR = "../output/results"

# Where models are saved to be loaded in later on
MODEL_OUT_DIR = "../output/models"

# Where label file is located, and its name
LABEL_FILE = "../data/mic_data.csv"

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

# Number of MICs that are being used
NUM_CLASSES = len(OPTIONAL_MICS)

# List of antibiotics that will be made into ML models
DRUGS = ["Aztreonam", "Ceftazidime-avibactam", "Piperacillin-tazobactam", "Ampicillin-sulbactam",
               "Ceftolozane-tazobactam",
               "Cefepime", "Ceftaroline", "Ceftazidime", "Ceftobiprole", "Ceftriaxone", "Imipenem", "Doripenem",
               "Meropenem"]

# Best epoch to train models to (found when initially training)
# If this is set to None, then a temporary model is trained to find the best epoch
BEST_EPOCH = 52

# Whether you want to skip training and just plot results or you want to tune and train each antibiotic model
SKIP_TRAIN = False

class BahdanauAttention(layers.Layer):
  """Code found at: 
  https://medium.com/analytics-vidhya/neural-machine-translation-using-bahdanau-attention-mechanism-d496c9be30c3
  """
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.hidden_units = units
    self.W1 = layers.Dense(units)
    self.W2 = layers.Dense(units)
    self.V = layers.Dense(1)

  def call(self, query, values):
    query_with_time_axis = tf.expand_dims(query, 1)
    score = self.V(tf.nn.tanh(
        self.W1(query_with_time_axis) + self.W2(values)))

    attention_weights = tf.nn.softmax(score, axis=1)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)
    return context_vector, attention_weights
    
  def get_config(self):
      return {"hidden_units": self.hidden_units}

  @classmethod
  def from_config(cls, config):
      return cls(**config)


def rnn_model_for_tuning(hp):
  """The main model building function used for tuning.

  Args:
      hp (HyperBand): A keras-tuner HyperBand used to tell the tuner options for tuning

  Returns:
      Mode: RNN model that is compiled
  """
  num_kmers = int(math.pow(4, 5))
  input = layers.Input(shape=(3, num_kmers))
  gru_layer = layers.GRU(hp.Int('gru_hidden_units', 24, 128, step=24, default=48), activation="tanh")
  [forward_output, backward_output] = layers.Bidirectional(gru_layer, merge_mode=None)(input)
  context_vector, _ = BahdanauAttention(forward_output.shape[-1])(forward_output, backward_output)
  x = layers.Concatenate()([context_vector, forward_output, backward_output])
  x = layers.Dropout(hp.Float('dropout_1', 0, 0.7, step=0.1, default=0.5))(x)
  x = layers.Dense(hp.Int('dense_hidden_units', 24, 128, step=24, default=48), activation="relu")(x)
  x = layers.Dropout(hp.Float('dropout_2', 0, 0.7, step=0.1, default=0.5))(x)
  x = layers.Dense(NUM_CLASSES, activation="softmax")(x)

  adam_opt = optimizers.Adam(
      learning_rate=hp.Float('learning_rate', 1e-4, 1e-2, sampling='log', default=0.001),
      beta_1=0.9,
      beta_2=0.999,
      epsilon=hp.Float('epsilon', 1e-9, 1e-7, sampling='log', default=1e-8))

  model = models.Model(input, x)
  model.compile(optimizer=adam_opt, loss='categorical_crossentropy', metrics=[metrics.AUC(name='auc')])

  return model


def best_rnn_model(gru_hidden_units=120, dropout_1=0.1, dense_hidden_units=24, dropout_2=0.2, learning_rate=0.00033, epsilon=3.705e-09):
  """Model building function that can be used for testing after tuning is done. Not used in main code right now.

  Args:
      gru_hidden_units (int, optional): GRU number of units. Defaults to 120.
      dropout_1 (float, optional): First Dropout layer. Defaults to 0.3.
      dense_hidden_units (int, optional): Number of units in hidden dense layer. Defaults to 96.
      dropout_2 (float, optional): Second dropout layer. Defaults to 0.6.
      learning_rate (float, optional): Learning rate. Defaults to 0.00965.
      epsilon (_type_, optional): Epsilon. Defaults to 1e-8.

  Returns:
      Mode: RNN model that is compiled
  """
  num_kmers = int(math.pow(4, 5))
  input = layers.Input(shape=(3, num_kmers))
  gru_layer = layers.GRU(gru_hidden_units, activation="tanh")
  [forward_output, backward_output] = layers.Bidirectional(gru_layer, merge_mode=None)(input)
  context_vector, _ = BahdanauAttention(forward_output.shape[-1])(forward_output, backward_output)
  x = layers.Concatenate()([context_vector, forward_output, backward_output])
  x = layers.Dropout(dropout_1)(x)
  x = layers.Dense(dense_hidden_units, activation="relu")(x)
  x = layers.Dropout(dropout_2)(x)
  x = layers.Dense(NUM_CLASSES, activation="softmax")(x)

  adam_opt = optimizers.Adam(
      learning_rate=learning_rate,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=epsilon)

  model = models.Model(input, x)
  model.compile(optimizer=adam_opt, loss='categorical_crossentropy', metrics=[metrics.AUC(name='auc')])

  return model


def normalize_mics(col):
  """Normalize MICs to remove '<=' and '>'

  Args:
      col (list): Column of MICs for an antibiotic

  Returns:
      list: List of normalized MICs
  """
  mics = []
  for mic in col:
    if pd.isna(mic):
      mics.append(np.nan)
    elif "<=" in str(mic):
      mics.append(float(mic[2:]))
    elif ">" in str(mic):
      actual_mic = float(mic[1:])
      next_mic_index = OPTIONAL_MICS.index(actual_mic) + 1
      mics.append(OPTIONAL_MICS[next_mic_index])
    else:
      mics.append(float(mic))

  return mics


def get_drug_data(drug, mics_df, train_x_dict, test_x_dict):
  """Collect processed MICs and K-Mer counts for a particular antibiotic

  Args:
      drug (str): Antibiotic name to collect data for
      mics_df (DataFrame): DataFrame for labels where each column is an antibiotic
      train_x_dict (dict): Dictionary of Fasta file keys and their processed K-Mers for training
      test_x_dict (dict): Dictionary of Fasta file keys and their processed K-Mers for testing

  Returns:
      tuple: Numpy arrays of train_X, train_y, test_X, test_y
  """
  drug_series = mics_df[[drug]].dropna()
  train_y = []
  train_X = []
  test_y = []
  test_X = []
  for key in drug_series.index:
    fasta_file = f"{key}_contigs.fasta"
    if drug_series.at[key, drug] is not None:
      if fasta_file in train_x_dict:
        train_y.append(drug_series.at[key, drug])
        train_X.append(train_x_dict.get(fasta_file))
      elif fasta_file in test_x_dict:
        test_y.append(drug_series.at[key, drug])
        test_X.append(test_x_dict.get(fasta_file))

  test_y = [OPTIONAL_MICS.index(x) for x in test_y]
  train_y = [OPTIONAL_MICS.index(x) for x in train_y]

  test_y = utils.to_categorical(test_y, num_classes=NUM_CLASSES)
  train_y = utils.to_categorical(train_y, num_classes=NUM_CLASSES)

  train_X = np.asarray(train_X)
  test_X = np.asarray(test_X)

  train_X = np.squeeze(train_X, axis=1)
  test_X = np.squeeze(test_X, axis=1)

  print(train_X.shape)
  print(test_X.shape)

  return train_X, train_y, test_X, test_y


def plot_train_history(train_history, filename_prefix):
  """Plot loss and AUC for training and validation history over epochs

  Args:
      train_history (Keras History): Output from fit() function
      filename_prefix (str): Prefix to give to PDFs
  """
  plt.plot(train_history.history['loss'])
  plt.plot(train_history.history['val_loss'])
  plt.title('Model Loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Validation'], loc='upper left')
  plt.savefig(f"{TRAINING_OUT_DIR}/{filename_prefix}_train_loss.pdf", bbox_inches='tight')
  plt.clf()
  
  plt.plot(train_history.history['auc'])
  plt.plot(train_history.history['val_auc'])
  plt.title('Model AUC')
  plt.ylabel('AUC')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Validation'], loc='upper left')
  plt.savefig(f"{TRAINING_OUT_DIR}/{filename_prefix}_train_auc.pdf", bbox_inches='tight')
  plt.clf()


def determine_pm_1_test(y_test, y_pred):
  """Determine +-1 2-fold dilution prediction accuracy.
  Then determine F1 scores, accuracy, and AUC. Print and return those values.

  Args:
      y_test (numpy array): Actual labels
      y_pred (numpy array): Predicted labels

  Returns:
      tuple: All defined metrics f1_macro, f1_micro, accuracy, auc
  """
  y_pred_pm_1 = []
  for test, pred in zip(y_test, y_pred):
    test_label = np.argmax(test)
    pred_label = np.argmax(pred)
    if pred_label == test_label or pred_label + 1 == test_label or pred_label - 1 == test_label:
      label = test_label
    else:
      label = pred_label

    new_pred = [0 for _ in range(NUM_CLASSES)]
    new_pred[label] = 1
    y_pred_pm_1.append(new_pred)

  y_pred_pm_1 = np.asarray(y_pred_pm_1)

  f1_macro = f1_score(y_test, y_pred_pm_1, average='macro', zero_division=1)
  f1_micro = f1_score(y_test, y_pred_pm_1, average='micro', zero_division=1)
  accuracy = accuracy_score(y_test, y_pred_pm_1)
  auc = roc_auc_score(y_test, y_pred_pm_1, average="micro", multi_class="ovo")

  print("USE MICRO SINCE THERE IS MAJOR CLASS IMBALANCE, BUT SHOWING MACRO FOR COMPLETENESS")
  print(f"+-1 test macro F1 score: {f1_macro}")
  print(f"+-1 test micro F1 score: {f1_micro}")
  print(f"+-1 test accuracy: {accuracy}")
  print(f"+-1 test AUC: {auc}")

  return f1_macro, f1_micro, accuracy, auc


def train_for_drug(drug, hypermodel, train_X, train_y, best_epoch, test_X, test_y):
  """Take a model and train it given training data and test data. Also plot loss and AUC history.

  Args:
      drug (str): Antibiotic name
      hypermodel (Model): Model compiled based on best hyperparameters
      train_X (numpy array): training input
      train_y (numpy array): training labels
      best_epoch (int): Epoch to train to
      test_X (numpy array): test input
      test_y (numpy array): test labels

  Returns:
      tuple: All defined test metrics f1_macro, f1_micro, accuracy, auc
  """
  train_history = hypermodel.fit(train_X, train_y, shuffle=True, epochs=best_epoch, validation_split=0.2, verbose=0)
  plot_train_history(train_history, drug)

  eval_result = hypermodel.evaluate(test_X, test_y)
  print("Exact [test loss, test AUC]:", eval_result)

  # Save drug model
  # To use later, load the model with: keras.models.load_model(f"{MODEL_OUT_DIR}/{drug}")
  hypermodel.save(f"{MODEL_OUT_DIR}/{drug}")

  return determine_pm_1_test(test_y, hypermodel.predict(test_X))
  

def main():
  if not SKIP_TRAIN:
    contig_worker = contigs2vec.Contigs2Vec()
    train_x_dict, test_x_dict = contig_worker.prepare_data_for_training(DATA_DIR)

    mics_df = pd.read_csv(LABEL_FILE, index_col="index")
    mics_df = mics_df.apply(normalize_mics)

    train_X, train_y, test_X, test_y = get_drug_data(DRUGS[0], mics_df, train_x_dict, test_x_dict)

    # Hyperparameters:
    # gru_hidden_units:   120
    # dropout_1:          0.1
    # dense_hidden_units: 24
    # dropout_2:          0.2
    # learning_rate:      0.0003329759758300238
    # epsilon:            3.705e-09
    # Score: 0.9821969270706177
    tuner = kt.Hyperband(
      rnn_model_for_tuning,
      objective=kt.Objective("val_auc", direction="max"),
      max_epochs=1000,
      hyperband_iterations=2)

    tuner.search(train_X, train_y,
              validation_split=0.1,
              epochs=1000)

    # tuner.results_summary()
    best_model = tuner.get_best_models(1)[0]
    best_model.summary()
    best_hyperparameters = tuner.get_best_hyperparameters(1)[0]
    print("SHOWING BEST HYPERPARAMETERS")
    for param in ["gru_hidden_units", "dropout_1", "dense_hidden_units", "dropout_2", "learning_rate", "epsilon"]:
      print(f"{param}:\t{best_hyperparameters.get(param)}")

    if BEST_EPOCH is None:
      model = tuner.hypermodel.build(best_hyperparameters)
      # model = best_rnn_model()
      history = model.fit(train_X, train_y, shuffle=True, epochs=1000, validation_split=0.2, verbose=0)

      val_acc_per_epoch = history.history['val_auc']
      best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
      print(f'Best epoch: {best_epoch}')
    else:
      best_epoch = BEST_EPOCH

    results = []
    for drug in DRUGS:
      print(f"===================WORKING ON DRUG: {drug}===============================================")
      train_X, train_y, test_X, test_y = get_drug_data(drug, mics_df, train_x_dict, test_x_dict)
      f1_macro, f1_micro, accuracy, auc = train_for_drug(drug, tuner.hypermodel.build(best_hyperparameters), train_X, train_y, best_epoch, test_X, test_y)
      # f1_macro, f1_micro, accuracy, auc = train_for_drug(drug, best_rnn_model(), train_X, train_y, best_epoch, test_X, test_y)
      results.append([drug, f1_macro, f1_micro, accuracy, auc])
      print("======================================================================")

    results_df = pd.DataFrame(results, columns=["Antibiotic", "F1 Macro Score", "F1 Micro Score", "Accuracy", "AUC"])
    results_df.to_csv(f"{RESULT_OUT_DIR}/overall_results.csv", index=False)
  else:
    results_df = pd.read_csv(f"{RESULT_OUT_DIR}/overall_results.csv")

  sns.barplot(data=results_df, x="F1 Micro Score", y="Antibiotic", color="blue", saturation=.5)
  sns.despine()  # Remove the top and right graph lines
  plt.savefig(f'{RESULT_OUT_DIR}/f1_micro_results.pdf', bbox_inches='tight')
  plt.clf()


if __name__ == '__main__':
  main()

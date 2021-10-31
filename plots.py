import os
import glob
import json

import tensorflow.compat.v2 as tf

import pandas as pd
import matplotlib.pyplot as plt

from absl import flags


FLAGS = flags.FLAGS

def print_contrastive_history(training_history, flags):
  con_acc = training_history['train/contrast_acc']
  nrows = training_history.shape[0]
  if flags['checkpoint_epochs'] == 1 and flags['checkpoint_steps'] == 0:
    xvalues = pd.Series(range(nrows))+1
    plt.xlabel('Epochs')
  else:
    xvalues = training_history['global_step']
    plt.xlabel('Steps')
 
  plt.plot(xvalues, con_acc, color='black', label='Contrastive accuracy')
  plt.title('Accuracy')
  plt.ylabel('Accuracy')
  plt.legend()

  # plt.savefig(os.path.join(FLAGS.model_dir, 'Accuracy.jpeg'))
  save_path = os.path.join(FLAGS.tmp_folder, 'Accuracy.jpeg')
  plt.savefig(save_path)
  plt.close()

  if tf.io.gfile.exists(save_path):
    dest_path = os.path.join(FLAGS.model_dir, 'Accuracy.jpeg')
    tf.io.gfile.copy(save_path, dest_path)

  return

def print_accuracy_history(training_history, flags):
  acc = training_history['train/supervised_acc']
  val_acc = training_history['eval/label_top_1_accuracy']
  nrows = training_history.shape[0]
  if flags['checkpoint_epochs'] == 1 and flags['checkpoint_steps'] == 0:
    xvalues = pd.Series(range(nrows))+1
    plt.xlabel('Epochs')
  else:
    xvalues = training_history['global_step']
    plt.xlabel('Steps')
 
  plt.plot(xvalues, acc, color='darkorange', label='Train')
  plt.plot(xvalues, val_acc, color='steelblue', label='Valid')
  plt.title('Accuracy')
  plt.ylabel('Accuracy')
  plt.legend()
  plt.savefig(os.path.join(FLAGS.model_dir, 'Accuracy.jpeg'))
  plt.close()
  return

def print_loss_history(training_history, flags, logscale=False):
  val_sup_loss = training_history['eval/supervised_loss']
  train_sup_loss = training_history['train/supervised_loss']
  nrows = training_history.shape[0]
  if flags['checkpoint_epochs'] == 1 and flags['checkpoint_steps'] == 0:
    xvalues = pd.Series(range(nrows))+1
    plt.xlabel('Epochs')
  else:
    xvalues = training_history['global_step']
    plt.xlabel('Steps')

  plt.plot(xvalues, train_sup_loss, color='darkorange', label='Train')
  plt.plot(xvalues, val_sup_loss, color='steelblue', label='Valid')
  plt.title('Supervised Loss')
  plt.ylabel('Loss')
  plt.legend()
  if logscale:
      plt.yscale('log')
  plt.savefig(os.path.join(FLAGS.model_dir, 'Loss.jpeg'))
  plt.close()
  return

def gen_plots():
  """Generate train plots."""
  def create_df(fnames):
    results = []

    for fname in fnames:
      with tf.io.gfile.GFile(fname, 'r') as f:
        result = json.load(f)
      results.append(result)

    df = pd.DataFrame(results)
    return df
  
  # result_paths = glob.glob(os.path.join(FLAGS.model_dir, 'result_[0-9]*.json'))
  # eval_df = create_df(result_paths)
  metric_paths = tf.io.gfile.glob(os.path.join(FLAGS.model_dir, 'metric_[0-9]*.json'))
  train_df = create_df(metric_paths)
  # results_df = train_df.merge(eval_df, how='left', on='global_step')

  flags_path = os.path.join(FLAGS.model_dir, 'flags.json')
  with tf.io.gfile.GFile(flags_path, 'r') as f:
    flags_dict = json.load(f)

  print_contrastive_history(train_df, flags_dict)
  # print_accuracy_history(results_df, flags_dict)
  # print_loss_history(results_df, flags_dict)
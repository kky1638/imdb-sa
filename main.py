"""Main to run TensorFlow models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import flags

import tensorflow as tf

from . import input_fn
from . import model
from . import util

FLAGS = flags.FLAGS

flags.DEFINE_enum('mode', None, ['train', 'eval'], 'Execution mode.')

flags.DEFINE_string('logdir', '/tmp/sentiment-analysis', 'Model directory.')

flags.DEFINE_enum('model', 'rnn', ['mlp', 'rnn'], 'Type of model to use.')

flags.DEFINE_enum('optimizer', 'adam',
                  ['sgd', 'rmsprop', 'adam'],
                  'Type of optimizer to use for training.')

flags.DEFINE_enum('encoding', 'subwords8k',
                  ['plain_text', 'bytes', 'subwords8k', 'subwords32k'],
                  'Type of text encoding to use.')

flags.DEFINE_integer('num_epochs', 10, 'Number of epochs to run for training.')

flags.DEFINE_integer('num_layers', 1, 'Number of hidden layers.')

flags.DEFINE_list('num_units', [64], 'Number of hidden units.')

flags.DEFINE_enum('cell_type', 'lstm',
                  ['gru', 'lstm', 'bidi-gru', 'bidi-lstm'],
                  'Type of RNN cell to use.')

flags.DEFINE_integer('embedding_size', 32, 'Size of the input embedding.')

flags.DEFINE_integer('batch_size', 16, 'Size of the batch.')

flags.DEFINE_bool('verbose', True, 'Verbosity.')

flags.DEFINE_integer('max_length', None, 'Maximum length input to train on.')

flags.DEFINE_bool('early_stop', False, 'True to early stop')


def create_model(vocab_size):
  """Creates a Keras model."""
  num_units = [int(num_unit) for num_unit in FLAGS.num_units]
  if FLAGS.model == 'rnn':
    new_model = model.rnn_model(FLAGS.num_layers, FLAGS.cell_type, num_units,
                                vocab_size, FLAGS.embedding_size)
  else:
    new_model = model.mlp_model(FLAGS.num_layers, num_units, vocab_size)
  new_model.compile(optimizer=FLAGS.optimizer, loss='binary_crossentropy',
                    metrics=['accuracy'])
  new_model.summary()
  return new_model


def run_train():
  """Trains a model."""
  # Set up input pipeline.
  input_dataset = input_fn.InputDataset(FLAGS.encoding)
  tokenizer = input_dataset.tokenizer

  use_bow = (FLAGS.model == 'mlp')
  train_dataset = input_dataset.input_fn('train', FLAGS.batch_size, bow=use_bow)
  test_dataset = input_dataset.input_fn('test', 10, bow=use_bow)

  new_model = create_model(tokenizer.vocab_size)
  latest_checkpoint = tf.train.latest_checkpoint(FLAGS.logdir)
  if latest_checkpoint:
    print("Reloading from {}".format(latest_checkpoint))
    new_model.load_weights(latest_checkpoint)

  # Define callbacks to run during training.
  callbacks = []

  checkpoint = util.CNSModelCheckpoint(os.path.join(FLAGS.logdir, FLAGS.model))
  callbacks.append(checkpoint)

  tensorboard = tf.keras.callbacks.TensorBoard(
      log_dir=FLAGS.logdir, update_freq='batch')
  callbacks.append(tensorboard)

  if FLAGS.early_stop:
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', min_delta=0.0001, patience=10)
    callbacks.append(early_stop)

  # Start training.
  history = new_model.fit(train_dataset, epochs=FLAGS.num_epochs,
                          callbacks=callbacks,
                          validation_data=test_dataset,
                          validation_steps=25,
                          verbose=int(FLAGS.verbose))

  # Write out the training history.
  dirname = os.path.dirname(FLAGS.logdir)
  if not tf.gfile.Exists(dirname):
    tf.gfile.MakeDirs(dirname)
  with tf.gfile.GFile(os.path.join(FLAGS.logdir, 'history.txt'), 'w') as f:
    f.write(str(history.history))


def run_eval():
  """Evaluates a model."""
  # Set up input pipeline.
  input_dataset = input_fn.InputDataset(FLAGS.encoding)
  tokenizer = input_dataset.tokenizer

  use_bow = (FLAGS.model == 'mlp')
  dataset = input_dataset.input_fn('test', FLAGS.batch_size, bow=use_bow)

  new_model = create_model(tokenizer.vocab_size)
  latest_checkpoint = tf.train.latest_checkpoint(FLAGS.logdir)
  if latest_checkpoint:
    print("Reloading from {}".format(latest_checkpoint))
    new_model.load_weights(latest_checkpoint)

  ret = new_model.evaluate(dataset)
  print('Eval results: {}'.format(ret))


def main(unused_argv):
  if FLAGS.mode == 'train':
    run_train()
  elif FLAGS.mode == 'eval':
    run_eval()


if __name__ == '__main__':
  tf.app.run(main)

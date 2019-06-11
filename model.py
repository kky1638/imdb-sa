"""Models for sentiment analysis task."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags

import tensorflow as tf

flags = flags.FLAGS


def mlp_model(num_layers, num_units, vocab_size):
  """Returns a feed-forward network Keras model.

  Args:
    num_layers: Number of feed-forward layers.
    num_units: Number of hidden units. Should be a list of ints of length equal
      to num_layers.
    vocab_size: Size of the input vocabulary.
  """
  assert len(num_units) == num_layers

  # Create an MLP Keras model.
  model = tf.keras.Sequential()

  # Create feed-forward layers.
  for l in range(num_layers):
    input_size = vocab_size if l == 0 else num_units[l - 1]
    dense_layer = tf.keras.layers.Dense(
        num_units[l], input_shape=(input_size,), activation='relu')
    model.add(dense_layer)

  output_layer = tf.keras.layers.Dense(1, activation='sigmoid')
  model.add(output_layer)

  return model


def rnn_model(num_layers, cell_type, num_units, vocab_size, embedding_size):
  """Returns an RNN-based Keras model.

  Args:
    num_layers: Number of RNN layers.
    cell_type: Type of RNN cell. See rnn_cell() method.
    num_units: Number of hidden units. Should be a list of ints of length equal
      to num_layers.
    vocab_size: Size of the input vocabulary.
    embedding_size: Size of the input embedding.
  """
  assert len(num_units) == num_layers

  # Create a Keras model.
  model = tf.keras.Sequential()

  # Create an input embedding layer.
  embedding_layer = tf.keras.layers.Embedding(
      input_dim=vocab_size, output_dim=embedding_size)
  model.add(embedding_layer)

  # Create RNN layers.
  for l in range(num_layers):
    return_sequences = (l + 1) < num_layers
    hidden_units = num_units[l]
    rnn_layer = _rnn_layer(cell_type, hidden_units, return_sequences)
    model.add(rnn_layer)

  # Create output layers.
  output_layer = tf.keras.layers.Dense(1, activation='sigmoid')
  model.add(output_layer)

  return model


def _rnn_layer(cell_type, num_units, return_sequences=False):
  """Returns an RNN layer.

  Args:
    cell_type: One of 'gru', 'lstm', 'bidi-gru', 'bidi-lstm'.
  """
  if cell_type == 'gru':
    return tf.keras.layers.GRU(
        units=num_units, return_sequences=return_sequences)
  if cell_type == 'lstm':
    return tf.keras.layers.LSTM(
        units=num_units, return_sequences=return_sequences)
  if cell_type == 'bidi-gru':
    layer = tf.keras.layers.GRU(
        units=num_units, return_sequences=return_sequences)
    return tf.keras.layers.Bidirectional(layer)
  if cell_type == 'bidi-lstm':
    layer = tf.keras.layers.LSTM(
        units=num_units, return_sequences=return_sequences)
    return tf.keras.layers.Bidirectional(layer)
  raise ValueError('Unsupported cell type: %s' % cell_type)

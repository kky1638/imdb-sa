"""Input pipeline using Dataset API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from absl import flags

import tensorflow as tf
import tensorflow_datasets as tfds

FLAGS = flags.FLAGS

flags.DEFINE_integer('shuffle_buffer_size', 5000, 'Size of the shuffle buffer.')


class InputDataset(object):
  """Input pipeline for the IMDB dataset.

  Attributes:
    tokenizer: Tokenizer used to encode and decode text.
  """

  def __init__(self, encoding, max_length=None):
    """Creates an InputDataset instance.

    Args:
      encoding: Type of encoding to use. Should be one of 'plain_text', 'bytes',
        'subwords8k', and 'subwords32k'.
    """
    if encoding not in ('plain_text', 'bytes', 'subwords8k', 'subwords32k'):
      raise ValueError('Unsupported encoding type %s' % encoding)

    loaded_imdb = tfds.load(
        'imdb_reviews/{}'.format(encoding), with_info=True, as_supervised=True)
    self._dataset, self._info = loaded_imdb
    self.tokenizer = self._info.features['text'].encoder
    self.max_length = max_length

  def input_fn(self, mode, batch_size, bucket_boundaries=None, bow=False):
    """Returns an instance of tf.data.Dataset.

    Args:
      mode: One of 'train' or 'test'.
      batch_size: Size of a batch.
      bucket_boundaries: List of boundaries for bucketing.
      bow: True to process the input as a bag-of-words.
    """
    if mode not in ('train', 'test'):
      raise ValueError('Unsupported mode type %s' % mode)
    dataset = self._dataset[mode]

    # Transform into a bag-of-words input if applicable.
    def bag_of_words(tokens, label):
      indices = tf.expand_dims(tokens, axis=-1)
      updates = tf.ones([tf.shape(indices)[0]])
      shape = tf.constant([self.tokenizer.vocab_size], dtype=indices.dtype)
      scatter = tf.scatter_nd(indices, updates, shape)
      return scatter, label
    if bow:
      dataset = dataset.map(bag_of_words, num_parallel_calls=12)

    # Shuffle the data.
    if self.max_length:
      dataset = dataset.filter(lambda f, l: tf.shape(f)[0] < self.max_length)
    dataset = dataset.shuffle(
        buffer_size=FLAGS.shuffle_buffer_size, reshuffle_each_iteration=True)

    # Create batches of examples and pad.
    if mode == 'train' and bucket_boundaries:
      bucket_batch_sizes = [batch_size] * (len(bucket_boundaries) + 1)
      dataset = dataset.apply(
          tf.data.experimental.bucket_by_sequence_length(
              lambda feature, label: tf.shape(feature)[0],
              bucket_boundaries=bucket_boundaries,
              bucket_batch_sizes=bucket_batch_sizes,
              padded_shapes=dataset.output_shapes))
    else:
      output_shapes = dataset.output_shapes
      if self.max_length:
        output_shapes = (tf.TensorShape([tf.Dimension(sefl.max_length)]),
                         tf.TensorShape([]))
      dataset = dataset.padded_batch(batch_size, output_shapes)

    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset

"""Sentiment analysis using KNN."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import heapq
import random
import time

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_enum('mode', 'knn', ['knn', 'analyze'], 'Execution mode.')

flags.DEFINE_string('train_data', None, 'Train file in LIBSVM format.')

flags.DEFINE_string('test_data', None, 'Test file in LIBSVM format.')

flags.DEFINE_integer('k_value', 0, 'Value of k.')


def parse_libsvm_file(filename):
  """Parses a file in LIBSVM format."""
  # Features and label.
  data_points = []
  with open(filename) as f:
    for line in f:
      line = line.split()
      assert len(line) > 1
      d = {'features': {}, 'norm': 0.0, 'label': int(line[0])}
      for bow in line[1:]:
        word_id, num_occur = bow.split(':')
        num_occur = float(num_occur)
        d['features'][word_id] = num_occur
        d['norm'] += num_occur ** 2
      data_points.append(d)
  return data_points


def l2_dist(d1, d2):
  """L2 distance between two sparse vectors represented as dicts."""
  if len(d1['features']) < len(d2['features']):
    return l2_dist(d2, d1)
  d1_norm, d2_norm = d1['norm'], d2['norm']
  return (d1_norm + d2_norm - 2 * sum(
      d1['features'].get(key, 0.0) * d2['features'].get(key, 0.0)
      for key in d2['features'].keys()))


def find_knn(data_points, d, k):
  """Finds k-nearest data points."""
  neighbors = []
  heapq.heapify(neighbors)
  for data_point in data_points:
    l2d = l2_dist(data_point, d)
    if len(neighbors) < k:
      heapq.heappush(neighbors, (-l2d, data_point))
    else:
      heapq.heappushpop(neighbors, (-l2d, data_point))
  return [item[1] for item in neighbors]


def run_knn(train_data_points, test_data_points, k):
  """Runs knn and report the overall error rate."""
  count = 0
  num_pos, num_neg = 0.0, 0.0
  num_pos_correct, num_neg_correct = 0.0, 0.0
  for test_data_point in test_data_points:
    count += 1
    if count % 1000 == 0:
      print('Processed {} examples.'.format(count))
    neighbors = find_knn(train_data_points, test_data_point, k)
    score = sum(neighbor['label'] for neighbor in neighbors)
    score /= float(len(neighbors))
    true_score = test_data_point['label']
    if true_score >= 7:
      num_pos += 1
      if score >= 7:
        num_pos_correct += 1
    if true_score <= 4:
      num_neg += 1
      if score <= 4:
        num_neg_correct += 1

  pos_error_rate = 1.0 - num_pos_correct / (num_pos + 1e-8)
  neg_error_rate = 1.0 - num_neg_correct / (num_neg + 1e-8)
  tot_error_rate = (
      1.0 - (num_pos_correct + num_neg_correct) / (num_pos + num_neg + 1e-8))
  print('Pos error rate: {}'.format(round(pos_error_rate, 5)))
  print('Neg error rate: {}'.format(round(neg_error_rate, 5)))
  print('Tot error rate: {}'.format(round(tot_error_rate, 5)))


def run_analysis(data_points):
  """Analyzes input data."""
  num_unique_words_dict = collections.defaultdict(int)
  num_total_words_dict = collections.defaultdict(int)
  for d in data_points:
    num_unique_words_dict[len(d['features']) // 100] += 1
    num_words = sum(d['features'].values())
    num_total_words_dict[num_words // 100] += 1
  num_total = float(len(data_points))
  avg_unique_words = (
      sum(k * v for k, v in num_unique_words_dict.items()) / num_total)
  avg_total_words = (
      sum(k * v for k, v in num_total_words_dict.items()) / num_total)
  print('Dist of unique words count: {}'.format(num_unique_words_dict))
  print('Dist of total words count: {}'.format(num_total_words_dict))


def main(unused_argv):
  print('Start parsing input data..')
  train_data_points = parse_libsvm_file(FLAGS.train_data)
  test_data_points = parse_libsvm_file(FLAGS.test_data)
  if FLAGS.mode == 'knn':
    random.shuffle(train_data_points)
    random.shuffle(test_data_points)
    print('Start running knn..')
    start = time.time()
    run_knn(train_data_points, test_data_points, FLAGS.k_value)
    end = time.time()
    print('Run time: {} secs'.format(round(end - start, 2)))
  elif FLAGS.mode == 'analyze':
    print('Analyze train data:')
    run_analysis(train_data_points)
    print('Analyze test data:')
    run_analysis(test_data_points)


if __name__ == '__main__':
  app.run(main)

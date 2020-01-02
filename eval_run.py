"""Evaluates a model against clean examples both train and test as specified
   in config.json"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import sys

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tqdm import tqdm

import cifar10_input#cifar100_input
from model import Model


def run_eval(checkpoint, train=False, num_eval_examples=None, data_dir=None, num_classes=100, bad_classes=None, savem=False):
    model = Model(mode='eval',class_count=num_classes)
    saver = tf.train.Saver()
    eval_batch_size = 64
    total_corr = 0

    if bad_classes is None:
      cifar = cifar10_input.CIFAR10Data(data_dir)
    else:
      cifar = cifar10_input.CIFAR10Data(data_dir,bad_classes)

    if train:
        x_nat = cifar.train_data.xs
        y_nat = cifar.train_data.ys
    else:
        x_nat = cifar.eval_data.xs
        y_nat = cifar.eval_data.ys

    if num_eval_examples is None:
        num_eval_examples = len(x_nat)

    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
    y_pred = []  # label accumulator
    red_neck = []
    loj_eet = []

    with tf.Session() as sess:
        # Restore the checkpoint
        saver.restore(sess, checkpoint)

        # Iterate over the samples batch-by-batch
        for ibatch in tqdm(range(num_batches)):
            bstart = ibatch * eval_batch_size
            bend = min(bstart + eval_batch_size, num_eval_examples)

            x_batch = x_nat[bstart:bend, :]
            y_batch = y_nat[bstart:bend]
            dict_clean = {model.x_input: x_batch,
                          model.y_input: y_batch}
            cur_corr, feat_reps = sess.run([model.num_correct, model.penultimate],
                                                    feed_dict=dict_clean)

            total_corr += cur_corr
            red_neck.append(feat_reps)
            # print('ibatch: ', ibatch, '/', num_batches)
        print('final accuracy:', float(total_corr)/num_eval_examples)
        if savem:
          red_neck = np.concatenate(red_neck, 0) 
          np.save('robust_CIFAR_10_feats.npy', red_neck)

    accuracy = total_corr / num_eval_examples

    print('Accuracy: {:.2f}%'.format(100.0 * accuracy))
    test_str = 'test'
    if train:
        test_str = 'train'
    #np.save(sys.argv[1] + test_str + '_red_neck.npy', np.concatenate(red_neck,
    #                                                                 axis=0))
    #np.save(sys.argv[1] + test_str + '_logits.npy', np.concatenate(loj_eet,
    #                                                                 axis=0))
    # print('Output saved at pred.npy')


if __name__ == '__main__':
    import json

    with open('config.json') as config_file:
        config = json.load(config_file)

    model_dir = config['model_dir']

    if len(sys.argv) > 2:
        model_dir = 'models/' + sys.argv[1]

    do_train = True
    #if len(sys.argv) > 2:
    #    do_train = sys.argv[2].lower() == 'true'
    #elif len(sys.argv) > 1:
    #    do_train = sys.argv[1].lower() == 'true'


    data_path = config['data_path']
    #bad_classes = config['pretrained_model_classes']
    bad_classes = None

    checkpoint = tf.train.latest_checkpoint(model_dir)

    if checkpoint is None:
        print('No checkpoint found')
    if '100' in model_dir:
      num_classes = 100
    else:
      num_classes = 10
    print('running: ', do_train, model_dir)
    run_eval(checkpoint, train=do_train, data_dir=data_path,num_classes=num_classes, bad_classes=bad_classes, savem=True)

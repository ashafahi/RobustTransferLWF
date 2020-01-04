"""Trains a model, saving checkpoints along
   the way."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datetime import datetime
import json
import os
import shutil
import math
from tqdm import tqdm

import tensorflow as tf
import numpy as np
import sys
from pgd_attack import LinfPGDAttack
from my_eval import my_eval
from model import Model

_DAT_AUG = False

with open('config.json') as config_file:
    config = json.load(config_file)
    config_json = config


if config['data_path'] == 'cifar10_data':
 npy_dir = 'robust_CIFAR_10_feats.npy'
 import cifar10_input
elif config['data_path'] == 'cifar100_data':
 import cifar100_input
 npy_dir = 'robust_CIFAR_100_feat_reps.npy'

# seeding randomness
tf.set_random_seed(config['tf_random_seed'])
np.random.seed(config['np_random_seed'])

# Setting up training parameters
max_num_training_steps = config['max_num_training_steps']
num_output_steps = config['num_output_steps']
num_checkpoint_steps = config['num_checkpoint_steps']
step_size_schedule = config['step_size_schedule']
weight_decay = config['weight_decay']
data_path = config['data_path']
momentum = config['momentum']
batch_size = config['training_batch_size']
feat_sim_pen_val = config['feat_sim']
warmstart_step = config['warmstart_step']

# Setting up the data and the model
if config['data_path'] == 'cifar10_data':
  raw_cifar = cifar10_input.CIFAR10Data(data_path)
elif config['data_path'] == 'cifar100_data':
  raw_cifar = cifar100_input.CIFAR100Data(data_path)
global_step = tf.contrib.framework.get_or_create_global_step()

if config['data_path'] == 'cifar10_data':
  model = Model(mode='eval', class_count=10)
elif config['data_path'] == 'cifar100_data':
  model = Model(mode='eval', class_count=100)


# LWF loss on feat reps
model_feat_reps = model.penultimate
feat_pl = tf.placeholder(tf.float32)
feat_sim_pen = tf.constant(feat_sim_pen_val)
feat_sim_loss = tf.multiply(feat_sim_pen,tf.reduce_mean(tf.norm(model_feat_reps-feat_pl,axis=1,ord=1)))

# Setting up the optimizer
boundaries = [int(sss[0]) for sss in step_size_schedule]
boundaries = boundaries[1:]
values = [sss[1] for sss in step_size_schedule]
learning_rate = tf.train.piecewise_constant(
    tf.cast(global_step, tf.int32),
    boundaries,
    values)
total_loss = model.mean_xent + weight_decay * model.weight_decay_loss + feat_sim_loss

with tf.variable_scope('optimizer_last'):
  train_optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
  train_step_last = train_optimizer.minimize(
    total_loss,
    var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='logit'),
    global_step=global_step)

with tf.variable_scope('optimizer_all_vars'):
  train_optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
  train_step_all = train_optimizer.minimize(
    total_loss,
    var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES),
    global_step=global_step)

# Set up adversary to be used for evaluation while training
attack = LinfPGDAttack(model,
                       config['epsilon'],
                       config['num_steps'],
                       config['step_size'],
                       config['random_start'],
                       config['loss_func'])

# Setting up the Tensorboard and checkpoint outputs
model_dir = config['model_dir']
if not os.path.exists(model_dir):
  os.makedirs(model_dir)


saver_for_saving = tf.train.Saver(max_to_keep=2)
train_vars_last = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='logit')
optimizer_vars_last = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='optimizer_last')
optimizer_vars_all = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='optimizer_all')
fixed_vars = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if v not in train_vars_last and v not in optimizer_vars_last and v not in optimizer_vars_all]
saver = tf.train.Saver(max_to_keep=2, var_list=fixed_vars)


# keep the configuration file with the model for reproducibility
shutil.copy('config.json', model_dir)

config = tf.ConfigProto()
config.gpu_options.allow_growth = False
with tf.Session(config=config) as sess:

  if _DAT_AUG:
    # initialize data augmentation
    if config_json['data_path'] == 'cifar10_data':
      cifar = cifar10_input.AugmentedCIFAR10Data(raw_cifar, sess, model)
    elif config_json['data_path'] == 'cifar100_data':
      cifar = cifar100_input.AugmentedCIFAR100Data(raw_cifar, sess, model)
  else:
    cifar = raw_cifar


  saver.restore(sess, tf.train.latest_checkpoint(config_json['pretrained_model_dir']))
  sess.run([v.initializer for v in train_vars_last])
  sess.run([v.initializer for v in optimizer_vars_last])
  sess.run([v.initializer for v in optimizer_vars_all])
  sess.run(global_step.initializer)

  print('done loading model')
  feats_dir = os.path.join(data_path, npy_dir)
  if os.path.exists(feats_dir):
    print('the feature representations already exist ... moving on to training')
    sys.stdout.flush()
  else:
    # go over all the data and store their feature representations 
    def get_start_end(bid,bs,lendat):
     start = bid*bs
     end = min(start+bs, lendat)   
     return start, end
    print('saving feature representations')
    sys.stdout.flush()
    n_train = raw_cifar.train_data.n
    all_raw_train_xs = raw_cifar.train_data.xs
    import math
    n_b_train = int(math.ceil(n_train/batch_size))
    for jj in tqdm(range(n_b_train)):
      start, end = get_start_end(jj,batch_size,n_train)
      these_feats = sess.run(model_feat_reps, feed_dict={model.x_input: all_raw_train_xs[start:end]})
      if jj == 0:
        all_feats = these_feats
      else:
        all_feats = np.vstack((all_feats,these_feats))
    all_feats = all_feats.reshape(-1, 640) 
    np.save(feats_dir,all_feats)
    print('saved all feat reps')
    sys.stdout.flush()
  
  # Main training loop
  for ii in range(1, max_num_training_steps+1):
    x_batch, y_batch, ft_batch = cifar.train_data.get_next_batch(batch_size,
                                                       multiple_passes=True)



    nat_dict = {model.x_input: x_batch,
                model.y_input: y_batch,
                feat_pl: ft_batch}


    # Output to stdout
    if ii % num_output_steps == 0:
      nat_acc, nat_loss, nat_xent, fsm = sess.run([model.accuracy, total_loss, model.mean_xent, feat_sim_loss], feed_dict=nat_dict)
      print('Step {}:    ({})'.format(ii, datetime.now()))
      print('    training nat accuracy {:.4}% - total loss {:4} | xent loss {:4} - feat_sim {:4}'.format(nat_acc * 100,nat_loss, nat_xent, fsm/feat_sim_pen_val))
      sys.stdout.flush()


    # Write a checkpoint
    if ii % num_checkpoint_steps == 0 and ii != 0:
      saver_for_saving.save(sess,
                 os.path.join(model_dir, 'checkpoint'),
                 global_step=global_step)

      if ii == max_num_training_steps:
        print('Results of Eval Data:')
        my_eval(config=config_json, cifar=raw_cifar, model=model, attack=attack, sess=sess, source='eval_data')

    # Actual training step
    if ii <= warmstart_step: 
      sess.run(train_step_last, feed_dict=nat_dict)
    else:
      sess.run(train_step_all, feed_dict=nat_dict)

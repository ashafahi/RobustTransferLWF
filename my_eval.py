import tensorflow as tf
import json
import math

from model import Model
from pgd_attack import LinfPGDAttack

def my_eval(config, cifar, model, attack, sess, source):
	#======
	total_loss = model.mean_xent + config['weight_decay'] * model.weight_decay_loss

	num_eval_examples = config['num_eval_examples']
	eval_batch_size = config['eval_batch_size']
	num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
	total_xent_nat = 0.
	total_xent_adv = 0.
	total_corr_nat = 0
	total_corr_adv = 0

	total_total_nat = 0.
	total_total_adv = 0.

	for ibatch in range(num_batches):
		bstart = ibatch * eval_batch_size
		bend = min(bstart + eval_batch_size, num_eval_examples)

		if source == 'eval_data':
			x_batch = cifar.eval_data.xs[bstart:bend, :]
			y_batch = cifar.eval_data.ys[bstart:bend]
		elif source == 'train_data':
			x_batch = cifar.train_data.xs[bstart:bend, :]
			y_batch = cifar.train_data.ys[bstart:bend]

		dict_nat = {model.x_input: x_batch, model.y_input: y_batch}

		x_batch_adv = attack.perturb(x_batch, y_batch, sess)

		dict_adv = {model.x_input: x_batch_adv, model.y_input: y_batch}

		cur_corr_nat, cur_xent_nat = sess.run(
			[model.num_correct,model.xent],
			feed_dict = dict_nat)
		cur_corr_adv, cur_xent_adv = sess.run(
			[model.num_correct,model.xent],
			feed_dict = dict_adv)

		cur_total_nat = sess.run(total_loss, feed_dict = dict_nat)
		cur_total_adv = sess.run(total_loss, feed_dict = dict_adv)

		# print("batch {}/{} size: {}".format(ibatch, num_batches, eval_batch_size))
		# print("Correctly classified natural examples: {}".format(cur_corr_nat))
		# print("Correctly classified adversarial examples: {}".format(cur_corr_adv))
		total_xent_nat += cur_xent_nat
		total_xent_adv += cur_xent_adv
		total_corr_nat += cur_corr_nat
		total_corr_adv += cur_corr_adv

		total_total_nat += cur_total_nat
		total_total_adv += cur_total_adv

	avg_xent_nat = total_xent_nat / float(num_eval_examples)
	avg_xent_adv = total_xent_adv / float(num_eval_examples)
	acc_nat = total_corr_nat / float(num_eval_examples)
	acc_adv = total_corr_adv / float(num_eval_examples)

	avg_total_nat = total_total_nat / float(num_batches)
	avg_total_adv = total_total_adv / float(num_batches)


	print('natural: {:.2f}%'.format(100 * acc_nat))
	print('adversarial: {:.2f}%'.format(100 * acc_adv))
	print('avg nat loss: {:.4f}'.format(avg_xent_nat))
	print('avg adv loss: {:.4f}'.format(avg_xent_adv))
	print('avg nat total loss: {:.4f}'.format(avg_total_nat))
	print('avg adv total loss: {:.4f}'.format(avg_total_adv))

	#======

if __name__=='__main__':
	with open('config.json') as config_file:
		config = json.load(config_file)
	if config['data_path'] == "cifar100_data":
		mode = 'cifar_100'
	elif config['data_path'] == "cifar10_data":
		mode = 'cifar_10'
	else:
		print("mode should be either cifar_10 ro cifar_100")
		exit(1)
	if mode == 'cifar_10':
		import cifar10_input
	elif mode == 'cifar_100':
		import cifar100_input
	if mode == 'cifar_10':
		cifar = cifar10_input.CIFAR10Data(config['data_path'])
	elif mode == 'cifar_100':
		cifar = cifar100_input.CIFAR100Data(config['data_path'])
	if mode == 'cifar_10':
		model = Model(mode='eval', class_count=10)
	elif mode == 'cifar_100':
		model = Model(mode='eval', class_count=100)
	attack = LinfPGDAttack(model,
		config['epsilon'],
		config['num_steps'],
		config['step_size'],
		config['random_start'],
		config['loss_func'])
	saver = tf.train.Saver()
	with tf.Session() as sess:
		saver.restore(sess, tf.train.latest_checkpoint(config['pretrained_model_dir']))
		# print('Results of Eval Data:')
		# my_eval(config=config, cifar=cifar, model=model, attack=attack, sess=sess, source='eval_data')
		print('Results of Train Data:')
		my_eval(config=config, cifar=cifar, model=model, attack=attack, sess=sess, source='train_data')





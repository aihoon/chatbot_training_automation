#! /usr/bin/env python
# -*- coding: UTF-8 -*-
# vi:ts=4:tw=78:shiftwidth=4
# vim600:fdm=marker

from collections import OrderedDict
import copy
import cPickle, gzip
import os
import urllib
import random
import stat
import subprocess
import sys, time
import numpy

import theano
from theano import tensor as T
from compiler.ast import flatten

# utils functions
def shuffle(lol, seed):
	'''
	lol :: list of list as input
	seed :: seed the shuffling
	shuffle inplace each list in the same order
	'''
	for l in lol:
		random.seed(seed)
		random.shuffle(l)

# return numpy (n_steps * batch_size)
def get_minibatch(train_x, train_y, index, batch_size, src_padding_id, tgt_padding_id):
	min_x_len = min_y_len = 100000000
	max_x_len = max_y_len = 0
	end_index = min(len(train_x), index+batch_size)
	train_x_batch = train_x[index:end_index]
	train_y_batch = train_y[index:end_index]
	for x in train_x_batch:
		min_x_len = min(min_x_len, len(x))
		max_x_len = max(max_x_len, len(x))
	for y in train_y_batch:
		min_y_len = min(min_y_len, len(y))
		max_y_len = max(max_y_len, len(y))
	#print 'min_x_len:', min_x_len, 'min_y_len:', min_y_len
	#print 'max_x_len:', max_x_len, 'max_y_len:', max_y_len
	# resize to (max_x_len, max_y_len) (fill -1)
	mask_x = []
	mask_y = []
	for x in train_x_batch:
		mask_x_i = [1] * len(x)
		while len(x) < max_x_len:
			x.append(src_padding_id)
			mask_x_i.append(0)
		mask_x.append(mask_x_i)
	for y in train_y_batch:
		mask_y_i = [1] * len(y)
		while len(y) < max_y_len:
			y.append(tgt_padding_id)
			mask_y_i.append(0)
		mask_y.append(mask_y_i)
	# convert numpy
	train_x_batch = numpy.asarray(train_x_batch).astype('int32')
	train_y_batch = numpy.asarray(train_y_batch).astype('int32')
	mask_x = numpy.asarray(mask_x).astype('float32')
	mask_y = numpy.asarray(mask_y).astype('float32')
	train_x_batch = train_x_batch.T
	train_y_batch = train_y_batch.T
	mask_x = mask_x.T
	mask_y = mask_y.T
	#print 'train_x_batch:', train_x_batch.shape
	#print 'train_y_batch:', train_y_batch.shape
	#print 'mask_y:', mask_y.shape
	return train_x_batch, train_y_batch, mask_x, mask_y

def contextwin(l, win):
	'''
	win :: int corresponding to the size of the window
	given a list of indexes composing a sentence
	it will return a list of list of indexes corresponding
	to context windows surrounding each word in the sentence
	'''
	assert (win % 2) == 1
	assert win >=1
	l = list(l)

	lpadded = win/2 * [-1] + l + win/2 * [-1]
	out = [ lpadded[i:i+win] for i in range(len(l)) ]

	assert len(out) == len(l)
	return out

# by leeck for LM
def left_context_win(l, padding_id, win):
	'''
	win :: int corresponding to the size of the window
	given a list of indexes composing a sentence
	it will return a list of list of indexes corresponding
	to context windows surrounding each word in the sentence
	'''
	assert win >=1
	l = list(l)
	lpadded = (win-1) * [padding_id] + l
	out = [ lpadded[i:i+win] for i in range(len(l)) ]
	assert len(out) == len(l)
	return out

# accuracy - by leeck
def get_accuracy(p, g):
	'''
	INPUT:
	p :: predictions
	g :: groundtruth
	OUTPUT:
	accuracy
	'''
	correct = total = 0
	for (ans_sent, sys_sent) in zip(g, p):
		for ans, sys in zip(ans_sent, sys_sent):
			if ans == sys: correct += 1
			total += 1
			if ans == '</s>': break
	return 100.0 * correct / total

# metrics function using conlleval.pl
def conlleval(p, g, w, filename):
	'''
	INPUT:
	p :: predictions
	g :: groundtruth
	w :: corresponding words
	OUTPUT:
	filename :: name of the file where the predictions
	are written. it will be the input of conlleval.pl script
	for computing the performance in terms of precision
	recall and f1 score
	'''
	out = ''
	for sl, sp, sw in zip(g, p, w):
		#out += 'BOS O O\n'
		for wl, wp, w in zip(sl, sp, sw):
			out += w + ' ' + wl + ' ' + wp + '\n'
		#out += 'EOS O O\n\n'
		out += '\n'

	f = open(filename, 'w')
	f.writelines(out)
	f.close()
	return get_perf(filename)

def get_perf(filename):
	''' run conlleval.pl perl script to obtain
	precision/recall and F1 score '''
	_conlleval = 'conlleval.pl'
	proc = subprocess.Popen(["perl", _conlleval],
							stdin=subprocess.PIPE,
							stdout=subprocess.PIPE)
	stdout, _ = proc.communicate(''.join(open(filename).readlines()))
	for line in stdout.split('\n'):
		if 'accuracy' in line:
			out = line.split()
			break

	precision = float(out[6][:-2])
	recall = float(out[8][:-2])
	f1score = float(out[10])
	return {'p': precision, 'r': recall, 'f1': f1score}

def dropout_from_layer(layer, p):
	""" p is the probablity of dropping a unit """
	rng = numpy.random.RandomState(3435)
	srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))
	# p=1-p because 1's indicate keep and p is prob of dropping
	mask = srng.binomial(n=1, p=1-p, size=layer.shape)
	# The cast is important because
	# int * float32 = float64 which pulls things off the gpu
	output = layer * T.cast(mask, theano.config.floatX)
	return output

def shared_dataset(data_xy, borrow=True):
	""" Function that loads the dataset into shared variables
	The reason we store our dataset in shared variables is to allow
	Theano to copy it into the GPU memory (when code is run on GPU).
	Since copying data into the GPU is slow, copying a minibatch everytime
	is needed (the default behaviour if the data is not in a shared
	variable) would lead to a large decrease in performance.
	"""
	data_x, data_y = data_xy
	shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX),
							 borrow=borrow)
	shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX),
							 borrow=borrow)
	return shared_x, T.cast(shared_y, 'int32')

def as_floatX(variable):
	if isinstance(variable, float):
		return numpy.cast[theano.config.floatX](variable)

	if isinstance(variable, numpy.ndarray):
		return numpy.cast[theano.config.floatX](variable)
	return theano.tensor.cast(variable, theano.config.floatX)

def sgd_updates(params, cost, learning_rate=0.1):
	updates = []
	for param in params:
		grad = T.grad(cost, param)
		updates.append((param, param - learning_rate*grad))
	return updates

def sgd_updates_with_clipping(params, cost, learning_rate=0.1, clipping_value=9, exclude_params=[]):
	updates = []
	for param in params:
		grad = T.grad(cost, param)
		if param in exclude_params:
			updates.append((param, param - learning_rate*grad))
		else:
			# clipping gradient
			norm = T.sqrt(T.sum(grad ** 2))
			grad2 = T.switch(T.ge(norm, clipping_value), grad / norm * clipping_value, grad)
			updates.append((param, param - learning_rate*grad2))
	return updates

def sgd_updates_momentum(params, cost, learning_rate=0.1, momentum=0.9):
	'''
	Compute updates for gradient descent with momentum
	:parameters:
		- cost : Theano cost function to minimize
		- params : Parameters to compute gradient against
		- learning_rate : Gradient descent learning rate
		- momentum : Momentum parameter, should be at least 0 (standard gradient descent) and less than 1
	:returns:
		updates : List of updates, one for each parameter
	'''
	assert momentum < 1 and momentum >= 0
	# List of update steps for each parameter
	updates = []
	# Just gradient descent on cost
	for param in params:
		# For each parameter, we'll create a param_update shared variable.
		# This variable will keep track of the parameter's update step across iterations.
		# We initialize it to 0
		param_update = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
		# Each parameter is updated by taking a step in the direction of the gradient.
		# However, we also "mix in" the previous step according to the given momentum value.
		# Note that when updating param_update, we are using its old value and also the new gradient step.
		updates.append((param, param - learning_rate*param_update))
		# Note that we don't need to derive backpropagation to compute updates - just use T.grad!
		g = T.grad(cost, param)
		# gradient clipping
		if False:
			if param.name != 'embeddings' and param.name != 'f_embeddings' \
				and param.name != 'emb_src' and param.name != 'emb_tgt':
				norms = T.sqrt(T.sum(g ** 2))
				desired_norms = T.clip(norms, 0, 10)
				scale = desired_norms / (1e-7 + norms)
				g = g * scale

		updates.append((param_update, momentum*param_update + (1.-momentum)*g))
	return updates

def sgd_updates_momentum_with_clipping(params, cost, learning_rate=0.1, momentum=0.9, clipping_value=9, exclude_params=[]):
	assert momentum < 1 and momentum >= 0
	updates = []
	for param in params:
		param_update = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
		updates.append((param, param - learning_rate*param_update))
		grad = T.grad(cost, param)
		if param in exclude_params:
			updates.append((param_update, momentum*param_update + (1.-momentum)*grad))
		else:
			# clipping gradient
			norm = T.sqrt(T.sum(grad ** 2))
			grad2 = T.switch(T.ge(norm, clipping_value), grad / norm * clipping_value, grad)
			updates.append((param_update, momentum*param_update + (1.-momentum)*grad2))
	return updates

def sgd_updates_rmsprop(params, cost, learning_rate=1.0, rho=0.9, epsilon=1e-6):
 	"""
	RMSProp updates
	Scale learning rates by dividing with the moving average of the root mean
	squared (RMS) gradients. See [1]_ for further description.
	Notes
	-----
	`rho` should be between 0 and 1. A value of `rho` close to 1 will decay the
	moving average slowly and a value close to 0 will decay the moving average
	fast.
	References
	----------
	.. [1] Tieleman, T. and Hinton, G. (2012):
		   Neural Networks for Machine Learning, Lecture 6.5 - rmsprop.
		   Coursera. http://www.youtube.com/watch?v=O3sxAc4hxZU (formula @5:20)
	"""
	grads = T.grad(cost, params)
	updates = OrderedDict()
	for param, grad in zip(params, grads):
		value = param.get_value(borrow=True)
		accu = theano.shared(numpy.zeros(value.shape, dtype=value.dtype),
				broadcastable=param.broadcastable)
		accu_new = rho * accu + (1 - rho) * grad ** 2
		updates[accu] = accu_new
		updates[param] = param - (learning_rate * grad / T.sqrt(accu_new + epsilon))
	return updates

def sgd_updates_adadelta(params,cost,rho=0.95,epsilon=1e-6,norm_lim=9):
	"""
	adadelta update rule, mostly from
	https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
	"""
	updates = OrderedDict({})
	exp_sqr_grads = OrderedDict({})
	exp_sqr_ups = OrderedDict({})
	gparams = []
	for param in params:
		empty = numpy.zeros_like(param.get_value())
		exp_sqr_grads[param] = theano.shared(value=as_floatX(empty),name="exp_grad_%s" % param.name)
		gp = T.grad(cost, param)
		exp_sqr_ups[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
		gparams.append(gp)
	for param, gp in zip(params, gparams):
		exp_sg = exp_sqr_grads[param]
		exp_su = exp_sqr_ups[param]
		up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
		updates[exp_sg] = up_exp_sg
		step =  -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
		updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
		stepped_param = param + step
		if (param.get_value(borrow=True).ndim == 2) \
			and (param.name!='embeddings' and param.name!='f_embeddings') \
			and (param.name!='emb_src' and param.name!='emb_tgt'):
			col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
			desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim))
			scale = desired_norms / (1e-7 + col_norms)
			updates[param] = stepped_param * scale
		else:
			updates[param] = stepped_param	  
	return updates


def load_bin_vec1(fname):
	"""
	Loads word vectors from Google (Mikolov) word2vec
	"""
	word_vecs = {}
	with open(fname, "rb") as f:
		header = f.readline()
		vocab_size, layer1_size = map(int, header.split())
		binary_len = numpy.dtype('float32').itemsize * layer1_size
		for line in xrange(vocab_size):
			word = []
			while True:
				ch = f.read(1)
				if ch == ' ':
					word = ''.join(word)
					break
				if ch != '\n':
					word.append(ch)   
			word_vecs[word] = numpy.fromstring(f.read(binary_len), dtype='float32')  
	return word_vecs

def load_bin_vec(fname, vocab):
	"""
	Loads word vectors from Google (Mikolov) word2vec
	"""
	word_vecs = {}
	with open(fname, "rb") as f:
		header = f.readline()
		vocab_size, layer1_size = map(int, header.split())
		binary_len = numpy.dtype('float32').itemsize * layer1_size
		for line in xrange(vocab_size):
			word = []
			while True:
				ch = f.read(1)
				if ch == ' ':
					word = ''.join(word)
					break
				if ch != '\n':
					word.append(ch)   
			if word in vocab:
				word_vecs[word] = numpy.fromstring(f.read(binary_len), dtype='float32')  
			else:
				f.read(binary_len)
	return word_vecs

def load_txt_vec(fname, vocab):
	"""
	Loads word vectors from Google (Mikolov) word2vec
	"""
	word_vecs = {}
	with open(fname, "r") as f:
		for line in f:
			words = line.split()
			key = words[0]
			values = map(float, words[1:])
			if key in vocab:
				word_vecs[key] = numpy.asarray(values)
	return word_vecs

# for GRU, LSTM
def slice(x, n, dim):
	if x.ndim == 3: return x[:, :, n*dim:(n+1)*dim]
	elif x.ndim == 2: return x[:, n*dim:(n+1)*dim]
	return x[n*dim:(n+1)*dim]

# my sotfmax - because of Nan error --> error
def mysoftmax(x):
	if x.ndim == 1:
		max_x = T.max(x, keepdims=True)
		e_x = T.exp(x - max_x)
		return e_x / e_x.sum(keepdims=True)
	max_x = T.max(x, axis=1, keepdims=True)
	e_x = T.exp(x - max_x)
	return e_x / e_x.sum(axis=1, keepdims=True)

def activation(act_type, x):
	if act_type == 'sigm':
		return T.nnet.sigmoid(x)
	elif act_type == 'tanh':
	   	return T.tanh(x)
	elif act_type == 'relu':
		return T.maximum(0.0, x)
	else:
		print 'Error:', activation
		sys.exit(1)

def logsumexp(x, axis=None):
	"""
	Compute log(sum(exp(x), axis=axis) in a numerically stable fashion.
	x : A Theano tensor (any dimension will do).
	axis : int or symbolic integer scalar, or None
		Axis over which to perform the summation.
		`None`, the default, performs over all axes.
	result : ndarray or scalar
		The result of the log(sum(exp(...))) operation.
	"""
	xmax = x.max(axis=axis, keepdims=True)
	xmax_ = x.max(axis=axis)
	return xmax_ + T.log(T.exp(x - xmax).sum(axis=axis))

def get_ans_score(obs_potentials, chain_potentials, ans_labels):
	"""
	obs_potentials : tensor_like (n_steps, n_classes)
		Axes correspond to time and the value of the discrete label variable.
	chain_potentials : tensor_like (n_classes, n_classes)
		Axes correspond to left label state and current label state.
	ans_labels : tensor_like (n_steps)
	Returns : TensorVariable, 0-dimensional The score assigned for a given label.
	"""
	def inner_function(obs, prev_label, cur_label, prior_result, chain_potentials):
		result = prior_result + obs[cur_label] + chain_potentials[prev_label, cur_label]
		return result

	assert obs_potentials.ndim == 2
	assert chain_potentials.ndim == 2
	assert ans_labels.ndim == 1
	ans_score, _ = theano.scan(fn=inner_function,
							outputs_info=obs_potentials[0][ans_labels[0]],
							sequences=[obs_potentials[1:], ans_labels[:-1], ans_labels[1:]],
							non_sequences=chain_potentials)
	return ans_score[-1]

def forward(obs_potentials, chain_potentials):
	"""
	obs_potentials : tensor_like (n_steps, n_classes)
		Axes correspond to time and the value of the discrete label variable.
	chain_potentials : tensor_like (n_classes, n_classes)
		Axes correspond to left label state and current label state.
	Returns : The score for all possible labels.
	"""
	def inner_function(obs, prior_result, chain_potentials):
		prior_result = prior_result.dimshuffle(0, 'x')
		obs = obs.dimshuffle('x', 0)
		result = logsumexp(prior_result + obs + chain_potentials, axis=0)
		return result

	score, _ = theano.scan(fn=inner_function,
							outputs_info=[obs_potentials[0]],
							sequences=[obs_potentials[1:]],
							non_sequences=chain_potentials)
	return logsumexp(score[-1], axis=0)

def viterbi(obs_potentials, chain_potentials):
	"""
	obs_potentials : tensor_like (n_steps, n_classes)
		Axes correspond to time and the value of the discrete label variable.
	chain_potentials : tensor_like (n_classes, n_classes)
		Axes correspond to left label state and right label state.
	Returns : TensorVariable, 1-dimensional label array.
	"""
	def inner_function(obs, prior_result, prev_path):
		prior_result = prior_result.dimshuffle(0, 'x')
		obs = obs.dimshuffle('x', 0)
		result = (prior_result + obs + chain_potentials).max(axis=0)
		path = (prior_result + obs + chain_potentials).argmax(axis=0)
		return [result, path]

	[score, path], _ = theano.scan(fn=inner_function,
							outputs_info=[obs_potentials[0], None],
							sequences=[obs_potentials[1:]],
							non_sequences=chain_potentials)
	max_score = score[-1].max(axis=0)
	max_y = score[-1].argmax(axis=0)
	reverse_path = path[::-1,:]

	def get_best_path(cur_path, next_pred):
		cur_pred = cur_path[next_pred]
		return cur_pred
	reverse_path, _ = theano.scan(fn=get_best_path,
							outputs_info=max_y,
							sequences=reverse_path)
	best_path = reverse_path[::-1]
	return best_path


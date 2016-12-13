#! /usr/bin/env python
# -*- coding: UTF-8 -*-
# vi:ts=4:shiftwidth=4
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
import myutil
import math

import theano
from theano import tensor as T
from compiler.ast import flatten

class GRU_encdec(object):
	''' Gated Recurrent Unit RNN based encoder-decoder model '''
	def build_param(self, hyper_param, word2idx_dic):
		nh = hyper_param['nhidden']
		ne = hyper_param['vocsize']
		de = hyper_param['emb_dimension']
		# parameters of the model
		embedding = self.load_embedding(hyper_param['emb_file'], word2idx_dic, ne, de)
	   	self.emb = theano.shared(name='emb', value=embedding)
		# Wg : Wz, Wr
		self.Wgx_src = theano.shared(name='Wgx_src', value=0.01 * numpy.random.randn(de, 2*nh).astype(theano.config.floatX))
		self.Wgx = theano.shared(name='Wgx', value=0.01 * numpy.random.randn(de, 2*nh).astype(theano.config.floatX))
		self.Wgc = theano.shared(name='Wgc', value=0.01 * numpy.random.randn(nh, 2*nh).astype(theano.config.floatX))
		self.Whx_src = theano.shared(name='Whx_src', value=0.01 * numpy.random.randn(de, nh).astype(theano.config.floatX))
		self.Whx = theano.shared(name='Whx', value=0.01 * numpy.random.randn(de, nh).astype(theano.config.floatX))
		self.Whc = theano.shared(name='Whc', value=0.01 * numpy.random.randn(nh, nh).astype(theano.config.floatX))
		self.Wh0c = theano.shared(name='Wh0c', value=0.01 * numpy.random.randn(nh, nh).astype(theano.config.floatX))
		# Ug : Uz, Ur
		identity1 = self.identity_weight(nh, nh)
		identity2 = self.identity_weight(nh, 2*nh)
		self.Ugh_src = theano.shared(name='Ugh_src', value=identity2.astype(theano.config.floatX))
		self.Ugh = theano.shared(name='Ugh', value=identity2.astype(theano.config.floatX))
		self.Uhh_src = theano.shared(name='Uhh_src', value=identity1.astype(theano.config.floatX))
		self.Uhh = theano.shared(name='Uhh', value=identity1.astype(theano.config.floatX))
		# bg : bz, br
		self.bg_src = theano.shared(name='bg_src', value=numpy.zeros(2*nh, dtype=theano.config.floatX))
		self.bg = theano.shared(name='bg', value=numpy.zeros(2*nh, dtype=theano.config.floatX))
		self.bh_src = theano.shared(name='bh_src', value=numpy.zeros(nh, dtype=theano.config.floatX))
		self.bh = theano.shared(name='bh', value=numpy.zeros(nh, dtype=theano.config.floatX))
		self.bh0 = theano.shared(name='bh0', value=numpy.zeros(nh, dtype=theano.config.floatX))
		# others
		self.Wyh = theano.shared(name='Wyh', value=0.01 * numpy.random.randn(nh, ne).astype(theano.config.floatX))
		self.Wyc = theano.shared(name='Wyc', value=0.01 * numpy.random.randn(nh, ne).astype(theano.config.floatX))
		self.Wyy = theano.shared(name='Wyy', value=0.01 * numpy.random.randn(de, ne).astype(theano.config.floatX))
		self.by = theano.shared(name='by', value=numpy.zeros(ne, dtype=theano.config.floatX))
		self.h0_src = theano.shared(name='h0_src', value=numpy.zeros(nh, dtype=theano.config.floatX))
	
	def load_param(self, hyper_param, model_name):
		nh = hyper_param['nhidden']
		ne = hyper_param['vocsize']
		de = hyper_param['emb_dimension']
		print 'loading previous model:', model_name, '...',
		#f = open(model_name, 'rb')
		f = gzip.open(model_name, 'rb')
		[numpy_names, numpy_params] = cPickle.load(f)
		f.close()
		for numpy_param, name in zip(numpy_params, numpy_names):
			print name,
			if name == 'emb': self.emb = theano.shared(name=name, value=numpy_param)
			elif name == 'Wgx_src': self.Wgx_src = theano.shared(name=name, value=numpy_param)
			elif name == 'Wgx': self.Wgx = theano.shared(name=name, value=numpy_param)
			elif name == 'Wgc': self.Wgc = theano.shared(name=name, value=numpy_param)
			elif name == 'Whx_src': self.Whx_src = theano.shared(name=name, value=numpy_param)
			elif name == 'Whx': self.Whx = theano.shared(name=name, value=numpy_param)
			elif name == 'Whc': self.Whc = theano.shared(name=name, value=numpy_param)
			elif name == 'Wh0c': self.Wh0c = theano.shared(name=name, value=numpy_param)
			elif name == 'Ugh_src': self.Ugh_src = theano.shared(name=name, value=numpy_param)
			elif name == 'Ugh': self.Ugh = theano.shared(name=name, value=numpy_param)
			elif name == 'Uhh_src': self.Uhh_src = theano.shared(name=name, value=numpy_param)
			elif name == 'Uhh': self.Uhh = theano.shared(name=name, value=numpy_param)
			elif name == 'Wyh': self.Wyh = theano.shared(name=name, value=numpy_param)
			elif name == 'Wyc': self.Wyc = theano.shared(name=name, value=numpy_param)
			elif name == 'Wyy': self.Wyy = theano.shared(name=name, value=numpy_param)
			elif name == 'bg_src': self.bg_src = theano.shared(name=name, value=numpy_param)
			elif name == 'bg': self.bg = theano.shared(name=name, value=numpy_param)
			elif name == 'bh_src': self.bh_src = theano.shared(name=name, value=numpy_param)
			elif name == 'bh': self.bh = theano.shared(name=name, value=numpy_param)
			elif name == 'bh0': self.bh0 = theano.shared(name=name, value=numpy_param)
			elif name == 'by': self.by = theano.shared(name=name, value=numpy_param)
			elif name == 'h0_src': self.h0_src = theano.shared(name=name, value=numpy_param)
			else: print 'skip:', name
		if 'Wgc' not in numpy_names:
			print 'random: Wgc'
			self.Wgc = theano.shared(name='Wgc', value=0.01 * numpy.random.randn(nh, 2*nh).astype(theano.config.floatX))
		if 'Wh0c' not in numpy_names:
			print 'random: Wh0c'
			self.Wh0c = theano.shared(name='Wh0c', value=0.01 * numpy.random.randn(nh, nh).astype(theano.config.floatX))
		if 'bh0' not in numpy_names:
			print 'random: bh0'
			self.bh0 = theano.shared(name='bh0', value=numpy.zeros(nh, dtype=theano.config.floatX))
		print 'done.'

	def __init__(self, hyper_param, word2idx_dic):
		'''
		nh :: dimension of the hidden layer
		ne :: number of word embeddings in the vocabulary
		de :: dimension of the word embeddings
		nf :: number of feature
		nfe:: number of feature embeddings in the vocabulary - by leeck
		dfe:: dimension of the feature embeddings - by leeck
		cs :: word window context size
		emb_file :: word embedding file
		weight_decay :: weight decay
		dropout_rate :: dropout rate
		activation :: activation function: simg, tanh, relu
		word2idx_dic :: word to index dictionary
		'''
		self.hyper_param = hyper_param
		nh = hyper_param['nhidden']
		ne = hyper_param['vocsize']
		de = hyper_param['emb_dimension']
		weight_decay = hyper_param['weight_decay']
		dropout_rate = hyper_param['dropout_rate']
		activation = hyper_param['activation']
		learning_method = hyper_param['learning_method']
		verbose = False
		# parameters of the model
		if hyper_param['load_model'] != '':
			self.load_param(hyper_param, hyper_param['load_model'])
		else:
			self.build_param(hyper_param, word2idx_dic)

		# parameters
		self.params = [self.emb, self.Wgx_src, self.Wgx, self.Wgc, \
				self.Whx_src, self.Whx, self.Whc, self.Wh0c, self.Ugh_src, self.Ugh, \
				self.Uhh_src, self.Uhh, self.Wyh, self.Wyc, self.Wyy, \
				self.bg_src, self.bg, self.bh_src, self.bh, self.bh0, self.by, \
				self.h0_src]

		if hyper_param['fixed_emb']:
			print 'fixed embeddig.'
			self.params.remove(self.emb)

		# as many columns as context window size
		# as many lines as words in the sentence
		x_sentence = T.ivector('x_sentence') # x_sentence : n_steps
		x_org = self.emb[x_sentence].reshape((x_sentence.shape[0], de))
		x = x_org[:-1] # remove '</s>'
		x_reverse = x[::-1] # reverse for backward

		y_sentence = T.ivector('y_sentence') # labels
		y_input_sentence = T.concatenate([y_sentence[-1:], y_sentence[:-1]], axis=0)
		y = self.emb[y_input_sentence].reshape((y_input_sentence.shape[0], de))

		# for scan
		def source_step(x_t, h_tm1):
			#print 'z_t and r_t are combined!'
			all_t = T.nnet.sigmoid(T.dot(x_t, self.Wgx_src) + T.dot(h_tm1, self.Ugh_src) + self.bg_src)
			z_t = myutil.slice(all_t, 0, nh)
			r_t = myutil.slice(all_t, 1, nh)
			# candidate h_t
			ch_t = myutil.activation(activation, T.dot(x_t, self.Whx_src) + T.dot(r_t * h_tm1, self.Uhh_src) + self.bh_src)
			h_t = (1.0 - z_t) * h_tm1 + z_t * ch_t
			return h_t

		def target_step(x_t, h_tm1, c):
			#print 'z_t and r_t are combined!'
			all_t = T.nnet.sigmoid(T.dot(x_t, self.Wgx) + T.dot(h_tm1, self.Ugh) + T.dot(c, self.Wgc) + self.bg)
			z_t = myutil.slice(all_t, 0, nh)
			r_t = myutil.slice(all_t, 1, nh)
			# candidate h_t
			ch_t = myutil.activation(activation, T.dot(x_t, self.Whx) + T.dot(r_t * h_tm1, self.Uhh) + T.dot(c, self.Whc) + self.bh)
			h_t = (1.0 - z_t) * h_tm1 + z_t * ch_t
			return h_t

		# make score, h_src, h0 (for beam search)
		def make_score(x, y, use_noise):
			# input layer dropout: ex. [0.2, 0.2, 0.5]
			if use_noise:
				print "X's projection layer dropout:", dropout_rate[0]
				dropout_x = myutil.dropout_from_layer(x, dropout_rate[0])
			else:
				dropout_x = x * (1.0 - dropout_rate[0])
			# recurrent for source language
			h_src, _ = theano.scan(fn=source_step,
								sequences=dropout_x,
								outputs_info=self.h0_src,
								n_steps=dropout_x.shape[0])
			# context
			c = h_src[-1]
			h0 = myutil.activation(activation, T.dot(c, self.Wh0c) + self.bh0)
			# output layer dropout: ex. [0.2, 0.2, 0.5]
			if use_noise:
				print "Y's projection layer dropout:", dropout_rate[1]
				dropout_y = myutil.dropout_from_layer(y, dropout_rate[1])
			else:
				dropout_y = y * (1.0 - dropout_rate[1])
			# forward recurrent for target language
			h, _ = theano.scan(fn=target_step,
								sequences=dropout_y,
								outputs_info=h0,
								non_sequences=[c],
								n_steps=dropout_y.shape[0])
			# hidden layer dropout
			if use_noise:
				print "Y's hidden layer dropout:", dropout_rate[2]
				dropout_h = myutil.dropout_from_layer(h, dropout_rate[2])
			else:
				dropout_h = h * (1.0 - dropout_rate[2])
			# score
			score = T.dot(dropout_h, self.Wyh) + T.dot(dropout_y, self.Wyy) + T.dot(c, self.Wyc) + self.by
			return score, h_src, h0

		# dropout version (for training)
		if 'reverse_input' in hyper_param and hyper_param['reverse_input']:
			print 'reverse input.'
			dropout_score, _, _ = make_score(x_reverse, y, True)
		else:
			dropout_score, _, _ = make_score(x, y, True)
	   	dropout_p_y_given_x = myutil.mysoftmax(dropout_score)
		# scaled version (for prediction)
		if 'reverse_input' in hyper_param and hyper_param['reverse_input']:
			print 'reverse input.'
			score, h_src, h0 = make_score(x_reverse, y, False)
		else:
			score, h_src, h0 = make_score(x, y, False)
	   	p_y_given_x = myutil.mysoftmax(score)

		# prediction
		y_pred = T.argmax(p_y_given_x, axis=1)
		test_nll = -T.mean(T.log(p_y_given_x)[T.arange(y.shape[0]), y_sentence])

		# beam search decoding: input=[c, h_tm1, y_tm1], output=[h_t, log_p_y_t]
		input_h_src = T.fmatrix('input_h_src')
		input_h_tm1 = T.fvector('input_h_tm1')
		input_y_tm1 = T.iscalar('input_y_tm1') # input_y_tm1 == x_t
		x_t = self.emb[input_y_tm1]
		c = input_h_src[-1]
		all_t = T.nnet.sigmoid(T.dot(x_t, self.Wgx) + T.dot(input_h_tm1, self.Ugh) + T.dot(c, self.Wgc) + self.bg)
		z_t = myutil.slice(all_t, 0, nh)
		r_t = myutil.slice(all_t, 1, nh)
		# candidate h_t
		ch_t = myutil.activation(activation, T.dot(x_t, self.Whx) + T.dot(r_t * input_h_tm1, self.Uhh) + T.dot(c, self.Whc) + self.bh)
		h_t = (1.0 - z_t) * input_h_tm1 + z_t * ch_t
		# prediction
		score_y_t = T.dot(h_t, self.Wyh) + T.dot(x_t, self.Wyy) + T.dot(c, self.Wyc) + self.by
	   	max_s = T.max(score_y_t)
		exp_s = T.exp(score_y_t - max_s)
		log_p_y_t = T.log(exp_s / exp_s.sum())


		# cost and gradients and learning rate
		lr = T.scalar('lr') # for SGD

		# NLL + L2-norm
	   	nll = -T.mean(T.log(dropout_p_y_given_x)[T.arange(y.shape[0]), y_sentence])
		cost = nll
		for param in self.params:
			if param.name == 'emb':
				continue
			cost += weight_decay * T.sum(param ** 2)

		# SGD
		#gradients = T.grad(cost, self.params)
		#sgd_updates = OrderedDict((p, p - lr*g) for p, g in zip(self.params, gradients))
		sgd_updates = myutil.sgd_updates(self.params, cost, lr)
		# SGD + momentum (0.9)
		momentum_updates = myutil.sgd_updates_momentum(self.params, cost, lr, 0.9)
		# RMSProp (rho = 0.9)
		rmsprop_updates = myutil.sgd_updates_rmsprop(self.params, cost, lr, 0.9, 1)
		# AdaDelta (lr --> rho = 0.95)
		adadelta_updates = myutil.sgd_updates_adadelta(self.params, cost, lr, 1e-6, 9)

		# theano functions to compile
		self.classify = theano.function(inputs=[x_sentence, y_sentence], outputs=[y_pred, test_nll])
		# for beam search
		self.encoding_src_lang = theano.function(inputs=[x_sentence], outputs=[h_src, h0])
		self.search_next_word = theano.function(inputs=[input_h_src, input_h_tm1, input_y_tm1],
				outputs=[log_p_y_t, h_t])
		# for reranking
		self.get_nll = theano.function(inputs=[x_sentence, input_h_src, input_h_tm1, y_sentence],
				outputs=test_nll, on_unused_input='ignore')
		# SGD
		self.train_sgd = theano.function(inputs=[x_sentence, y_sentence, lr], outputs=[cost, nll], updates=sgd_updates)
		# SGD with momentum
		self.train_momentum = theano.function(inputs=[x_sentence, y_sentence, lr], outputs=[cost, nll], updates=momentum_updates)
		# RMSProp
		self.train_rmsprop = theano.function(inputs=[x_sentence, y_sentence, lr], outputs=[cost, nll], updates=rmsprop_updates)
		# AdaDelta
		self.train_adadelta = theano.function(inputs=[x_sentence, y_sentence, lr], outputs=[cost, nll], updates=adadelta_updates)
	
	def beam_search(self, x_sentence, nbest=10, ignore_UNK=False):
		[h_src, h_tm1] = self.encoding_src_lang(x_sentence)
		#print 'h_src:', h_src.shape, 'h_tm1:', h_tm1.shape
		nbest_list = [(0, h_tm1, 1, [])]
		# loop
		for i in xrange(10*len(x_sentence)):
			#print '###', i
			update_flag = False
			new_nbest_list = []
			for score, h_tm1, y_tm1, partial_y_list in nbest_list:
				if i > 0 and y_tm1 == 1: # y == 1 --> '</s>'
					new_nbest_list.append((score, h_tm1, y_tm1, partial_y_list))
				else:
					update_flag = True
					#print 'score:', score
					#print 'partial_y_list:', partial_y_list
					[log_p_y_t, h_t] = self.search_next_word(h_src, h_tm1, y_tm1)
					#print 'log_p_y_t:', log_p_y_t.shape, 'alignment:', alignment.shape
					if ignore_UNK:
						log_p_y_t[0] = -numpy.inf # 0 --> UNK
					for j in xrange(nbest):
						max_y = numpy.argmax(log_p_y_t)
						#print 'max_y:', max_y, 'max_y_score:', log_p_y_t[max_y]
						new_score = score + log_p_y_t[max_y]
						new_partial_y_list = partial_y_list + [max_y]
						new_nbest_list.append((new_score, h_t, max_y, new_partial_y_list))
						log_p_y_t[max_y] = -numpy.inf
						if i >= 3 and j >= nbest/2: break

			new_nbest_list.sort(key=lambda tup: tup[0], reverse=True)
			nbest_list = new_nbest_list[:nbest]
			if not update_flag: break

		nbest_list.sort(key=lambda tup: tup[0], reverse=True)
		# nbest
		nbest_y_list = []
		for score, h_tm1, y_tm1, y_list in nbest_list:
			nbest_y_list.append(y_list)
		# 1-best
		score, h_tm1, y_tm1, y_list = nbest_list[0]
		return y_list, [], nbest_y_list

# ADDED BY EB
	def beam_search_percentage(self, x_sentence, nbest=20, ignore_UNK=False):
		[h_src, h_tm1] = self.encoding_src_lang(x_sentence)
		#print 'h_src:', h_src.shape, 'h_tm1:', h_tm1.shape
		nbest_list = [(0, h_tm1, 1, [])]
		# loop
		for i in xrange(10*len(x_sentence)):
			#print '###', i
			update_flag = False
			new_nbest_list = []
			for score, h_tm1, y_tm1, partial_y_list in nbest_list:
				if i > 0 and y_tm1 == 1: # y == 1 --> '</s>'
					new_nbest_list.append((score, h_tm1, y_tm1, partial_y_list))
				else:
					update_flag = True
					#print 'score:', score
					#print 'partial_y_list:', partial_y_list
					[log_p_y_t, h_t] = self.search_next_word(h_src, h_tm1, y_tm1)
					#print 'log_p_y_t:', log_p_y_t.shape, 'alignment:', alignment.shape
					if ignore_UNK:
						log_p_y_t[0] = -numpy.inf # 0 --> UNK
					for j in xrange(nbest):
						max_y = numpy.argmax(log_p_y_t)
						#print 'max_y:', max_y, 'max_y_score:', log_p_y_t[max_y]
						new_score = score + log_p_y_t[max_y]
						new_partial_y_list = partial_y_list + [max_y]
						new_nbest_list.append((new_score, h_t, max_y, new_partial_y_list))
						log_p_y_t[max_y] = -numpy.inf
						if i >= 3 and j >= nbest/2: break

			new_nbest_list.sort(key=lambda tup: tup[0], reverse=True)
			nbest_list = new_nbest_list[:nbest]
			if not update_flag: break

		nbest_list.sort(key=lambda tup: tup[0], reverse=True)
		# Percentage
		score_sum = 0
		for score, h_tm1, y_tm1, y_list, alignment_list in nbest_list:
			partial_sum = math.exp(score)
			score_sum += partial_sum
		# nbest
		nbest_y_list = []
		for score, h_tm1, y_tm1, y_list in nbest_list:
			percentage = math.exp(score) / score_sum
			percentage = round(percentage, 2)
			nbest_y_list.append((y_list, percentage))
		# 1-best
		score, h_tm1, y_tm1, y_list = nbest_list[0]
		percentage = math.exp(score) / score_sum
		percentage = round(percentage, 2)
		return (y_list ,percentage), [], nbest_y_list

	def rerank(self, x_sentence, y_sentence_list, weight):
		[h_src, h_tm1] = self.encoding_src_lang(x_sentence)
		#print 'h_src:', h_src.shape, 'h_tm1:', h_tm1.shape
		new_nbest_list = []
		for score, y_sentence, y_word_list in y_sentence_list:
			nll = self.get_nll(x_sentence, h_src, h_tm1, y_sentence)
			new_nbest_list.append((nll + score*weight, y_sentence, y_word_list))

		#new_nbest_list.sort(key=lambda tup: tup[0], reverse=False)
		return new_nbest_list
	
	def train(self, x, y, learning_method, learning_rate):
		words = map(lambda x: numpy.asarray(x).astype('int32'), x)
		labels = y
		# learning_method : sgd, momentum, adadelta
		if learning_method == 'sgd':
			[cost, nll] = self.train_sgd(words, labels, learning_rate)
		elif learning_method == 'momentum':
			[cost, nll] = self.train_momentum(words, labels, learning_rate)
		elif learning_method == 'rmsprop':
			[cost, nll] = self.train_rmsprop(words, labels, learning_rate)
		elif learning_method == 'adadelta':
			[cost, nll] = self.train_adadelta(words, labels, learning_rate)
		return [cost, nll]

	def load_embedding(self, file_name, word2id_dic, ne, de):
		"""
		Loads word vectors from word2vec embedding (key value1 value2 ...)
		"""
		embedding = 0.01 * numpy.random.randn(ne, de).astype(theano.config.floatX)
		if file_name != '':
			print >> sys.stderr, 'load embedding:', file_name, '...',
			count = 0
			if '.txt' in file_name:
				wv = myutil.load_txt_vec(file_name, word2id_dic)
			else:
				wv = myutil.load_bin_vec(file_name, word2id_dic)
			for w in wv:
				idx = word2id_dic[w]
				if idx < ne:
				   	embedding[idx] = wv[w]
					count += 1
				else: print >> sys.stderr, 'Warning(load_embedding):', idx, ne
			print >> sys.stderr, 'done:', count
		else:
			print >> sys.stderr, 'randomly init. embedding.'
		return embedding

	def identity_weight(self, x_dim, y_dim):
		print >> sys.stderr, 'identity weight:', x_dim, y_dim, '...',
		weight = numpy.zeros((x_dim, y_dim)).astype(theano.config.floatX)
		for i in range(x_dim):
			for j in range(y_dim):
				if i==j or i%x_dim==j%x_dim or i%y_dim==j%y_dim:
					weight[i,j] = 1.0
		print >> sys.stderr, 'done.'
		return weight

	# Add Project Name, best - EB
	def save(self, folder, model_name, epoch, prjt_name):
		activation = self.hyper_param['activation']
		nh = self.hyper_param['nhidden']
		de = self.hyper_param['emb_dimension']
		dropout_rate = self.hyper_param['dropout_rate']
		wd = self.hyper_param['weight_decay']
		learning_method = self.hyper_param['learning_method']

		if self.hyper_param['fixed_emb']:
			print 'fixed embeddig.',
			if self.emb not in self.params:
				self.params.append(self.emb)

		numpy_names = []
		numpy_params = []
		for param in self.params:
			numpy_names.append(param.name)
			numpy_params.append(param.get_value(borrow=True))
		if 'reverse_input' in self.hyper_param and self.hyper_param['reverse_input']:
			model_name2 = model_name + '.reverse'
		else: model_name2 = model_name
		model_name3 = prjt_name + '-' + model_name2+'.%s.%s.h%d.e%d.d%g-%g-%g.wd%g.%s'%(activation,learning_method,nh,de,dropout_rate[0],dropout_rate[1],dropout_rate[2],wd,epoch)+'.pkl.gz'
		#f = open(os.path.join(folder, model_name3), 'wb')
		f = gzip.open(os.path.join(folder, model_name3), 'wb', compresslevel=1)
		cPickle.dump([numpy_names, numpy_params], f)
		f.close()

	def load(self, folder, model_name):
		#f = open(os.path.join(folder, model_name), 'rb')
		f = gzip.open(os.path.join(folder, model_name), 'rb')
		[self.numpy_names, self.numpy_params] = cPickle.load(f)
		f.close()


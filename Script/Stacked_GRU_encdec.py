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
import GRU_encdec
import math

import theano
from theano import tensor as T

class Stacked_GRU_encdec(GRU_encdec.GRU_encdec):
	''' Stacked GRU based encoder-decoder model '''
	def build_param(self, hyper_param, word2idx_dic):
		nh = hyper_param['nhidden']
		nh2 = hyper_param['nhidden2']
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
		# Wg : Wz, Wr for second layer
		self.Wg2h_src = theano.shared(name='Wg2h_src', value=0.01 * numpy.random.randn(nh, 2*nh2).astype(theano.config.floatX))
		self.Wg2h = theano.shared(name='Wg2h', value=0.01 * numpy.random.randn(nh, 2*nh2).astype(theano.config.floatX))
		self.Wg2c2 = theano.shared(name='Wg2c2', value=0.01 * numpy.random.randn(nh2, 2*nh2).astype(theano.config.floatX))
		self.Wh2h_src = theano.shared(name='Wh2h_src', value=0.01 * numpy.random.randn(nh, nh2).astype(theano.config.floatX))
		self.Wh2h = theano.shared(name='Wh2h', value=0.01 * numpy.random.randn(nh, nh2).astype(theano.config.floatX))
		#self.Wh2c2 = theano.shared(name='Wh2c', value=0.01 * numpy.random.randn(nh2, nh2).astype(theano.config.floatX))
		self.Wh2c2 = theano.shared(name='Wh2c2', value=0.01 * numpy.random.randn(nh2, nh2).astype(theano.config.floatX))
		#self.Wh20c2 = theano.shared(name='Wh20c', value=0.01 * numpy.random.randn(nh2, nh2).astype(theano.config.floatX))
		self.Wh20c2 = theano.shared(name='Wh20c2', value=0.01 * numpy.random.randn(nh2, nh2).astype(theano.config.floatX))
		# Ug : Uz, Ur
		identity1 = self.identity_weight(nh, nh)
		identity2 = self.identity_weight(nh, 2*nh)
		self.Ugh_src = theano.shared(name='Ugh_src', value=identity2.astype(theano.config.floatX))
		self.Ugh = theano.shared(name='Ugh', value=identity2.astype(theano.config.floatX))
		self.Uhh_src = theano.shared(name='Uhh_src', value=identity1.astype(theano.config.floatX))
		self.Uhh = theano.shared(name='Uhh', value=identity1.astype(theano.config.floatX))
		# Ug : Uz, Ur for second layer
		identity3 = self.identity_weight(nh2, nh2)
		identity4 = self.identity_weight(nh2, 2*nh2)
		self.Ug2h2_src = theano.shared(name='Ug2h2_src', value=identity4.astype(theano.config.floatX))
		self.Ug2h2 = theano.shared(name='Ug2h2', value=identity4.astype(theano.config.floatX))
		self.Uh2h2_src = theano.shared(name='Uh2h2_src', value=identity3.astype(theano.config.floatX))
		self.Uh2h2 = theano.shared(name='Uh2h2', value=identity3.astype(theano.config.floatX))
		# bg : bz, br
		self.bg_src = theano.shared(name='bg_src', value=numpy.zeros(2*nh, dtype=theano.config.floatX))
		self.bg = theano.shared(name='bg', value=numpy.zeros(2*nh, dtype=theano.config.floatX))
		self.bh_src = theano.shared(name='bh_src', value=numpy.zeros(nh, dtype=theano.config.floatX))
		self.bh = theano.shared(name='bh', value=numpy.zeros(nh, dtype=theano.config.floatX))
		self.bh0 = theano.shared(name='bh0', value=numpy.zeros(nh, dtype=theano.config.floatX))
		# bg : bz, br for second layer
		self.bg2_src = theano.shared(name='bg2_src', value=numpy.zeros(2*nh2, dtype=theano.config.floatX))
		self.bg2 = theano.shared(name='bg2', value=numpy.zeros(2*nh2, dtype=theano.config.floatX))
		self.bh2_src = theano.shared(name='bh2_src', value=numpy.zeros(nh2, dtype=theano.config.floatX))
		self.bh2 = theano.shared(name='bh2', value=numpy.zeros(nh2, dtype=theano.config.floatX))
		self.bh20 = theano.shared(name='bh20', value=numpy.zeros(nh2, dtype=theano.config.floatX))
		# others
		self.Wyh2 = theano.shared(name='Wyh2', value=0.01 * numpy.random.randn(nh2, ne).astype(theano.config.floatX))
		self.Wyc2 = theano.shared(name='Wyc2', value=0.01 * numpy.random.randn(nh2, ne).astype(theano.config.floatX))
		self.Wyy = theano.shared(name='Wyy', value=0.01 * numpy.random.randn(de, ne).astype(theano.config.floatX))
		self.by = theano.shared(name='by', value=numpy.zeros(ne, dtype=theano.config.floatX))
		self.h0_src = theano.shared(name='h0_src', value=numpy.zeros(nh, dtype=theano.config.floatX))
		self.h20_src = theano.shared(name='h20_src', value=numpy.zeros(nh2, dtype=theano.config.floatX))
	
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
			# Wg : Wz, Wr
			elif name == 'Wgx_src': self.Wgx_src = theano.shared(name=name, value=numpy_param)
			elif name == 'Wgx': self.Wgx = theano.shared(name=name, value=numpy_param)
			elif name == 'Wgc': self.Wgc = theano.shared(name=name, value=numpy_param)
			elif name == 'Whx_src': self.Whx_src = theano.shared(name=name, value=numpy_param)
			elif name == 'Whx': self.Whx = theano.shared(name=name, value=numpy_param)
			elif name == 'Whc': self.Whc = theano.shared(name=name, value=numpy_param)
			elif name == 'Wh0c': self.Wh0c = theano.shared(name=name, value=numpy_param)
			# Wg : Wz, Wr for second layer
			elif name == 'Wg2h_src': self.Wg2h_src = theano.shared(name=name, value=numpy_param)
			elif name == 'Wg2h': self.Wg2h = theano.shared(name=name, value=numpy_param)
			elif name == 'Wg2c2': self.Wg2c2 = theano.shared(name=name, value=numpy_param)
			elif name == 'Wh2h_src': self.Wh2h_src = theano.shared(name=name, value=numpy_param)
			elif name == 'Wh2h': self.Wh2h = theano.shared(name=name, value=numpy_param)
			# for error correction
			#elif name == 'Wh2c': self.Wh2c2 = theano.shared(name='Wh2c2', value=numpy_param)
			elif name == 'Wh2c2': self.Wh2c2 = theano.shared(name=name, value=numpy_param)
			# for error correction
			#elif name == 'Wh20c': self.Wh20c2 = theano.shared(name='Wh20c2', value=numpy_param)
			elif name == 'Wh20c2': self.Wh20c2 = theano.shared(name=name, value=numpy_param)
			# Ug : Uz, Ur
			elif name == 'Ugh_src': self.Ugh_src = theano.shared(name=name, value=numpy_param)
			elif name == 'Ugh': self.Ugh = theano.shared(name=name, value=numpy_param)
			elif name == 'Uhh_src': self.Uhh_src = theano.shared(name=name, value=numpy_param)
			elif name == 'Uhh': self.Uhh = theano.shared(name=name, value=numpy_param)
			# Ug : Uz, Ur for second layer
			elif name == 'Ug2h2_src': self.Ug2h2_src = theano.shared(name=name, value=numpy_param)
			elif name == 'Ug2h2': self.Ug2h2 = theano.shared(name=name, value=numpy_param)
			elif name == 'Uh2h2_src': self.Uh2h2_src = theano.shared(name=name, value=numpy_param)
			elif name == 'Uh2h2': self.Uh2h2 = theano.shared(name=name, value=numpy_param)
			# bg : bz, br
			# for error correction
			#elif name == 'bg_src':
				#self.bg_src = theano.shared(name=name, value=numpy_param)
				#self.bg2_src = theano.shared(name='bg2_src', value=numpy_param)
			elif name == 'bg_src': self.bg_src = theano.shared(name=name, value=numpy_param)
			elif name == 'bg': self.bg = theano.shared(name=name, value=numpy_param)
			elif name == 'bh_src': self.bh_src = theano.shared(name=name, value=numpy_param)
			elif name == 'bh': self.bh = theano.shared(name=name, value=numpy_param)
			elif name == 'bh0': self.bh0 = theano.shared(name=name, value=numpy_param)
			# bg : bz, br for second layer
			elif name == 'bg2_src': self.bg2_src = theano.shared(name=name, value=numpy_param)
			elif name == 'bg2': self.bg2 = theano.shared(name=name, value=numpy_param)
			elif name == 'bh2_src': self.bh2_src = theano.shared(name=name, value=numpy_param)
			elif name == 'bh2': self.bh2 = theano.shared(name=name, value=numpy_param)
			elif name == 'bh20': self.bh20 = theano.shared(name=name, value=numpy_param)
			# others
			elif name == 'Wyh2': self.Wyh2 = theano.shared(name=name, value=numpy_param)
			elif name == 'Wyc2': self.Wyc2 = theano.shared(name=name, value=numpy_param)
			elif name == 'Wyy': self.Wyy = theano.shared(name=name, value=numpy_param)
			elif name == 'by': self.by = theano.shared(name=name, value=numpy_param)
			elif name == 'h0_src': self.h0_src = theano.shared(name=name, value=numpy_param)
			elif name == 'h20_src': self.h20_src = theano.shared(name=name, value=numpy_param)
			else: print 'skip:', name
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
		nh2 = hyper_param['nhidden2']
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
				self.Whx_src, self.Whx, self.Whc, self.Wh0c, \
				self.Wg2h_src, self.Wg2h, self.Wg2c2, # new \
				self.Wh2h_src, self.Wh2h, self.Wh2c2, self.Wh20c2, # new \
				self.Ugh_src, self.Ugh, self.Uhh_src, self.Uhh, \
				self.Ug2h2_src, self.Ug2h2, self.Uh2h2_src, self.Uh2h2, # new \
				self.bg_src, self.bg, self.bh_src, self.bh, self.bh0, \
				self.bg2_src, self.bg2, self.bh2_src, self.bh2, self.bh20, # new \
				self.Wyh2, self.Wyc2, self.Wyy, \
				self.by, self.h0_src, self.h20_src]

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
		def source_step(x_t, h_tm1, h2_tm1):
			#print 'z_t and r_t are combined!'
			all_t = T.nnet.sigmoid(T.dot(x_t, self.Wgx_src) + T.dot(h_tm1, self.Ugh_src) + self.bg_src)
			z_t = myutil.slice(all_t, 0, nh)
			r_t = myutil.slice(all_t, 1, nh)
			# candidate h_t
			ch_t = myutil.activation(activation, T.dot(x_t, self.Whx_src) + T.dot(r_t * h_tm1, self.Uhh_src) + self.bh_src)
			h_t = (1.0 - z_t) * h_tm1 + z_t * ch_t
			# second layer
			all2_t = T.nnet.sigmoid(T.dot(h_t, self.Wg2h_src) + T.dot(h2_tm1, self.Ug2h2_src) + self.bg2_src)
			z2_t = myutil.slice(all2_t, 0, nh2)
			r2_t = myutil.slice(all2_t, 1, nh2)
			ch2_t = myutil.activation(activation, T.dot(h_t, self.Wh2h_src) + T.dot(r2_t * h2_tm1, self.Uh2h2_src) + self.bh2_src)
			h2_t = (1.0 - z2_t) * h2_tm1 + z2_t * ch2_t
			return [h_t, h2_t]

		def target_step(x_t, h_tm1, h2_tm1, c, c2):
			#print 'z_t and r_t are combined!'
			all_t = T.nnet.sigmoid(T.dot(x_t, self.Wgx) + T.dot(h_tm1, self.Ugh) + T.dot(c, self.Wgc) + self.bg)
			z_t = myutil.slice(all_t, 0, nh)
			r_t = myutil.slice(all_t, 1, nh)
			# candidate h_t
			ch_t = myutil.activation(activation, T.dot(x_t, self.Whx) + T.dot(r_t * h_tm1, self.Uhh) + T.dot(c, self.Whc) + self.bh)
			h_t = (1.0 - z_t) * h_tm1 + z_t * ch_t
			# second layer
			all2_t = T.nnet.sigmoid(T.dot(h_t, self.Wg2h) + T.dot(h2_tm1, self.Ug2h2) + T.dot(c2, self.Wg2c2) + self.bg2)
			z2_t = myutil.slice(all2_t, 0, nh2)
			r2_t = myutil.slice(all2_t, 1, nh2)
			ch2_t = myutil.activation(activation, T.dot(h_t, self.Wh2h) + T.dot(r2_t * h2_tm1, self.Uh2h2) + T.dot(c2, self.Wh2c2) + self.bh2)
			h2_t = (1.0 - z2_t) * h2_tm1 + z2_t * ch2_t
			return [h_t, h2_t]

		# greedy search search
		def greedy_search_step(h_tm1, h2_tm1, y_tm1, h_src, h2_src):
			x_t = self.emb[y_tm1]
			c = h_src[-1]
			c2 = h2_src[-1]
			#print 'z_t and r_t are combined!'
			all_t = T.nnet.sigmoid(T.dot(x_t, self.Wgx) + T.dot(h_tm1, self.Ugh) + T.dot(c, self.Wgc) + self.bg)
			z_t = myutil.slice(all_t, 0, nh)
			r_t = myutil.slice(all_t, 1, nh)
			# candidate h_t
			ch_t = myutil.activation(activation, T.dot(x_t, self.Whx) + T.dot(r_t * h_tm1, self.Uhh) + T.dot(c, self.Whc) + self.bh)
			h_t = (1.0 - z_t) * h_tm1 + z_t * ch_t
			# second layer
			all2_t = T.nnet.sigmoid(T.dot(h_t, self.Wg2h) + T.dot(h2_tm1, self.Ug2h2) + T.dot(c2, self.Wg2c2) + self.bg2)
			z2_t = myutil.slice(all2_t, 0, nh2)
			r2_t = myutil.slice(all2_t, 1, nh2)
			ch2_t = myutil.activation(activation, T.dot(h_t, self.Wh2h) + T.dot(r2_t * h2_tm1, self.Uh2h2) + T.dot(c2, self.Wh2c2) + self.bh2)
			h2_t = (1.0 - z2_t) * h2_tm1 + z2_t * ch2_t
			# score
			s = T.dot(h2_t, self.Wyh2) + T.dot(x_t, self.Wyy) + T.dot(c2, self.Wyc2) + self.by
	   		max_s, y_t = T.max_and_argmax(s)
			exp_s = T.exp(s - max_s)
			p_y = exp_s / exp_s.sum()
			log_p_y = T.log(exp_s / exp_s.sum())
			return [h_t, h2_t, y_t, log_p_y], theano.scan_module.until(T.eq(y_t,1)) # 1 --> '</s>'

		# make score, h_src, h2_src, h0, h20 (for beam search)
		def make_score(x, y, use_noise):
			# input layer dropout: ex. [0.2, 0.2, 0.5]
			if use_noise:
				print "X's projection layer dropout:", dropout_rate[0]
				dropout_x = myutil.dropout_from_layer(x, dropout_rate[0])
			else:
				dropout_x = x * (1.0 - dropout_rate[0])
			# recurrent for source language
			[h_src, h2_src], _ = theano.scan(fn=source_step,
								sequences=dropout_x,
								outputs_info=[self.h0_src, self.h20_src],
								n_steps=dropout_x.shape[0])
			# context
			c = h_src[-1]
			c2 = h2_src[-1]
			h0 = myutil.activation(activation, T.dot(c, self.Wh0c) + self.bh0)
			h20 = myutil.activation(activation, T.dot(c2, self.Wh20c2) + self.bh20)
			# output layer dropout: ex. [0.2, 0.2, 0.5]
			if use_noise:
				print "Y's projection layer dropout:", dropout_rate[1]
				dropout_y = myutil.dropout_from_layer(y, dropout_rate[1])
			else:
				dropout_y = y * (1.0 - dropout_rate[1])
			# forward recurrent for target language
			[h, h2], _ = theano.scan(fn=target_step,
								sequences=dropout_y,
								outputs_info=[h0, h20],
								non_sequences=[c, c2],
								n_steps=dropout_y.shape[0])
			# hidden layer dropout
			if use_noise:
				print "Y's hidden layer dropout:", dropout_rate[2]
				dropout_h2 = myutil.dropout_from_layer(h2, dropout_rate[2])
			else:
				dropout_h2 = h2 * (1.0 - dropout_rate[2])
			# score
			score = T.dot(dropout_h2, self.Wyh2) + T.dot(dropout_y, self.Wyy) + T.dot(c2, self.Wyc2) + self.by
			return score, h_src, h2_src, h0, h20

		# dropout version (for training)
		if 'reverse_input' in hyper_param and hyper_param['reverse_input']:
			print 'reverse input.'
			dropout_score, _, _, _, _ = make_score(x_reverse, y, True)
		else:
			dropout_score, _, _, _, _ = make_score(x, y, True)
	   	dropout_p_y_given_x = myutil.mysoftmax(dropout_score)
		# scaled version (for prediction)
		if 'reverse_input' in hyper_param and hyper_param['reverse_input']:
			print 'reverse input.'
			score, h_src, h2_src, h0, h20 = make_score(x_reverse, y, False)
		else:
			score, h_src, h2_src, h0, h20 = make_score(x, y, False)
	   	p_y_given_x = myutil.mysoftmax(score)

		# prediction
		y_pred = T.argmax(p_y_given_x, axis=1)
		test_nll = -T.mean(T.log(p_y_given_x)[T.arange(y.shape[0]), y_sentence])

		# greedy search decoding
		[_, _, y_pred_greedy, _], _ = theano.scan(fn=greedy_search_step,
							outputs_info=[h0, h20, T.constant(1).astype('int64'), None],
							non_sequences=[h_src, h2_src],
							n_steps=10*x.shape[0])

		# beam search decoding: input=[h_src, h2_src, h_tm1, h_tm2, y_tm1], output=[h_t, h2_t, log_p_y_t]
		input_h_src = T.fmatrix('input_h_src')
		input_h2_src = T.fmatrix('input_h2_src')
		input_h_tm1 = T.fvector('input_h_tm1')
		input_h2_tm1 = T.fvector('input_h2_tm1')
		input_y_tm1 = T.iscalar('input_y_tm1') # input_y_tm1 == x_t
		x_t = self.emb[input_y_tm1]
		c = input_h_src[-1]
		c2 = input_h2_src[-1]
		all_t = T.nnet.sigmoid(T.dot(x_t, self.Wgx) + T.dot(input_h_tm1, self.Ugh) + T.dot(c, self.Wgc) + self.bg)
		z_t = myutil.slice(all_t, 0, nh)
		r_t = myutil.slice(all_t, 1, nh)
		# candidate h_t
		ch_t = myutil.activation(activation, T.dot(x_t, self.Whx) + T.dot(r_t * input_h_tm1, self.Uhh) + T.dot(c, self.Whc) + self.bh)
		h_t = (1.0 - z_t) * input_h_tm1 + z_t * ch_t
		# second layer
		all2_t = T.nnet.sigmoid(T.dot(h_t, self.Wg2h) + T.dot(input_h2_tm1, self.Ug2h2) + T.dot(c2, self.Wg2c2) + self.bg2)
		z2_t = myutil.slice(all2_t, 0, nh2)
		r2_t = myutil.slice(all2_t, 1, nh2)
		ch2_t = myutil.activation(activation, T.dot(h_t, self.Wh2h) + T.dot(r2_t * input_h2_tm1, self.Uh2h2) + T.dot(c2, self.Wh2c2) + self.bh2)
		h2_t = (1.0 - z2_t) * input_h2_tm1 + z2_t * ch2_t
		# prediction
		score_y_t = T.dot(h2_t, self.Wyh2) + T.dot(x_t, self.Wyy) + T.dot(c2, self.Wyc2) + self.by
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
		rmsprop_updates = myutil.sgd_updates_rmsprop(self.params, cost, lr, 0.9, 0.1)
		# AdaDelta (lr --> rho = 0.95)
		adadelta_updates = myutil.sgd_updates_adadelta(self.params, cost, lr, 1e-6, 9)

		# theano functions to compile
		self.classify = theano.function(inputs=[x_sentence, y_sentence], outputs=[y_pred, test_nll])
		self.greedy_search = theano.function(inputs=[x_sentence], outputs=[y_pred_greedy])
		# for beam search
		self.encoding_src_lang = theano.function(inputs=[x_sentence], outputs=[h_src, h2_src, h0, h20])
		self.search_next_word = theano.function(inputs=[input_h_src, input_h2_src, input_h_tm1, input_h2_tm1, input_y_tm1], outputs=[log_p_y_t, h_t, h2_t])
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
		[h_src, h2_src, h_tm1, h2_tm1] = self.encoding_src_lang(x_sentence)
		#print 'h_src:', h_src.shape, 'h_tm1:', h_tm1.shape
		nbest_list = [(0, h_tm1, h2_tm1, 1, [])]
		# loop
		for i in xrange(10*len(x_sentence)):
			#print '###', i
			update_flag = False
			new_nbest_list = []
			for score, h_tm1, h2_tm1, y_tm1, partial_y_list in nbest_list:
				if i > 0 and y_tm1 == 1: # y == 1 --> '</s>'
					new_nbest_list.append((score, h_tm1, h2_tm1, y_tm1, partial_y_list))
				else:
					update_flag = True
					#print 'score:', score
					#print 'partial_y_list:', partial_y_list
					[log_p_y_t, h_t, h2_t] = self.search_next_word(h_src, h2_src, h_tm1, h2_tm1, y_tm1)
					#print 'log_p_y_t:', log_p_y_t.shape, 'alignment:', alignment.shape
					if ignore_UNK:
						log_p_y_t[0] = -numpy.inf # 0 --> UNK
					for j in xrange(nbest):
						max_y = numpy.argmax(log_p_y_t)
						#print 'max_y:', max_y, 'max_y_score:', log_p_y_t[max_y]
						new_score = score + log_p_y_t[max_y]
						new_partial_y_list = partial_y_list + [max_y]
						new_nbest_list.append((new_score, h_t, h2_t, max_y, new_partial_y_list))
						log_p_y_t[max_y] = -numpy.inf
						if i >= 3 and j >= nbest/2: break

			new_nbest_list.sort(key=lambda tup: tup[0], reverse=True)
			nbest_list = new_nbest_list[:nbest]
			if not update_flag: break

		nbest_list.sort(key=lambda tup: tup[0], reverse=True)
		# nbest
		nbest_y_list = []
		for score, h_tm1, h2_t, y_tm1, y_list in nbest_list:
			nbest_y_list.append(y_list)
		# 1-best
		score, h_tm1, h2_t, y_tm1, y_list = nbest_list[0]
		return y_list, [], nbest_y_list

# ADDED BY EB
	def beam_search_percentage(self, x_sentence, nbest=20, ignore_UNK=False):
		[h_src, h2_src, h_tm1, h2_tm1] = self.encoding_src_lang(x_sentence)
		#print 'h_src:', h_src.shape, 'h_tm1:', h_tm1.shape
		nbest_list = [(0, h_tm1, h2_tm1, 1, [])]
		# loop
		for i in xrange(10*len(x_sentence)):
			#print '###', i
			update_flag = False
			new_nbest_list = []
			for score, h_tm1, h2_tm1, y_tm1, partial_y_list in nbest_list:
				if i > 0 and y_tm1 == 1: # y == 1 --> '</s>'
					new_nbest_list.append((score, h_tm1, h2_tm1, y_tm1, partial_y_list))
				else:
					update_flag = True
					#print 'score:', score
					#print 'partial_y_list:', partial_y_list
					[log_p_y_t, h_t, h2_t] = self.search_next_word(h_src, h2_src, h_tm1, h2_tm1, y_tm1)
					#print 'log_p_y_t:', log_p_y_t.shape, 'alignment:', alignment.shape
					if ignore_UNK:
						log_p_y_t[0] = -numpy.inf # 0 --> UNK
					for j in xrange(nbest):
						max_y = numpy.argmax(log_p_y_t)
						#print 'max_y:', max_y, 'max_y_score:', log_p_y_t[max_y]
						new_score = score + log_p_y_t[max_y]
						new_partial_y_list = partial_y_list + [max_y]
						new_nbest_list.append((new_score, h_t, h2_t, max_y, new_partial_y_list))
						log_p_y_t[max_y] = -numpy.inf
						if i >= 3 and j >= nbest/2: break

			new_nbest_list.sort(key=lambda tup: tup[0], reverse=True)
			nbest_list = new_nbest_list[:nbest]
			if not update_flag: break

		nbest_list.sort(key=lambda tup: tup[0], reverse=True)
		#Percentage
		score_sum = 0
		for score, h_tm1, y_tm1, y_list, alignment_list in nbest_list:
			partial_sum = math.exp(score)
			score_sum += partial_sum

		# nbest
		nbest_y_list = []
		for score, h_tm1, h2_t, y_tm1, y_list in nbest_list:
			percentage = math.exp(score) / score_sum
			percentage = round(percentage, 2)
			nbest_y_list.append((y_list, percentage))
		# 1-best
		score, h_tm1, h2_t, y_tm1, y_list = nbest_list[0]
		percentage = math.exp(score) / score_sum
		percentage = round(percentage, 2)
		return (y_list, percentage), [], nbest_y_list

	# Add Project Name
	def save(self, folder, model_name, epoch, prjt_name):
		activation = self.hyper_param['activation']
		nh = self.hyper_param['nhidden']
		nh2 = self.hyper_param['nhidden']
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
		model_name3 = prjt_name + model_name2+'.%s.%s.h%d-%d.e%d.d%g-%g-%g.wd%g.%s'%(activation,learning_method,nh,nh2,de,dropout_rate[0],dropout_rate[1],dropout_rate[2],wd,epoch)+'.pkl.gz'
		#f = open(os.path.join(folder, model_name3), 'wb')
		f = gzip.open(os.path.join(folder, model_name3), 'wb', compresslevel=1)
		cPickle.dump([numpy_names, numpy_params], f)
		f.close()
	


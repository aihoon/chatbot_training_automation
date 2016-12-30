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

class GRU_DO_search(GRU_encdec.GRU_encdec):
	''' GRU based search model + Deep Output'''
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
		self.Wgxb_src = theano.shared(name='Wgxb_src', value=0.01 * numpy.random.randn(de, 2*nh).astype(theano.config.floatX))
		self.Wgx = theano.shared(name='Wgx', value=0.01 * numpy.random.randn(de, 2*nh).astype(theano.config.floatX))
		self.Wgc = theano.shared(name='Wgc', value=0.01 * numpy.random.randn(2*nh, 2*nh).astype(theano.config.floatX))
		self.Whx_src = theano.shared(name='Whx_src', value=0.01 * numpy.random.randn(de, nh).astype(theano.config.floatX))
		self.Whxb_src = theano.shared(name='Whxb_src', value=0.01 * numpy.random.randn(de, nh).astype(theano.config.floatX))
		self.Whx = theano.shared(name='Whx', value=0.01 * numpy.random.randn(de, nh).astype(theano.config.floatX))
		self.Whc = theano.shared(name='Whc', value=0.01 * numpy.random.randn(2*nh, nh).astype(theano.config.floatX))
		self.Wh0c = theano.shared(name='Wh0c', value=0.01 * numpy.random.randn(2*nh, nh).astype(theano.config.floatX))
		# Ug : Uz, Ur
		identity1 = self.identity_weight(nh, nh)
		identity2 = self.identity_weight(nh, 2*nh)
		self.Ugh_src = theano.shared(name='Ugh_src', value=identity2.astype(theano.config.floatX))
		self.Ughb_src = theano.shared(name='Ughb_src', value=identity2.astype(theano.config.floatX))
		self.Ugh = theano.shared(name='Ugh', value=identity2.astype(theano.config.floatX))
		self.Uhh_src = theano.shared(name='Uhh_src', value=identity1.astype(theano.config.floatX))
		self.Uhhb_src = theano.shared(name='Uhhb_src', value=identity1.astype(theano.config.floatX))
		self.Uhh = theano.shared(name='Uhh', value=identity1.astype(theano.config.floatX))
		# bg : bz, br
		self.bg_src = theano.shared(name='bg_src', value=numpy.zeros(2*nh, dtype=theano.config.floatX))
		self.bgb_src = theano.shared(name='bgb_src', value=numpy.zeros(2*nh, dtype=theano.config.floatX))
		self.bg = theano.shared(name='bg', value=numpy.zeros(2*nh, dtype=theano.config.floatX))
		self.bh_src = theano.shared(name='bh_src', value=numpy.zeros(nh, dtype=theano.config.floatX))
		self.bhb_src = theano.shared(name='bhb_src', value=numpy.zeros(nh, dtype=theano.config.floatX))
		self.bh = theano.shared(name='bh', value=numpy.zeros(nh, dtype=theano.config.floatX))
		self.bh2 = theano.shared(name='bh2', value=numpy.zeros(nh2, dtype=theano.config.floatX))
		self.bh0 = theano.shared(name='bh0', value=numpy.zeros(nh, dtype=theano.config.floatX))
		# aligment
		#self.Wa = theano.shared(name='Wa', value=0.01 * numpy.random.randn(nh, nh).astype(theano.config.floatX))
		self.Wah = theano.shared(name='Wah', value=0.01 * numpy.random.randn(nh, nh).astype(theano.config.floatX))
		self.Wax = theano.shared(name='Wax', value=0.01 * numpy.random.randn(de, nh).astype(theano.config.floatX))
		self.Ua = theano.shared(name='Ua', value=0.01 * numpy.random.randn(2*nh, nh).astype(theano.config.floatX))
		self.ba = theano.shared(name='ba', value=numpy.zeros(nh, dtype=theano.config.floatX))
		self.va = theano.shared(name='va', value=numpy.zeros(nh, dtype=theano.config.floatX))
		# others
		self.Wh2h = theano.shared(name='Wh2h', value=0.01 * numpy.random.randn(nh, nh2).astype(theano.config.floatX))
		self.Wyh2 = theano.shared(name='Wyh2', value=0.01 * numpy.random.randn(nh2, ne).astype(theano.config.floatX))
		self.Wyh = theano.shared(name='Wyh', value=0.01 * numpy.random.randn(nh, ne).astype(theano.config.floatX))
		self.Wyc = theano.shared(name='Wyc', value=0.01 * numpy.random.randn(2*nh, ne).astype(theano.config.floatX))
		self.Wyy = theano.shared(name='Wyy', value=0.01 * numpy.random.randn(de, ne).astype(theano.config.floatX))
		self.by = theano.shared(name='by', value=numpy.zeros(ne, dtype=theano.config.floatX))
		self.h0_src = theano.shared(name='h0_src', value=numpy.zeros(nh, dtype=theano.config.floatX))
		self.h0b_src = theano.shared(name='h0b_src', value=numpy.zeros(nh, dtype=theano.config.floatX))
	
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
			elif name == 'Wgxb_src': self.Wgxb_src = theano.shared(name=name, value=numpy_param)
			elif name == 'Wgx': self.Wgx = theano.shared(name=name, value=numpy_param)
			elif name == 'Wgc': self.Wgc = theano.shared(name=name, value=numpy_param)
			elif name == 'Whx_src': self.Whx_src = theano.shared(name=name, value=numpy_param)
			elif name == 'Whxb_src': self.Whxb_src = theano.shared(name=name, value=numpy_param)
			elif name == 'Whx': self.Whx = theano.shared(name=name, value=numpy_param)
			elif name == 'Whc': self.Whc = theano.shared(name=name, value=numpy_param)
			elif name == 'Wh0c': self.Wh0c = theano.shared(name=name, value=numpy_param)
			# Ug : Uz, Ur
			elif name == 'Ugh_src': self.Ugh_src = theano.shared(name=name, value=numpy_param)
			elif name == 'Ughb_src': self.Ughb_src = theano.shared(name=name, value=numpy_param)
			elif name == 'Ugh': self.Ugh = theano.shared(name=name, value=numpy_param)
			elif name == 'Uhh_src': self.Uhh_src = theano.shared(name=name, value=numpy_param)
			elif name == 'Uhhb_src': self.Uhhb_src = theano.shared(name=name, value=numpy_param)
			elif name == 'Uhh': self.Uhh = theano.shared(name=name, value=numpy_param)
			# bg : bz, br
			elif name == 'bg_src': self.bg_src = theano.shared(name=name, value=numpy_param)
			elif name == 'bgb_src': self.bgb_src = theano.shared(name=name, value=numpy_param)
			elif name == 'bg': self.bg = theano.shared(name=name, value=numpy_param)
			elif name == 'bh_src': self.bh_src = theano.shared(name=name, value=numpy_param)
			elif name == 'bhb_src': self.bhb_src = theano.shared(name=name, value=numpy_param)
			elif name == 'bh': self.bh = theano.shared(name=name, value=numpy_param)
			elif name == 'bh2': self.bh2 = theano.shared(name=name, value=numpy_param)
			elif name == 'bh0': self.bh0 = theano.shared(name=name, value=numpy_param)
			# aligment
			elif name == 'Wah': self.Wah = theano.shared(name=name, value=numpy_param)
			elif name == 'Wax': self.Wax = theano.shared(name=name, value=numpy_param)
			elif name == 'Ua': self.Ua = theano.shared(name=name, value=numpy_param)
			elif name == 'ba': self.ba = theano.shared(name=name, value=numpy_param)
			elif name == 'va': self.va = theano.shared(name=name, value=numpy_param)
			# others
			elif name == 'Wh2h': self.Wh2h = theano.shared(name=name, value=numpy_param)
			elif name == 'Wyh2': self.Wyh2 = theano.shared(name=name, value=numpy_param)
			elif name == 'Wyh': self.Wyh = theano.shared(name=name, value=numpy_param)
			elif name == 'Wyc': self.Wyc = theano.shared(name=name, value=numpy_param)
			elif name == 'Wyy': self.Wyy = theano.shared(name=name, value=numpy_param)
			elif name == 'by': self.by = theano.shared(name=name, value=numpy_param)
			elif name == 'h0_src': self.h0_src = theano.shared(name=name, value=numpy_param)
			elif name == 'h0b_src': self.h0b_src = theano.shared(name=name, value=numpy_param)
			else: print '\nskip:', name
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
		# parameters of the model
		if hyper_param['load_model'] != '':
			self.load_param(hyper_param, hyper_param['load_model'])
		else:
			self.build_param(hyper_param, word2idx_dic)

		# parameters
		self.params = [self.emb, self.Wgx_src, self.Wgxb_src, self.Wgx, self.Wgc, \
				self.Whx_src, self.Whxb_src, self.Whx, self.Whc, self.Wh0c, \
				self.Ugh_src, self.Ughb_src, self.Ugh, self.Uhh_src, self.Uhhb_src, self.Uhh, \
				self.bg_src, self.bgb_src, self.bg, self.bh_src, self.bhb_src, self.bh, self.bh2, self.bh0, \
				self.Wah, self.Wax, self.Ua, self.ba, self.va, \
				self.Wh2h, self.Wyh2, self.Wyh, self.Wyc, self.Wyy, self.by, \
				self.h0_src, self.h0b_src]

		if hyper_param['fixed_emb']:
			print 'fixed embeddig.'
			self.params.remove(self.emb)

		# as many lines as words in the sentence
		x_sentence = T.ivector('x_sentence') # x_sentence : n_steps
		x = self.emb[x_sentence].reshape((x_sentence.shape[0], de)) # don't remove '</s>'
		x_reverse = x[::-1] # reverse for backward

		y_sentence = T.ivector('y_sentence') # labels
		y_input_sentence = T.concatenate([y_sentence[-1:], y_sentence[:-1]], axis=0) # move '</s>' to first position
		y = self.emb[y_input_sentence].reshape((y_input_sentence.shape[0], de))

		# for scan
		#def source_step(x_t, h_tm1):
		def source_step(dot_x_t_Wgx_src, dot_x_t_Whx_src, h_tm1):
			#print 'z_t, r_t are combined!'
			#all_t = T.nnet.sigmoid(T.dot(x_t, self.Wgx_src) + T.dot(h_tm1, self.Ugh_src) + self.bg_src)
			all_t = T.nnet.sigmoid(dot_x_t_Wgx_src + T.dot(h_tm1, self.Ugh_src))
			z_t = myutil.slice(all_t, 0, nh)
			r_t = myutil.slice(all_t, 1, nh)
			# candidate h_t
			#ch_t = myutil.activation(activation, T.dot(x_t, self.Whx_src) + T.dot(r_t * h_tm1, self.Uhh_src) + self.bh_src)
			ch_t = myutil.activation(activation, dot_x_t_Whx_src + T.dot(r_t * h_tm1, self.Uhh_src))
			h_t = (1.0 - z_t) * h_tm1 + z_t * ch_t
			return h_t

		#def source_backward_step(x_t, h_tm1):
		def source_backward_step(dot_x_t_Wgxb_src, dot_x_t_Whxb_src, h_tm1):
			#print 'z_t and r_t are combined!'
			#all_t = T.nnet.sigmoid(T.dot(x_t, self.Wgxb_src) + T.dot(h_tm1, self.Ughb_src) + self.bgb_src)
			all_t = T.nnet.sigmoid(dot_x_t_Wgxb_src + T.dot(h_tm1, self.Ughb_src))
			z_t = myutil.slice(all_t, 0, nh)
			r_t = myutil.slice(all_t, 1, nh)
			# candidate h_t
			#ch_t = myutil.activation(activation, T.dot(x_t, self.Whxb_src) + T.dot(r_t * h_tm1, self.Uhhb_src) + self.bhb_src)
			ch_t = myutil.activation(activation, dot_x_t_Whxb_src + T.dot(r_t * h_tm1, self.Uhhb_src))
			h_t = (1.0 - z_t) * h_tm1 + z_t * ch_t
			return h_t

		#def target_step(x_t, h_tm1, h_src, dot_h_src_Ua):
		def target_step(dot_x_t_Wgx, dot_x_t_Whx, dot_x_t_Wax, h_tm1, h_src, dot_h_src_Ua):
			# search c_t
			#z = T.tanh(T.dot(h_tm1, self.Wa) + T.dot(h_src, self.Ua) + self.ba)
			#z = T.tanh(T.dot(h_tm1, self.Wa) + dot_h_src_Ua)
			z = T.tanh(T.dot(h_tm1, self.Wah) + dot_x_t_Wax + dot_h_src_Ua)
			#print 'z:', z.ndim
			e = T.dot(self.va, z.T)
			#print 'e:', e.ndim
			max_e = T.max(e)
			exp_e = T.exp(e - max_e)
			a = exp_e / exp_e.sum()
			#print 'a:', a.ndim
			c_t = T.dot(a, h_src)
			#print 'c_t:', c_t.ndim
			#print 'z_t and r_t are combined!'
			all_t = T.nnet.sigmoid(dot_x_t_Wgx + T.dot(h_tm1, self.Ugh) + T.dot(c_t, self.Wgc))
			z_t = myutil.slice(all_t, 0, nh)
			r_t = myutil.slice(all_t, 1, nh)
			# candidate h_t
			ch_t = myutil.activation(activation, dot_x_t_Whx + T.dot(r_t * h_tm1, self.Uhh) +
					T.dot(c_t, self.Whc))
			h_t = (1.0 - z_t) * h_tm1 + z_t * ch_t
			return [h_t, c_t]


		# make score, h_src, h0 (for beam search)
		def make_score(x, y, use_noise):
			# input layer dropout: ex. [0.2, 0.2, 0.5]
			if use_noise:
				print "X's projection layer dropout:", dropout_rate[0]
				dropout_x = myutil.dropout_from_layer(x, dropout_rate[0])
			else:
				dropout_x = x * (1.0 - dropout_rate[0])
			dropout_x_reverse = dropout_x[::-1] # reverse for backward
			# RNN encoder
			dot_x_Wgx_src = T.dot(dropout_x, self.Wgx_src) + self.bg_src
			dot_x_Whx_src = T.dot(dropout_x, self.Whx_src) + self.bh_src
			dot_x_rev_Wgx_src = T.dot(dropout_x_reverse, self.Wgxb_src) + self.bgb_src
			dot_x_rev_Whx_src = T.dot(dropout_x_reverse, self.Whxb_src) + self.bhb_src
			# forward recurrent for source language
			hf_src, _ = theano.scan(fn=source_step,
								sequences=[dot_x_Wgx_src, dot_x_Whx_src],
								outputs_info=self.h0_src,
								n_steps=dropout_x.shape[0])
			# backward recurrent for source language
			hb_src_reverse, _ = theano.scan(fn=source_backward_step,
								sequences=[dot_x_rev_Wgx_src, dot_x_rev_Whx_src],
								outputs_info=self.h0b_src,
								n_steps=dropout_x_reverse.shape[0])
			hb_src = hb_src_reverse[::-1]
			h_src = T.concatenate([hf_src, hb_src], axis=1)
			# global context
			#c_global = h_src[0]
			c_global = T.concatenate([hf_src[-1], hb_src[0]], axis=0)
			# output layer dropout: ex. [0.2, 0.2, 0.5]
			# output layer (target language input layer) dropout: ex. [0.2, 0.2, 0.5]
			if use_noise:
				print "Y's projection layer dropout:", dropout_rate[1]
				dropout_y = myutil.dropout_from_layer(y, dropout_rate[1])
			else:
				dropout_y = y * (1.0 - dropout_rate[1])
			# RNN decoder
			dot_y_Wgx = T.dot(dropout_y, self.Wgx) + self.bg
			dot_y_Whx = T.dot(dropout_y, self.Whx) + self.bh
			dot_y_Wax = T.dot(dropout_y, self.Wax)
			dot_h_src_Ua = T.dot(h_src, self.Ua) + self.ba
			h0 = myutil.activation(activation, T.dot(c_global, self.Wh0c) + self.bh0)
			# forward recurrent for target language
			[h, c], _ = theano.scan(fn=target_step,
							sequences=[dot_y_Wgx, dot_y_Whx, dot_y_Wax],
							outputs_info=[h0, None],
							non_sequences=[h_src, dot_h_src_Ua],
							n_steps=dropout_y.shape[0])
			# h2 - Deep Output RNN
			print 'Deep Output RNN: ReLU'
			# hidden layer dropout
			if use_noise:
				print "Y's hidden layer dropout:", dropout_rate[2]
				dropout_h = myutil.dropout_from_layer(h, dropout_rate[2])
			else:
				dropout_h = h * (1.0 - dropout_rate[2])
			# h2 - Deep Output RNN
			print 'Deep Output RNN: ReLU'
			h2 = myutil.activation('relu', T.dot(dropout_h, self.Wh2h) + self.bh2)
			# score
			score = T.dot(h2, self.Wyh2) + T.dot(dropout_h, self.Wyh) + T.dot(dropout_y, self.Wyy) + \
					T.dot(c, self.Wyc) + self.by
			return score, h_src, h0

		# dropout version (for training)
		dropout_score, _, _ = make_score(x, y, True)
		dropout_p_y_given_x = myutil.mysoftmax(dropout_score)

		# scaled version (for prediction)
		score, h_src, h0 = make_score(x, y, False)
	   	p_y_given_x = myutil.mysoftmax(score)

		# prediction
		y_pred = T.argmax(p_y_given_x, axis=1)
		test_nll = -T.mean(T.log(p_y_given_x)[T.arange(y.shape[0]), y_sentence])

		# beam search decoding: input=[h_src, h_tm1, y_tm1], output=[h_t, log_p_y_t, alignment]
		input_h_src = T.fmatrix('input_h_src')
		input_h_tm1 = T.fvector('input_h_tm1')
		input_y_tm1 = T.iscalar('input_y_tm1') # input_y_tm1 == x_t
		x_t = self.emb[input_y_tm1]
		# search c_t
		#z = T.tanh(T.dot(input_h_tm1, self.Wa) + T.dot(input_h_src, self.Ua) + self.ba)
		z = T.tanh(T.dot(input_h_tm1, self.Wah) + T.dot(x_t, self.Wax) +
				T.dot(input_h_src, self.Ua) + self.ba)
		e = T.dot(self.va, z.T)
		max_e = T.max(e)
		exp_e = T.exp(e - max_e)
		alignment = exp_e / exp_e.sum()
		c_t = T.dot(alignment, input_h_src)
		all_t = T.nnet.sigmoid(T.dot(x_t, self.Wgx) + T.dot(input_h_tm1, self.Ugh) + 
				T.dot(c_t, self.Wgc) + self.bg)
		z_t = myutil.slice(all_t, 0, nh)
		r_t = myutil.slice(all_t, 1, nh)
		# candidate h_t
		ch_t = myutil.activation(activation, T.dot(x_t, self.Whx) + 
				T.dot(r_t * input_h_tm1, self.Uhh) + T.dot(c_t, self.Whc) + 
				self.bh)
		h_t = (1.0 - z_t) * input_h_tm1 + z_t * ch_t
		# h2 - Deep Output RNN
		h2_t = myutil.activation('relu', T.dot(h_t, self.Wh2h) + self.bh2)
		# prediction
		score_y_t = T.dot(h2_t, self.Wyh2) + T.dot(h_t, self.Wyh) + T.dot(x_t, self.Wyy) + \
				T.dot(c_t, self.Wyc) + self.by
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
		sgd_updates = myutil.sgd_updates(self.params, cost, lr)
		# SGD + momentum
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
				outputs=[log_p_y_t, h_t, alignment])
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
		nbest_list = [(0, h_tm1, 1, [], [])]
		# loop
		for i in xrange(10*len(x_sentence)):
			#print '###', i
			update_flag = False
			new_nbest_list = []
			for score, h_tm1, y_tm1, partial_y_list, alignment_list in nbest_list:
				if i > 0 and y_tm1 == 1: # y == 1 --> '</s>'
					new_nbest_list.append((score, h_tm1, y_tm1, partial_y_list, alignment_list))
				else:
					update_flag = True
					#print 'score:', score
					#print 'partial_y_list:', partial_y_list
					[log_p_y_t, h_t, alignment] = self.search_next_word(h_src, h_tm1, y_tm1)
					#print 'log_p_y_t:', log_p_y_t.shape, 'alignment:', alignment.shape
					if ignore_UNK:
						log_p_y_t[0] = -numpy.inf # 0 --> UNK
					for j in xrange(nbest):
						max_y = numpy.argmax(log_p_y_t)
						#print 'max_y:', max_y, 'max_y_score:', log_p_y_t[max_y]
						new_score = score + log_p_y_t[max_y]
						new_partial_y_list = partial_y_list + [max_y]
						new_alignment_list = alignment_list + [alignment]
						new_nbest_list.append((new_score, h_t, max_y, new_partial_y_list, new_alignment_list))
						log_p_y_t[max_y] = -numpy.inf
						if i >= 3 and j >= nbest/2: break

			new_nbest_list.sort(key=lambda tup: tup[0], reverse=True)
			nbest_list = new_nbest_list[:nbest]

			if not update_flag: break

		nbest_list.sort(key=lambda tup: tup[0], reverse=True)
		# nbest
		nbest_y_list = []
		for score, h_tm1, y_tm1, y_list, alignment_list in nbest_list:
			nbest_y_list.append(y_list)
			print '#', score, y_list
		# 1-best
		score, h_tm1, y_tm1, y_list, alignment_list = nbest_list[0]
		return y_list, alignment_list, nbest_y_list

# ADDED BY EB
	def beam_search_percentage(self, x_sentence, nbest=20, ignore_UNK=False):
		[h_src, h_tm1] = self.encoding_src_lang(x_sentence)
		#print 'h_src:', h_src.shape, 'h_tm1:', h_tm1.shape
		nbest_list = [(0, h_tm1, 1, [], [])]
		# loop
		for i in xrange(10*len(x_sentence)):
			#print '###', i
			update_flag = False
			new_nbest_list = []
			for score, h_tm1, y_tm1, partial_y_list, alignment_list in nbest_list:
				if i > 0 and y_tm1 == 1: # y == 1 --> '</s>'
					new_nbest_list.append((score, h_tm1, y_tm1, partial_y_list, alignment_list))
				else:
					update_flag = True
					#print 'score:', score
					#print 'partial_y_list:', partial_y_list
					[log_p_y_t, h_t, alignment] = self.search_next_word(h_src, h_tm1, y_tm1)
					#print 'log_p_y_t:', log_p_y_t.shape, 'alignment:', alignment.shape
					if ignore_UNK:
						log_p_y_t[0] = -numpy.inf # 0 --> UNK
					for j in xrange(nbest):
						max_y = numpy.argmax(log_p_y_t)
						#print 'max_y:', max_y, 'max_y_score:', log_p_y_t[max_y]
						new_score = score + log_p_y_t[max_y]
						new_partial_y_list = partial_y_list + [max_y]
						new_alignment_list = alignment_list + [alignment]
						new_nbest_list.append((new_score, h_t, max_y, new_partial_y_list, new_alignment_list))
						log_p_y_t[max_y] = -numpy.inf
						if i >= 3 and j >= nbest/2: break

			new_nbest_list.sort(key=lambda tup: tup[0], reverse=True)
			nbest_list = new_nbest_list[:nbest]
			if not update_flag: break

		nbest_list.sort(key=lambda tup: tup[0], reverse=True)

		#CALCULATE THE PERCENTAGE
		score_sum = 0.0
		for score, h_tm1, y_tm1, y_list, alignment_list in nbest_list:
			partial_sum = math.exp(score)
			score_sum += partial_sum

		# nbest
		nbest_y_list = []
		for score, h_tm1, y_tm1, y_list, alignment_list in nbest_list:
			percentage = math.exp(score) / score_sum
			percentage = round(percentage, 2)
			nbest_y_list.append((y_list, percentage))

		# 1-best
		score, h_tm1, y_tm1, y_list, alignment_list = nbest_list[0]
		percentage = math.exp(score) / score_sum
		percentage = round(percentage, 2)
		return (y_list, percentage), alignment_list, nbest_y_list
	

	def rerank(self, x_sentence, y_sentence_list, weight):
		[h_src, h_tm1] = self.encoding_src_lang(x_sentence)
		#print 'h_src:', h_src.shape, 'h_tm1:', h_tm1.shape
		new_nbest_list = []
		for score, y_sentence, y_word_list in y_sentence_list:
			nll = self.get_nll(x_sentence, h_src, h_tm1, y_sentence)
			new_nbest_list.append((nll + score*weight, y_sentence, y_word_list))

		#new_nbest_list.sort(key=lambda tup: tup[0], reverse=False)
		return new_nbest_list


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

import theano
from theano import tensor as T
from compiler.ast import flatten

# by leeck
import myutil
import GRU_encdec
import Stacked_GRU_encdec
import GRU_DO_search

theano.config.allow_gc=True
theano.config.floatX='float32'
theano.config.warn_float64='warn'
theano.config.dnn.enable=False
theano.config.cnmem=1
if 'gpu' in theano.config.device: theano.config.nvcc.fastmath=True
theano.config.mode='FAST_RUN'

def main(prjt_name, language, param=None):
    proj_kor_dir = os.path.join(os.environ['HOME'], "Mindsbot", "Train.kor", prjt_name)
    proj_eng_dir = os.path.join(os.environ['HOME'], "Mindsbot", "Train.eng", prjt_name)
    data_dir = os.path.join(os.environ['HOME'], "Mindsbot", "Data")

	if not param:
		param = {
		 #'nn_type': 'GRU_encdec', # GRU based encoder-decoder model
		 #'nn_type': 'Stacked_GRU_encdec', # Stacked GRU based encoder-decoder model
		 'nn_type': 'GRU_DO_search', # GRU search + Deep Output
		 #'activation': 'sigm', # sigmoid
		 'activation': 'tanh', # tanh
		 #'activation': 'relu', # ReLU
		 'reverse_input': False, # reverse input
		 #'learning_method': 'sgd', # SGD
		 #'learning_method': 'momentum', # SGD with momentum
		 'learning_method': 'rmsprop', # RMSProp
		 #'learning_method': 'adadelta', # adadelta
		 #'lr': 0.01, # SGD, mommentum
		 'lr': 0.1, # SGD, mommentum, RMSProp
		 #'lr': 0.98, # AdaDelta - rho
		 #'lr': 1.0, # RMSProp
		 'weight_decay': 1e-6, # weight decay - by leeck
		 'dropout_rate': [0, 0, 0], # dropout rate - by leeck
		 'nhidden': 500, # number of hidden units
		 'nhidden2': 500, # number of hidden units (Deep Output)
		 #'emb_dimension': 50, # dimension of word embedding
		 #'emb_dimension': 100, # dimension of word embedding
		 'emb_dimension': 200, # dimension of word embedding
		 'fixed_emb': False, # fixed embedding
		 'gradient_clip': False, # gradient clipping
		 #'emb_file': '', # random init.
		 ####'emb_file': '../data/Korean_word_char.nnlm.h200.bin', # Korean char embedding file
		 ####'eng_emb_file': '../eng_data/eng_train_20.vector.nnlm.h200.txt', # English emb
		 'emb_file':     data_dir + '/Korean_word_char.nnlm.h200.bin',      # Korean char embedding file
		 'eng_emb_file': data_dir + '/eng_train_20.vector.nnlm.h200.txt',   # English emb
		 'load_model': '', # '' --> build random parameters
		 ####'train_data': '../data/Project/' + prjt_name + '/' + prjt_name + '.train.pkl.gz', # training data
		 ####'eng_train_data': '../eng_data/Project/' + prjt_name + '/' + prjt_name + '.train.pkl.gz', # eng training data
		 ####'test_data': '../data/Project/' + prjt_name + '/' + prjt_name + '.test.pkl.gz', # test data
		 ####'eng_test_data': '../eng_data/Project/' + prjt_name + '/' + prjt_name + '.test.pkl.gz', # eng test data
		 ####'folder': '../data/Project/' + prjt_name, # folder
		 ####'eng_folder': '../eng_data/Project/' + prjt_name, # folder
		 'train_data':     proj_kor_dir + '/' + prjt_name + '.train.pkl.gz',    # training data
		 'eng_train_data': proj_eng_dir + '/' + prjt_name + '.train.pkl.gz',    # eng training data
		 'test_data':      proj_kor_dir + '/' + prjt_name + '.test.pkl.gz',     # test data
		 'eng_test_data':  proj_eng_dir + '/' + prjt_name + '.test.pkl.gz',     # eng test data
		 'folder':         proj_kor_dir,                                        # folder
		 'eng_folder':     proj_eng_dir,                                         # folder
		 'begin_epoch': 0,
		 'nepochs': 50, # 10 is recommended
		 'seed': 345,
		 'skipsave': -1, # don't save until 'skipsave' iterations
		 'savenum': 1000, # save model per sentences
		 #'savenum': 100000, # save model per sentences
		 'decay_lr_schedule': 5, # decay learning rate if the accuracy did not increase 
		 'decay_lr_rate': 0.5, # decay learning rate if the accuracy did not increase 
		 'savemodel': True}	 
	print param

	# Select Korean Or English
	if language == "kor":
		pass
	elif language == "eng":
		param['emb_file'] = param['eng_emb_file']
		param['train_data'] = param['eng_train_data']
		param['test_data'] = param['eng_test_data'] 
		param['folder'] = param['eng_folder']
	else:
		print "Error: Wrong choice of language " + language
		sys.exit(1)

	# load the dataset
	f = gzip.open(param['train_data'])
	train_src_sent_vec, train_tgt_sent_vec, vocab = cPickle.load(f)
	f = gzip.open(param['test_data'])
	test_src_sent_vec, test_tgt_sent_vec, vocab = cPickle.load(f)
	
	print 'train_src data size:', len(train_src_sent_vec), len(test_src_sent_vec)
	print 'train_tgt data size:', len(train_tgt_sent_vec), len(test_tgt_sent_vec)
	print 'test_src_sent[0]:', test_src_sent_vec[0]
	print 'test_tgt_sent[0]:', test_tgt_sent_vec[0]
	# En -> Jp
	train_x = train_src_sent_vec
	train_y = train_tgt_sent_vec
	test_x = test_src_sent_vec
	test_y = test_tgt_sent_vec
	word2idx = vocab
	folder = param['folder']

	if not os.path.exists(folder):
		os.mkdir(folder)

	idx2word = dict((k, v) for v, k in word2idx.iteritems())
	print 'Dic.'

	#import pdb; pdb.set_trace()

	# vocsize
	param['vocsize'] = len(word2idx)
	nsentences = len(train_x)

	print 'Size(voc,sent):', param['vocsize'], nsentences

	groundtruth_test = [map(lambda x: idx2word[x], y) for y in test_y]
	words_test = [map(lambda x: idx2word[x], w) for w in test_x]
	print 'ground truth data.'
	#print 'words_test[0]:', words_test[0]
	#print 'groundtruth_test[0]:', groundtruth_test[0]

	# instanciate the model
	numpy.random.seed(param['seed'])
	random.seed(param['seed'])

	if param['nn_type'] == 'GRU_encdec':
		nn = GRU_encdec.GRU_encdec(param, word2idx)
	elif param['nn_type'] == 'Stacked_GRU_encdec':
		nn = Stacked_GRU_encdec.Stacked_GRU_encdec(param, word2idx)
	elif param['nn_type'] == 'GRU_DO_search':
		nn = GRU_DO_search.GRU_DO_search(param, word2idx)
	else:
		print 'Error:', param['nn_type']
		sys.exit()
	print nn

	# train with early stopping on validation set
	best_test_ce = numpy.inf
	best_test_acc = 0
	no_acc_improvement_count = 0
	start_time = time.time()
	print 'training start.'
	for e in xrange(param['begin_epoch'], param['nepochs']):
		# shuffle
		if 'shuffle' in param and param['shuffle']:
			print 'Shuffle ...',
			sys.stdout.flush()
			myutil.shuffle([train_x, train_y], param['seed'])
			print 'Done.'
		# 적절히 조정 (없애거나...)
		#best_test_ce = numpy.inf
		#param['ce'] = e
		tic = time.time()
		tic2 = time.time()
		word_count = 0
		sum_nll = 0
		skip_count = 0
		for i, (x, y) in enumerate(zip(train_x, train_y)):
			# skip len(x)==1, becasue of error - by leeck
			#if len(x) == 1 or len(y) == 1:
				#print 'sent:', i, len(x), len(y)
				#skip_count += 1
				#continue
			word_count += len(y)
			[cost, nll] = nn.train(x, y, param['learning_method'], param['lr'])
			sum_nll += len(y) * nll
			print '[%i] %d'%(e, i+1),
			print 'CE=%.3f'%(sum_nll/word_count),
			print '%.1f(word/s)'%(word_count/(time.time()-tic2)),
			print '%.1f(sent/s)'%((i%param['savenum']+1)/(time.time()-tic2)),
			#print 'skip:%d'%(skip_count),
			print '%.1f(m) %.2f(h)\r'%((time.time()-tic2)/60, (time.time()-start_time)/3600),
			sys.stdout.flush()
			# evaluation // back into the real world : idx -> words
			# comment by yghwang
            #print i, param['savenum'], i + 1, len(train_x)

			# comment by yghwang
			if (i+1) % param['savenum'] == 0 or (i+2) == len(train_x):
			#if (i+1) % param['savenum'] == 0 or i == len(train_x):
				# comment by yghwang
                #print 'IN IF', i, param['savenum'], i + 1, train_x, train_y, len(train_x)
				sum_nll = 0
				word_count = 0
				# test set
				tic_test = time.time()
				test_word_count = 0
				test_ce = 0
				predictions_test = []
				for (x, y) in zip(test_x, test_y):
					[y_pred, test_nll] = nn.classify(x, y)
					test_word_count += len(y)
					test_ce += len(y) * test_nll
					predictions_test.append(map(lambda x: idx2word[x], y_pred))
				# evaluation // CE
				test_ce = test_ce / test_word_count
				# evaluation // accuracy
				test_acc = myutil.get_accuracy(predictions_test, groundtruth_test)
				test_acc = float('%.2f' % test_acc)
				print '\n(t: %.1f)'%(time.time()-tic_test),
				sys.stdout.flush()
				if skip_count > 0: print 'skip:%d'%(skip_count),

				if test_ce < best_test_ce:
					if param['savemodel'] and e > param['skipsave']:
						tic_save = time.time()
						nn.save(folder, param['nn_type'], 'BEST', prjt_name)
						print '(s: %.1f)'%(time.time()-tic_save),
						# write output
						file_name = folder+'/predict.%s.%s.%s.h%d.e%d.txt'%(param['nn_type'],param['activation'],param['learning_method'],param['nhidden'],param['emb_dimension'])
						f = open(file_name, 'w')
						for sent in predictions_test:
							for w in sent[:-1]: print >> f, w,
							print >> f
						f.close()
					best_test_ce = test_ce
					best_test_acc = test_acc
					param['bte'] = e
					no_acc_improvement_count = 0
					print 'New BEST Test CE: %.3f Acc: %.2f' % (test_ce, test_acc)
				else:
					no_acc_improvement_count += 1
		   			print 'Test CE: %.3f Acc: %.2f' % (test_ce, test_acc)
					if no_acc_improvement_count >= param['decay_lr_schedule']:
						old_lr = param['lr']
						param['lr'] *= param['decay_lr_rate']
		   				print 'Learning rate: %f -> %f' % (old_lr, param['lr'])
						no_acc_improvement_count = 0
				tic2 = time.time()
		print

	#print('BEST Test RESULT: epoch', param['bte'], 'best test CE', best_test_ce, 'Acc', best_test_acc)
	print('BEST Test RESULT: epoch', 'best test CE', best_test_ce, 'Acc', best_test_acc)

if __name__ == '__main__':
	if len(sys.argv) != 3:
		sys.stderr.write('Usage: ' + sys.argv[0] + ' project name & language (kor or eng)')
		sys.exit(1)
	prjt_name = str(sys.argv[1]).strip()
	language = str(sys.argv[2]).strip()
	main(prjt_name, language)

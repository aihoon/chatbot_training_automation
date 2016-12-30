#! /usr/bin/python
# -*- coding: UTF-8 -*-
# vi:ts=4:shiftwidth=4
# vim600:fdm=marker

import os
import cPickle, gzip
import random
import sys, time
import numpy
import codecs

# web server
import BaseHTTPServer
from subprocess import Popen, PIPE
import urllib

import theano

# by leeck
import myutil
import GRU_encdec
import Stacked_GRU_encdec
import GRU_DO_search

my_ip = "0.0.0.0"
my_port = "8888"

#theano.config.allow_gc=False
theano.config.floatX='float32'
theano.config.mode='FAST_COMPILE'

def word2char(sent):
	word = sent.split()
	result = ''
	for w in word:
		if w == '</s>':
			if result == '': result = w
			else: result += ' '+w
		else:
			unicode_w = unicode(w,'cp949')
			for i in range(len(unicode_w)):
				syl = unicode_w[i].encode('cp949')
				if i == 0:
					if result == '': result = syl+'/B'
					else: result += ' '+syl+'/B'
				else: result += ' '+syl+'/I'
	return result

def char2word(sent):
	word = sent.split()
	result = ''
    lex = ''    # HOON... Have to get confirmation from Dr. Hwang.
	for i, w in enumerate(word):
		if w == '</s>':
			result += w
			continue
		lex_tag = w.split('/')
		if len(lex_tag) == 2:
			lex, tag = lex_tag
			if i == 0: result += lex
			elif tag == 'B': result += ' '+lex
			elif tag == 'I': result += lex
		elif len(lex_tag) == 3:
			lex = '/'
			tag = lex_tag[2]
			if tag == 'B': result += ' '+lex
			elif tag == 'I': result += lex
		else: result += ' '+lex
	return result

class MTReqHandler(BaseHTTPServer.BaseHTTPRequestHandler):
	def do_GET(self):
		print 'header:'
		print self.headers

		print 'path:'
		print self.path

        global my_ip, my_port

		source_sentence = ''
		beam = 5
		ignore_unk = False
		if 'query=' in self.path:
			args = self.path.split('?')[1]
			args = args.split('&')
			for aa in args:
				cc = aa.split('=')
				if cc[0] == 'query':
					source_sentence = cc[1]
				if cc[0] == 'beam':
					beam = int(cc[1])
					#print 'beam:', beam
				if cc[0] == 'ignore_unk':
					if cc[1] == 'True': ignore_unk = True
					#print 'ignore_unk:', ignore_unk
		else:
			self.send_response(400)
			return

		source_sentence = urllib.unquote_plus(source_sentence)
		if source_sentence == '': source_sentence = '</s> </s>'

		#print 'query(cp949):', source_sentence
		print 'query(utf8):', source_sentence
		#source_sentence_949 = unicode(source_sentence,'utf8').encode('cp949')
		source_sentence2 = word2char(source_sentence)
		#print 'query2:', source_sentence2

		# 20161223 yghwang add for score issue
		#translation, alignment, nbest_translation = self.server.sampler.sample(source_sentence2, beam, ignore_unk)
		translation, alignment, nbest_translation, nbest_score = self.server.sampler.sample(source_sentence2, beam, ignore_unk)

		translation = char2word(translation)
		print 'answer(cp949):', translation
		print 'answer(utf8):', unicode(translation,'cp949').encode('utf8')

		self.send_response(200)
		self.send_header("Content-type", "text/html")
		self.end_headers()
		self.wfile.write('<meta http-equiv="Content-type" content="text/html; charset=CP949">\n')
		self.wfile.write('<h1>Chatbot</h1>\n')
		self.wfile.write('Parameters: '+str(beam)+' '+str(ignore_unk)+'<br>\n')
		self.wfile.write('<b>Query</b>: '+source_sentence+'<br>\n')
		self.wfile.write('<b>Answer</b>: '+translation+'<br><br>\n')
		self.wfile.write(alignment)
		self.wfile.write('<b>N-best</b>:\n')
		self.wfile.write('<ul>\n')


		
		# 20161223, yghwang add for score issue
		score_idx = 0
		for trans in nbest_translation:
			result = char2word(trans)
			oStr = '<li>' + result + '(' + str(nbest_score[score_idx]) + ')\n'
			self.wfile.write(oStr)
			score_idx = score_idx + 1

		#for trans in nbest_translation:
		#	result = char2word(trans)
		#	self.wfile.write('<li>'+result+'\n')


		self.wfile.write('</ul>\n')

		query_form = "<form method=\"GET\" action=\"http://" + my_ip + ":" + my_port + "\">" + \
                     """<textarea name="query" cols=80 rows=5>
                        </textarea>
                        <br>Beam:
                        <input type="radio" name="beam" value="20">20
                        <input type="radio" name="beam" value="10" checked>10
                        <input type="radio" name="beam" value="5" >5
                        <input type="submit" value="Submit">
                        </form>"""
		self.wfile.write(query_form)

class Sampler:
	def __init__(self, word2idx, idx2word, rnn_encoder_decoder, param):
		self.word2idx = word2idx
		self.idx2word = idx2word
		self.rnn_encoder_decoder = rnn_encoder_decoder
		self.param = param

	def sample(self, source_sentence, beam, ignore_unk):
		source_sentence += ' </s>'
		src_sent = source_sentence.split()
		x_idx_list = []
		for w in src_sent:
			if w in self.word2idx:
				x_idx_list.append(self.word2idx[w])
			else:
				x_idx_list.append(self.word2idx['UNK'])

		# 20161223 yghwang score issue
		#[y_idx_list, alignment, nbest_y_idx_list] = self.rnn_encoder_decoder.beam_search(x_idx_list, beam, ignore_unk)
		[ AAA, alignment, BBB] = self.rnn_encoder_decoder.beam_search_percentage(x_idx_list, beam, ignore_unk)
		print 'AAA', AAA
		y_idx_list = AAA[0]

		nbest_y_idx_list = []
		nbest_score = []
		for a in BBB:
			#print a[0], a[1]
			nbest_y_idx_list.append(a[0])
			nbest_score.append(a[1])
		
		#[y_idx_list, alignment, nbest_y_idx_list] = self.rnn_encoder_decoder.beam_search(x_idx_list, beam, ignore_unk)

		# 1-best
		best_tgt_sent = map(lambda x: self.idx2word[x].replace('</s>','EOS'), y_idx_list)
		translation = ' '.join(best_tgt_sent[:-1])
		# nbest
		nbest_translation = []
		for y_idx_list in nbest_y_idx_list:
			tgt_sent = map(lambda x: self.idx2word[x].replace('</s>','EOS'), y_idx_list)
			nbest_translation.append(' '.join(tgt_sent[:-1]))
		# alignment
		alignment_str = ''
		if len(alignment) > 0:
			alignment_str = '<table border="1">\n<tr><th>T\\S</th>\n'
			for j, w in enumerate(src_sent):
				alignment_str += '<th>'+w.replace('</s>','EOS')+'</th> '
			alignment_str += '</tr>\n'
			for w, a_list in zip(best_tgt_sent, alignment):
				alignment_str += '<tr><td><b>'+w+'</b></td>\n'
				for a in a_list:
					if a < 0.05: alignment_str += '<td></td>'
					elif a > 0.35: alignment_str += '<td><b>%.1f</b></td>'%a
					else: alignment_str += '<td>%.1f</td>'%a
				alignment_str += '</tr>\n'
			alignment_str += '</table>\n'

		# 20161223 yghwang add for score
		#return translation, alignment_str, nbest_translation
		return translation, alignment_str, nbest_translation, nbest_score


########################################################################

def main(prjt_name, language, param=None):

    proj_kor_dir = os.path.join(os.environ['HOME'], "Mindsbot", "Train.kor", prjt_name)
    proj_eng_dir = os.path.join(os.environ['HOME'], "Mindsbot", "Train.eng", prjt_name)
    data_dir     = os.path.join(os.environ['HOME'], "Mindsbot", "Data")

	if not param:
		param = {
		 #'nn_type': 'GRU_encdec', # GRU based encoder-decoder model (identity init.)
		 #'nn_type': 'Stacked_GRU_encdec', # Stacked GRU based encoder-decoder model (identity init.)
		 'nn_type': 'GRU_DO_search', # GRU based search model with Deep Output
		 #'activation': 'sigm', # sigmoid
		 'activation': 'tanh', # tanh
		 #'activation': 'relu', # ReLU
		 'reverse_input': False, # reverse input
		 'learning_method': 'sgd', # SGD
		 'weight_decay' : 1e-6, # weight decay - by leeck
		 'dropout_rate': [0, 0, 0], # dropout rate - by leeck
		 #'dropout_rate': [0, 0, 0.5], # dropout rate - by leeck
		 #'nhidden': 300, # number of hidden units
		 'nhidden': 500, # number of hidden units
		 #'nhidden': 1000, # number of hidden units
		 #'nhidden2': 300, # number of hidden units
		 'nhidden2': 500, # number of hidden units
		 #'nhidden2': 1000, # number of hidden units
		 #'emb_dimension': 50, # dimension of word embedding
		 ##'emb_dimension': 100, # dimension of word embedding
		 'emb_dimension': 200, # dimension of word embedding
		 #'emb_dimension': 400, # dimension of word embedding
		 'fixed_emb': False, # fixed embedding (error)
		 'gradient_clip': False, # gradient clipping
		 #'beam': 3, # beam size
		 #'beam': 5, # beam size
		 'beam': 10, # beam size (best)
		 #'beam': 20, # beam size (best1)
		 #'ignore_UNK': True, # ignore UNK (31.56 with penalty 1 and beam 5)
		 'ignore_UNK': False, # ignore UNK
		 ####'load_model': '../data/Project/' + prjt_name + '/' + prjt_name + '-GRU_DO_search.tanh.rmsprop.h500.e200.d0-0-0.wd1e-06.BEST.pkl.gz', # load parameters
		 ####'eng_load_model': '../eng_data/Project/' + prjt_name + '/' + prjt_name + '-GRU_DO_search.tanh.rmsprop.h500.e200.d0-0-0.wd1e-06.BEST.pkl.gz', # load parameters
		 ####'test_data': '../data/Project/' + prjt_name + '/' + prjt_name + '.test.pkl.gz', # kor test data
		 ####'eng_test_data': '../eng_data/Project/' + prjt_name + '/' + prjt_name + '.test.pkl.gz', # eng test data
		 ####'folder': '../data/Project/' + prjt_name, # folder
		 ####'eng_folder': '../eng_data/Project/' + prjt_name, # folder
		 'load_model':     proj_kor_dir + '/' + prjt_name + '-GRU_DO_search.tanh.rmsprop.h500.e200.d0-0-0.wd1e-06.BEST.pkl.gz', # load parameters
		 'eng_load_model': proj_eng_dir + '/' + prjt_name + '-GRU_DO_search.tanh.rmsprop.h500.e200.d0-0-0.wd1e-06.BEST.pkl.gz', # load parameters
		 'test_data':      proj_kor_dir + '/' + prjt_name + '.test.pkl.gz', # kor test data
		 'eng_test_data':  proj_eng_dir + '/' + prjt_name + '.test.pkl.gz', # eng test data
		 'folder':         proj_kor_dir, # folder
		 'eng_folder':     proj_eng_dir, # folder
		 'seed': 345}
	print param

	if language == "kor":
		pass
	elif language == "eng":
		param['load_model'] = param['eng_load_model']
		param['test_data'] = param['eng_test_data']
		param['folder'] = param['eng_folder']
	else:
		print "Wrong choice of language " + language
		sys.exit(1)
 
	# load the dataset
	f = gzip.open(param['test_data']); print 'test set:', param['test_data']
	test_src_sent_vec, test_tgt_sent_vec, vocab = cPickle.load(f)
	word2idx = vocab

	idx2word = dict((k, v) for v, k in word2idx.iteritems())

	param['vocsize'] = len(word2idx)
	print 'Size(voc):', param['vocsize']

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

	# web server
	server_address = ('', int(my_port))
	httpd = BaseHTTPServer.HTTPServer(server_address, MTReqHandler)
	sampler = Sampler(word2idx, idx2word, nn, param)
	httpd.sampler = sampler
	print 'Server starting..'
	httpd.serve_forever()
	
########################################################################
########################################################################
########################################################################

if __name__ == '__main__':

	if len(sys.argv) != 5:
		sys.stderr.write('Usage: ' + sys.argv[0] + ' ProjectName LANGUAGE IP PORT')
        sys.exit()

	prjt_name = str(sys.argv[1]).strip()
	language  = str(sys.argv[2]).strip()
    my_ip     = str(sys.argv[3]).strip()
    my_port   = str(sys.argv[4]).strip()

	#language = str(sys.argv[2]).strip()
	main(prjt_name, language)
'''
    if len(sys.argv) != 3:
        sys.stderr.write('Usage: ' + sys.argv[0] + ' ProjectName & Language (kor or eng)')
'''

#! /usr/bin/python
# -*- coding: UTF-8 -*-
# vi:ts=4:tw=78:shiftwidth=4
# vim600:fdm=marker
"""
author : Changki Lee <leeck@kangwon.ac.kr>
"""

__author__ = 'Changki Lee'
__version__ = '2015-10-8'

import sys, re
import cPickle
from optparse import OptionParser

# Usage and Options {{{
usage = "usage: %prog -v vocab_data -s source_data -t target_data -o output_file"
parser = OptionParser(usage)
parser.add_option("-v", "--vocab", type="string", help="Vocab. data")
parser.add_option("-s", "--src", type="string", help="Source data")
parser.add_option("-t", "--tgt", type="string", help="Target data")
parser.add_option("-o", "--output", type="string", help="output file")
(options, args) = parser.parse_args()
# }}}

if not (options.src and options.tgt and options.output):
	parser.print_usage()
	sys.exit(1)

# get vocab(word,id) from word2vec
print >> sys.stderr, "Source word2vec ...",
vocab = {}
i = 0
if options.vocab:
	f = open(options.vocab)
else:
	f = open('vocab_word.txt')

for line in f:
	line = line.replace('\n', '')
	word = line.split()
	if len(word) == 1 or len(word) == 2:
		if word[0] in vocab: print >> sys.stderr, "Warning(exist):", word[0]
		vocab[word[0]] = i
        i += 1
    else:
        print >> sys.stderr, "Warning:", line
print >> sys.stderr, "done:", i, len(vocab)

# read data file
def read_data(file_name, vocab):
	print >> sys.stderr, 'Reading data file:', file_name
	infile = open(file_name)
	sentence_vec = []
	for line in infile:
		line = line.replace('\n', '')
		word = line.split()
		if len(word) == 0: continue
	    sentence = []
        for w in word:
            if w in vocab:
		        sentence.append(vocab[w])
            else:
				print >> sys.stderr, 'UNK:', w,
		        sentence.append(vocab['UNK'])
		sentence.append(vocab['</s>'])
		sentence_vec.append(sentence)
	print >> sys.stderr, "Done:", len(sentence_vec)
	return sentence_vec

# main
src_sent_vec = read_data(options.src, vocab)
tgt_sent_vec = read_data(options.tgt, vocab)
# test
print 'src_sent_vec[0]:', src_sent_vec[0]
print 'tgt_sent_vec[0]:', tgt_sent_vec[0]
# write
f = open(options.output, 'wb')
cPickle.dump((src_sent_vec, tgt_sent_vec, vocab), f)
f.close()


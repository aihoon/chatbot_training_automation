#! /usr/bin/python
# -*- coding: UTF-8 -*-
# vi:ts=4:shiftwidth=4
# vim600:fdm=marker

# 1음절씩 (B/I 둘로 구분)

import sys, time, codecs
from optparse import OptionParser

def main():
	# Usage and Options {{{
	usage = "usage: %prog -d [options]"
	parser = OptionParser(usage)
	parser.add_option("-d", "--data", type="string", help="deep_learning data file")
	(options, args) = parser.parse_args()
	# }}}
	if not options.data:
		parser.print_usage()
		sys.exit(1)

	infile = open(options.data, 'r')
	for line in infile:
		line = line.replace('\n','')
		word = line.split()
		for w in word:
			if w == '</s>':
				print w,
			else:
				try:
					unicode_w = unicode(w,'cp949')
				except UnicodeDecodeError:
					print(line + "\n")
				for i in range(len(unicode_w)):
					syl = unicode_w[i].encode('cp949')
					if i == 0: print syl+'/B',
					else: print syl+'/I',
		print
if __name__ == "__main__":
    main()


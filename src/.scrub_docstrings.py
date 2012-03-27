#!/usr/bin/env python

import sys
import os

# has to be ordered tuple
REPLACEMENTS=[
	[' float64_t', ' float'],
	[' float32_t', ' float'],
	[' float128_t', ' float'],
	[' double', ' float'],
	[' int32_t', '  int'],
	[' uint32_t', ' int'],
	[' int64_t', ' int'],
	[' uint64_t', ' int'],
	[' uint8_t', ' int'],
	[' int16_t', ' int'],
	[' uint16_t', ' int'],
	[' char', ' str'],
	['-> CFeatures', '-> Features'],
]

class Scrub:
	def __init__ (self, extension, input):
		self.infile=input+extension
		self.outfile=self.infile+'.new'

	def run (self):
		input=open(self.infile, 'r')
		output=open(self.outfile, 'w')

		if sys.version_info >= (3,):
			for text in input.readlines():
				for pairs in REPLACEMENTS:
					text=text.replace(pairs[0], pairs[1])
				output.write(text)
		else:
			for text in input.xreadlines():
				for pairs in REPLACEMENTS:
					text=text.replace(pairs[0], pairs[1])
				output.write(text)

		input.close()
		output.close()

		os.rename(self.outfile, self.infile)


if __name__=='__main__':
	offset=sys.argv[2].find('_wrap.cxx')
	if sys.argv[1] == 'stop':
		sys.exit(0)
	scrubby=Scrub(sys.argv[1], sys.argv[2][0:offset])
	scrubby.run()

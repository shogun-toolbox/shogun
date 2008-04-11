#!/usr/bin/env python

import sys
import os

# has to be ordered tuple
REPLACEMENTS=[
	[' DREAL', ' float'],
	[' SHORTREAL', ' float'],
	[' LONGREAL', ' float'],
	[' double', ' float'],
	[' INT', '  int'],
	[' UINT', ' int'],
	[' LONG', ' int'],
	[' ULONG', ' int'],
	[' BYTE', ' int'],
	[' SHORT', ' int'],
	[' WORD', ' int'],
	[' CHAR', ' str'],
	['-> C', '-> '],
]

class Scrub:
	def __init__ (self, extension, input):
		self.infile=input+extension
		self.outfile=self.infile+'.new'

	def run (self):
		input=open(self.infile, 'r')
		output=open(self.outfile, 'w')

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

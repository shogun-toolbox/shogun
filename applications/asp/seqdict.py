import string

class predictions(object):
	def __init__(self, positions=None, scores=None):
		self.positions=positions
		self.scores=scores

	def set_positions(self, positions):
		self.positions=positions;
	def get_positions(self):
		return self.positions

	def set_scores(self, scores):
		self.scores=scores
	def get_scores(self):
		return self.scores

	def __str__(self):
		return 'positions: ' + `self.positions` + 'scores: ' + `self.scores`
	def __repr__(self):
		return self.__str__()

class sequence(object):
	def __init__(self, name, seq, (start,end)):
		assert(start<end<len(seq))
		self.start=start
		self.end=end
		self.name=name
		self.seq=seq
		self.preds=dict()
		self.preds['acceptor']=predictions()
		self.preds['donor']=predictions()

	def __str__(self):
		s="start:" + `self.start`
		s+=" end:" + `self.end`
		s+=" name:" + `self.name`
		s+=" sequence:" + `self.seq[0:10]`
		s+="... preds:" + `self.preds`
		return s
	def __repr__(self):
		return self.__str__()

def seqdict(dic, (start,end)):
	""" takes a fasta dict as input and
	generates a list of sequence objects from it """

	sequences=list()

	#translate string to ACGT / all non ACGT letters are mapped to A
	tab=''
	for i in xrange(256):
		if chr(i).upper() in 'ACGT':
			tab+=chr(i).upper()
		else:
			tab+='A'

	for seqname in dic.ordered_items():
		seq=string.translate(seqname[1], tab)
		seq=seq.upper()
		if end<0:
			stop=len(seq)+end
		else:
			stop=end

		sequences.append(sequence(seqname[0], seq, (start,stop)))

	return sequences

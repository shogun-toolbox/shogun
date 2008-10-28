import numpy
from shogun.Features import DNA,Labels,StringCharFeatures,StringWordFeatures,CombinedFeatures,StringUlongFeatures,FULL_NORMALIZATION
from shogun.Kernel import CombinedKernel,WeightedDegreePositionStringKernel,CommWordStringKernel 
from shogun.PreProc import SortWordString
from shogun.Classifier import SVMLight

class sensor(object):
	"""
	sensor has window (left,center,right) of length right-left+1
	with center at "center"
	"""

	def __init__(self, (left, center, right), kernel, train_data):
		self.kernel=kernel
		self.location=(left, center, right)
		self.train_data=train_data
		self.test_data=None

		self.setup_objects()

	def setup_objects(self):
		kname=sel.kernel['name']
		if  kname == 'spectrum':
			wf=StringWordFeatures(f.get_alphabet())
			wf.obtain_from_char(f, s.degree-1, s.degree, 0, False)
			del f
			if mode == "train":
				pre = SortWordString()
				pre.init(wf)
				wf.add_preproc(pre)
			ret = wf.apply_preproc()
			features.append_feature_obj(wf)

		elif kname in ('wdshift', wd):
			if mode == "train":
				s.train_feature=f
			else:
				s.test_feature=f
			features.append_feature_obj(f)
		else:
			raise "Currently, only WDS and SPEC kernels supported"

		return features 

class signal_sensor(object):
	"""
	A collection of sensors
	"""
	def __init__(self, sequence):

		self.sensors=list()
		self.kernel=CombinedKernel()
		self.features=CombinedFeatures()

	def parse_file(file):
		m=model()
		
		l=file.readline();

		if l != '%arts version: 1.0\n':
			sys.stderr.write("\nfile not a asplicer definition file\n")
			return None

		svm=dict()
				
		while l:
			if not ( l.startswith('%') or l.startswith('\n') ): # comment
				key,value=l.split('=')

				if not '[' in value:
					if '.' in value:
						svm[key]=float(value)
					else:
						try:
							svm[key]=int(value)
						except:
							svm[key]=value



			l=file.readline()

		sys.stderr.write('done\n')
		return m

	def init_kernel(self, kernel_cache_size=5000):
		kernel=CombinedKernel()
		kernel.set_cache_size(int(kernel_cache_size))

		for s in self.sensors:

			k=None

			if s.kernel_type == 'SPEC':
				k = CommWordStringKernel(s.train_feature, s.train_feature)
				k.set_use_dict_diagonal_optimization(s.degree<8)
			elif s.kernel_type == 'WD':
				k = WeightedDegreePositionStringKernel(s.train_feature, s.train_feature, s.degree)
				k.set_shifts( s.shift * numpy.ones(s.train_feature.get_max_vector_length(), dtype=numpy.int32) )
			if not k:
				raise "Only WDS & SPEC kernel types supported"

			s.kernel=k
			kernel.append_kernel(k)

		return kernel

	def train_svm(self, C, num_threads):
		self.kernel.init(self.train_features, self.train_features)
		self.kernel.parallel.set_num_threads(num_threads)

		self.svm = SVMLight(C, self.kernel, self.label)
		self.svm.parallel.set_num_threads(num_threads)
		self.svm.set_batch_computation_enabled(True)
		self.svm.set_linadd_enabled(True)
		self.svm.set_epsilon(1e-4)
		self.svm.train()
		self.svm.set_batch_computation_enabled(False)

	def optimize_svm(self):

		print "Starting kernel optimization."
		self.kernel.delete_optimization()
		self.kernel.init_optimization_svm(self.svm)
		print "Done."

	def apply_svm(self, seqs):
		self.test_features = self.init_features(seqs, "test")
		print "Start kernel initialization."
		self.kernel.init(self.train_features, self.test_features)
		print "Done."
		return self.svm.classify().get_labels()

	def slide_svm_over_string(self, seq, positions):
		self.test_features = self.init_features_seq(seq, positions)
		print "Start kernel initialization."
		self.kernel.init(self.train_features, self.test_features)
		print "Done."

		#outputs = numpy.zeros(len(positions), numpy.double)
		#for i in xrange(0,len(positions)):
		#    if numpy.mod(i,1000)==0:
		#        print i
		#    #outputs[i] = self.kernel.compute_optimized(i) + self.svm.get_bias()
		#return outputs

		return self.svm.classify().get_labels()


		self.sensors=sensors
		self.label=Labels(numpy.array(train_label, numpy.double))
		self.train_features=self.init_features(train_data, "train")
		self.kernel=self.init_kernel()

		self.kernel.init(self.train_features, self.train_features)
		self.kernel.parallel.set_num_threads(svm_num_threads)

		self.svm = SVMLight(svm_C, self.kernel, self.label)
		self.svm.parallel.set_num_threads(svm_num_threads)
		self.svm.set_batch_computation_enabled(False)
		self.svm.set_linadd_enabled(True)

		self.svm.create_new_model(len(sv_alphas))
		self.svm.set_support_vectors(numpy.array(xrange(0,len(sv_alphas)), numpy.int32))
		self.svm.set_alphas(numpy.array(sv_alphas,numpy.double))
		self.svm.set_bias(svm_bias)

#!/usr/bin/env python
data = '../data/fm_train_real.dat'
parameter_list = [[data,10],[data,20]]

def converter_diffusionmaps_modular (data_fname,t):
	try: 
		from modshogun import RealFeatures, DiffusionMaps, GaussianKernel, CSVFile
		
		features = RealFeatures(CSVFile(data_fname))
			
		converter = DiffusionMaps()
		converter.set_target_dim(1)
		converter.set_kernel(GaussianKernel(10,10.0))
		converter.set_t(t)
		converter.apply(features)

		return features
	except ImportError:
		print('No Eigen3 available')

if __name__=='__main__':
	print('DiffusionMaps')
	converter_diffusionmaps_modular(*parameter_list[0])


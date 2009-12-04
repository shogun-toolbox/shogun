def combined_custom():
	from shogun.Features import CombinedFeatures, RealFeatures
	from shogun.Kernel import CombinedKernel, PolyKernel, CustomKernel

	kernel=CombinedKernel()
	feats_train=CombinedFeatures()

        tfeats = RealFeatures(fm_train_real)
        tkernel = PolyKernel(10,3)
        tkernel.init(tfeats, tfeats)
        K = tkernel.get_kernel_matrix()
        kernel.append_kernel(CustomKernel(K))
        
        subkfeats_train = RealFeatures(fm_train_real)
	feats_train.append_feature_obj(subkfeats_train)
        subkernel = PolyKernel(10,2)
        subkernel.init(subkfeats_train, subkfeats_train)
        kernel.append_kernel(subkernel)
        
        print kernel.get_kernel_matrix()

if __name__=='__main__':
	from tools.load import LoadMatrix
	lm=LoadMatrix()
	fm_train_real=lm.load_numbers('../data/fm_train_real.dat')
	combined_custom()

# this was trancekoded by the awesome trancekoder
# ...and fixifikated by the awesum fixifikator
require 'modshogun'
require 'pp'

parameter_list = [[1,7],[2,8]]

def classifier_custom_kernel_modular(c=1,dim=7)

	Modshogun::Math.init_random(c)

	lab=sign(2*rand(dim) - 1)
	data=rand(dim, dim)
	symdata=data*data.T + diag(ones(dim))
    
# *** 	kernel=CustomKernel()
	kernel=Modshogun::CustomKernel.new
	kernel.set_features()
	kernel.set_full_kernel_matrix_from_full(data)
# *** 	labels=Labels(lab)
	labels=Modshogun::Labels.new
	labels.set_features(lab)
# *** 	svm=LibSVM(c, kernel, labels)
	svm=Modshogun::LibSVM.new
	svm.set_features(c, kernel, labels)
	svm.train()
	predictions =svm.apply() 
	out=svm.apply().get_labels()
	return svm,out

end

if __FILE__ == $0
	puts 'custom_kernel'
	classifier_custom_kernel_modular(*parameter_list[0])
end

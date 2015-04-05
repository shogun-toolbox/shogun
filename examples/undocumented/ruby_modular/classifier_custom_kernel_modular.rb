require 'rubygems'
require 'modshogun'
require 'load'
require 'narray'
require 'pp'

parameter_list = [[1,7],[2,8]]

def classifier_custom_kernel_modular(c=1,dim=7)

	Modshogun::Math.init_random(c)
	NArray.srand(17)

	lab = (2*NArray.float(dim).random! - 1).sign
	pp lab
	data= NMatrix.float(dim, dim).random!
	symdata=data*data.transpose + 10*NMatrix.float(dim,dim).unit

	kernel=Modshogun::CustomKernel.new
	kernel.set_full_kernel_matrix_from_full(data)
	labels=Modshogun::BinaryLabels.new(lab)
	svm=Modshogun::LibSVM.new(c, kernel, labels)
	svm.train()
	predictions =svm.apply()
	out=svm.apply().get_labels()
	return svm,out

end

if __FILE__ == $0
	puts 'custom_kernel'
	pp classifier_custom_kernel_modular(*parameter_list[0])
end

require 'nmatrix'
require 'modshogun'
require 'pp'

require_relative 'load'

parameter_list = [[1,7],[2,8]]

Numeric.class_eval do
  def sign
    return -1 if self < 0
    return 0 if self == 0
    return 1 if self > 0
  end
end

module ArrayHelpers
  def sign
    a = []
    self.each do |x|
      a << x.sign
    end
    a
  end
end

NMatrix.class_eval do
  include ArrayHelpers
end


def classifier_custom_kernel_modular(c=1,dim=7)

	Modshogun::Math.init_random(c)

	lab = (NMatrix.random([1,dim])-0.5).sign
	data= NMatrix.random([dim, dim])
	symdata=data*data.transpose + NMatrix.eye(dim)
    
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

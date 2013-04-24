from numpy import *
from tools.generate_circle_data import circle_data
cir=circle_data()
number_of_points_for_circle1=42
number_of_points_for_circle2=122
row_vector=2
data=cir.generate_data(number_of_points_for_circle1,number_of_points_for_circle2,row_vector)
d=zeros((row_vector,number_of_points_for_circle1))
d2=zeros((row_vector,number_of_points_for_circle2)) 
d=[data[i][0:number_of_points_for_circle1] for i in range(0,row_vector)]
d2=[data[i][number_of_points_for_circle1:(number_of_points_for_circle1+number_of_points_for_circle2)] for i in range(0,row_vector)]	
parameter_list = [[data,0.01,1.0], [data,0.05,2.0]]
def preprocessor_kernelpca_modular (data, threshold, width):
	
	from shogun.Features import RealFeatures
	from shogun.Preprocessor import KernelPCA
	from shogun.Kernel import GaussianKernel
	features = RealFeatures(data)
	kernel=GaussianKernel(features,features,width)
	preprocessor=KernelPCA(kernel)
	preprocessor.init(features)
	preprocessor.set_target_dim(2)
	#X=preprocessor.get_transformation_matrix()
	X2=preprocessor.apply_to_feature_matrix(features)
	lx0=len(X2)
	modified_d1=zeros((lx0,number_of_points_for_circle1))
	modified_d2=zeros((lx0,number_of_points_for_circle2))
	modified_d1=[X2[i][0:number_of_points_for_circle1] for i in range(lx0)]
	modified_d2=[X2[i][number_of_points_for_circle1:(number_of_points_for_circle1+number_of_points_for_circle2)] for i in range(lx0)]
	return features

if __name__=='__main__':
	print('KernelPCA')
	preprocessor_kernelpca_modular(*parameter_list[0])

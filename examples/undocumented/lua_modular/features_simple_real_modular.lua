require 'shogun'

matrix = {{1,2,3},{4,0,0},{0,0,0},{0,5,0},{0,0,6},{9,9,9}}

parameter_list = {{matrix}}

function features_simple_real_modular(A)
	a=RealFeatures(A)
	a:set_feature_vector({1,4,0,0,0,9}, 0)
    
	a_out = a:get_feature_matrix()
    
	return a_out
end

print 'simple_real'
features_simple_real_modular(unpack(parameter_list[1]))

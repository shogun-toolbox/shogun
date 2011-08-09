# this was trancekoded by the awesome trancekoder
require 'narray'
require 'modshogun'
require 'load'
require 'pp'

parameter_list = [[0,2,0,False],[0,3,0,False]]

def features_string_ulong_modular(start=0,order=2,gap=0,rev=False)
    
    

end
#create string features
    cf=StringCharFeatures(['hey','guys','string'], RAWBYTE)
    uf=StringUlongFeatures(RAWBYTE)
    
    uf.obtain_from_char(cf, start,order,gap,rev)
    
#replace string 0
    uf.set_feature_vector(array([1,2,3,4,5], dtype=uint64), 0)
    

    return uf.get_features(),uf.get_feature_vector(2), uf.get_num_vectors()

if __FILE__ == $0
    print 'simple_longint'
    features_string_ulong_modular(*parameter_list[0])

end

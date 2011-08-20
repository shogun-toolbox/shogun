# this was trancekoded by the awesome trancekoder
# ...and fixifikated by the awesum fixifikator
require 'modshogun'
require 'pp'

# create dense matrices A,B,C

matrixA=array([[1,2,3],[4,0,0],[0,0,0],[0,5,0],[0,0,6],[9,9,9]], dtype=float64)
matrixB=array([[1,2,3],[4,0,0],[0,0,0],[0,5,0],[0,0,6],[9,9,9]], dtype=int64)
matrixC=array([[1,2,3],[4,0,0],[0,0,0],[0,5,0],[0,0,6],[9,9,9]], dtype=uint8)

# ... of type Real, LongInt and Byte
parameter_list = [[matrixA,matrixB,matrixC]]

def features_simple_modular(A=matrixA,B=matrixB,C=matrixC)

# ***     a=RealFeatures(A)
    a=Modshogun::RealFeatures.new
    a.set_features(A)
# ***     b=LongIntFeatures(B)
    b=Modshogun::LongIntFeatures.new
    b.set_features(B)
# ***     c=ByteFeatures(C)
    c=Modshogun::ByteFeatures.new
    c.set_features(C)
    

end
# or 16bit wide ...
#feat1 = f.ShortFeatures(N.zeros((10,5),N.short))
#feat2 = f.WordFeatures(N.zeros((10,5),N.uint16))


#	puts some statistics about a

# get first feature vector and set it

    a.set_feature_vector(array([1,4,0,0,0,9], dtype=float64), 0)

# get matrices
    a_out = a.get_feature_matrix()
    b_out = b.get_feature_matrix()
    c_out = c.get_feature_matrix()

    assert(all(a_out==A))

    assert(all(b_out==B))

    assert(all(c_out==C))
    return a_out,b_out,c_out,a,b,c

if __FILE__ == $0
	puts 'simple'
    features_simple_modular(*parameter_list[0])

end

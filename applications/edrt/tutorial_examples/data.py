import numpy

def swissroll(N=1000):
	tt = numpy.array((5*numpy.pi/4)*(1+2*numpy.random.rand(N)))
	height = numpy.array((numpy.random.rand(N)-0.5))
	noise = 0.0
	X = numpy.array([(tt+noise*numpy.random.randn(N))*numpy.cos(tt), 10*height, (tt+noise*numpy.random.randn(N))*numpy.sin(tt)])
	return X


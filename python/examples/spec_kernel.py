from shogun.Kernel import SpectrumKernel
from shogun.Features import StringCharFeatures
from shogun.Features import Alphabet,DNA

alpha=Alphabet(DNA)
x=StringCharFeatures(alpha)
y=StringCharFeatures(alpha)

#k=SpectrumKernel(10)
k=SpectrumKernel(x,y,10)

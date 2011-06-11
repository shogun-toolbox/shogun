#!/usr/bin/env ruby

## copy from ../examples/undocumented/python_modular/classifier_libsvm_minimal_modular.py

require '../../../src/ruby_modular/Features'
require '../../../src/ruby_modular/Classifier'
require '../../../src/ruby_modular/Kernel'
# for randn func
require '../../../src/ruby_modular/Library'

include Features
include Classifier
include Kernel

# helper methods for all this fun stuff

def gen_ones_vec
  ary = []
  @num.times do
    ary << -1
  end
  @num.times do
    ary << 1
  end
  return ary
end

# for random numbers:
def randn
  Library::Math.randn_double
end

# 2 high, num wide random arrays, concatenated together [] + []
# randn - dist, randn + dist
def gen_rand_ary
  ary = [[],[]]
  ary.each do |p|
    p << ary_fill( @dist ) + ary_fill( -@dist )
    p.flatten!
  end
  return ary
end

def ary_fill dist
  ary = []
  @num.times do
    ary << randn + dist
  end
  return ary
end

Numeric.class_eval do
  def sign
    return -1 if self < 0
    return 0 if self == 0
    return 1 if self > 0
  end
end

Array.class_eval do
  def sign
    a = []
    self.each do |x|
      a << x.sign
    end
    a
  end

  def eql_items? other
    raise(ArgumentError, "Argument is not an Array") unless other.kind_of? Array
    raise(ArgumentError, "Arrays dont' have the same number of elements") if self.size != other.size
    output = []
    self.each_with_index do |x, i|
      output[i] = x == other[i]
    end
    return output
  end
end

# yes this is a very mean & dirty mean alg...
# no checking of stuff, can give wonky errors, but hey!
# i'm being lazy & not doing cool metaprogramming type stuff for now...
def mean ary
  num_items = ary.size
  tot = ary.inject(0) do |sum, n|
    if n == true
      sum + 1
    elsif (n == false) or (n == nil)
      sum + 0 # yes it's a dummy
    else
      sum + n
    end
  end
  tot.to_f / num_items.to_f
end
  
# the actual example
# example nums
@num = 1000
@dist = 1
@width = 2.1
C = 1

puts "generating training data"
traindata_real = gen_rand_ary
testdata_real = gen_rand_ary

puts "generating labels"
trainlab = gen_ones_vec
testlab = gen_ones_vec

puts "doing feature stuff"
feats_train = RealFeatures.new
feats_train.set_feature_matrix traindata_real
feats_test = RealFeatures.new
feats_test.set_feature_matrix testdata_real
kernel = GaussianKernel.new feats_train, feats_train, @width

puts "labeling stuff"
labels = Labels.new
labels.set_labels trainlab
svm = LibSVM.new C, kernel, labels
svm.train

puts "the grand finale"
kernel.init feats_train, feats_test
out = svm.apply.get_labels
testerr = mean out.sign.eql_items? testlab
puts testerr

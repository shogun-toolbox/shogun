require 'rubygems'
require 'modshogun'
require 'load'
require 'narray'

@num = 1000
@dist = 1
@width = 2.1
C = 1

puts "generating training data"
traindata_real = gen_rand_ary @num
testdata_real = gen_rand_ary @num

puts "generating labels"
trainlab = gen_ones_vec @num
testlab = gen_ones_vec @num

puts "doing feature stuff"
feats_train = Modshogun::RealFeatures.new
feats_train.set_feature_matrix traindata_real
feats_test = Modshogun::RealFeatures.new
feats_test.set_feature_matrix testdata_real
kernel = Modshogun::GaussianKernel.new feats_train, feats_train, @width

puts "labeling stuff"
labels = Modshogun::BinaryLabels.new
labels.set_labels trainlab
svm = Modshogun::LibSVM.new C, kernel, labels
svm.train

puts "the grand finale"
kernel.init feats_train, feats_test
out = svm.apply.get_labels
testerr = mean out.sign.eql_items? testlab
puts testerr

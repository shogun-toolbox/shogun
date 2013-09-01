require 'nmatrix'
require 'modshogun'
require 'pp'

require_relative 'load'



data = LoadMatrix.load_numbers('../data/fm_train_real.dat')
puts data

features = Modshogun::RealFeatures.new
features.set_feature_matrix(data)

puts features.get_feature_matrix()
puts " "


lbls = LoadMatrix.load_labels('../data/label_train_twoclass.dat')
puts lbls

labels = Modshogun::BinaryLabels.new
labels.set_labels(lbls)

puts labels.get_labels()
puts " "


lbls = NMatrix.ones([1,10])
puts lbls

labels = Modshogun::BinaryLabels.new
labels.set_labels(lbls)

puts labels.get_labels()

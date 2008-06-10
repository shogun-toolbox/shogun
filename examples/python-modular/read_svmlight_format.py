from shogun.Features import SparseRealFeatures
f=SparseRealFeatures()
lab=f.load_svmlight_file('../data/train_sparsereal.light')
f.write_svmlight_file('testwrite.light', lab)
